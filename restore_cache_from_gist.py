"""從 GitHub Gist 還原 stock_data_cache.pkl

新電腦 setup 時用：
  cd C:\\stock-evolution
  python restore_cache_from_gist.py

機制：
  1. 從 Gist a300b9e29372ac76f79eda39a2a86321 拉所有 chunk_*.b64
  2. 按 manifest.n_chunks 順序拼接
  3. base64 decode → zlib decompress → 寫入 stock_data_cache.pkl
  4. 驗證：讀回 pkl，print 檔數 + 末日

跟 backup_cache_to_gist.py 是配對的。
"""
import os, sys, pickle, base64, zlib, json, urllib.request

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
GIST_ID = "a300b9e29372ac76f79eda39a2a86321"

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
if not GH_TOKEN:
    print("⚠️ 沒設 GH_TOKEN，用匿名 read（公開 Gist 才行）")
    GH_TOKEN = None


def fetch_gist_meta():
    headers = {}
    if GH_TOKEN:
        headers["Authorization"] = f"token {GH_TOKEN}"
    req = urllib.request.Request(f"https://api.github.com/gists/{GIST_ID}", headers=headers)
    r = urllib.request.urlopen(req, timeout=60)
    return json.loads(r.read())


def main():
    if os.path.exists(CACHE_PATH):
        print(f"⚠️ {CACHE_PATH} 已存在")
        ans = input("覆蓋？[y/N]: ").strip().lower()
        if ans != "y":
            print("中止")
            sys.exit(0)
        # 備份舊版
        backup = CACHE_PATH + ".before_restore"
        os.replace(CACHE_PATH, backup)
        print(f"  舊 cache 改名為 {backup}")

    print(f"[1/4] 從 Gist {GIST_ID} 抓 manifest...")
    gist = fetch_gist_meta()
    files = gist["files"]
    if "manifest.json" not in files:
        print("❌ Gist 沒 manifest.json，可能是舊版備份格式")
        sys.exit(1)

    # Manifest 也可能 truncated（Gist API 對 > 1 MB 截斷）→ 從 raw_url 拉
    mf_meta = files["manifest.json"]
    if mf_meta.get("truncated"):
        req = urllib.request.Request(mf_meta["raw_url"])
        if GH_TOKEN:
            req.add_header("Authorization", f"token {GH_TOKEN}")
        manifest_str = urllib.request.urlopen(req, timeout=60).read().decode()
    else:
        manifest_str = mf_meta["content"]
    manifest = json.loads(manifest_str)
    n_chunks = manifest["n_chunks"]
    print(f"  format: {manifest['format']}")
    print(f"  n_chunks: {n_chunks}")
    print(f"  raw_size: {manifest['raw_size'] / 1024 / 1024:.1f} MB")
    print(f"  backup_time: {manifest.get('backup_time', 'N/A')}")

    print(f"[2/4] 拉 {n_chunks} chunks...")
    parts = []
    for i in range(n_chunks):
        fname = f"chunk_{i:03d}.b64"
        if fname not in files:
            print(f"  ❌ 缺 {fname}")
            sys.exit(1)
        f_meta = files[fname]
        # Gist API 對大 chunks 可能截斷，要從 raw_url 拉
        if f_meta.get("truncated"):
            req = urllib.request.Request(f_meta["raw_url"])
            if GH_TOKEN:
                req.add_header("Authorization", f"token {GH_TOKEN}")
            content = urllib.request.urlopen(req, timeout=60).read().decode()
        else:
            content = f_meta["content"]
        parts.append(content)
        if (i + 1) % 10 == 0:
            print(f"  進度 {i+1}/{n_chunks}")

    encoded = "".join(parts)
    print(f"  總 base64 大小: {len(encoded) / 1024 / 1024:.1f} MB")

    print(f"[3/4] 解碼 + 解壓...")
    compressed = base64.b64decode(encoded)
    raw = zlib.decompress(compressed)
    print(f"  解壓後: {len(raw) / 1024 / 1024:.1f} MB")

    print(f"[4/4] 寫入 {CACHE_PATH}...")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, CACHE_PATH)

    # 驗證
    print()
    print("=== 驗證 ===")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    from collections import Counter
    last_days = Counter(df.index[-1].date() for df in cache.values() if len(df))
    print(f"  ✅ 還原 {len(cache)} 檔")
    print(f"  最後一天分布:")
    for dt, n in sorted(last_days.items())[-5:]:
        print(f"    {dt}: {n} 檔")
    print(f"  ✅ 還原完成，可以開始跑 pipeline")


if __name__ == "__main__":
    main()
