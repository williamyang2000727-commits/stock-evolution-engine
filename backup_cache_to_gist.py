"""備份 stock_data_cache.pkl 到 GitHub Gist（b64 壓縮版）

用途：
  - 新電腦搬遷前，從 Windows 推一份最新 cache 到雲端
  - 萬一 Windows 硬碟壞掉，可從 Gist 還原（restore_cache_from_gist.py）

機制：
  1. pickle dump cache → bytes
  2. zlib compress（179 MB → 預估 30-50 MB）
  3. base64 encode（變成可貼進 Gist 的純文字）
  4. PATCH Gist a300b9e29372ac76f79eda39a2a86321

注意：
  - GitHub Gist 單檔 1 MB 純文字限制 → 必須分檔
  - 預估壓縮+b64 後 ~50 MB → 切 50 個檔案 chunk_01.b64 ... chunk_50.b64
  - 每檔 < 1 MB

Windows 用法：
  cd C:\\stock-evolution
  python backup_cache_to_gist.py
"""
import os, sys, pickle, base64, zlib, json, urllib.request, time

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
GIST_ID = "a300b9e29372ac76f79eda39a2a86321"
CHUNK_SIZE = 800_000  # 800 KB per chunk (< Gist 1 MB limit)

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
if not GH_TOKEN:
    print("❌ 請先設環境變數：$env:GH_TOKEN = 'ghp_xxx...'")
    sys.exit(1)


def patch_gist(gist_id, files_dict):
    """PATCH Gist 多檔，files_dict = {filename: content_str}"""
    payload = {"files": {fn: {"content": c} for fn, c in files_dict.items()}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        data=data, method="PATCH",
        headers={"Authorization": f"token {GH_TOKEN}",
                 "Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=120)
    return r.status


def main():
    if not os.path.exists(CACHE_PATH):
        print(f"❌ Cache not found: {CACHE_PATH}")
        sys.exit(1)

    print(f"[1/4] 讀 cache: {CACHE_PATH}")
    raw_size = os.path.getsize(CACHE_PATH)
    print(f"  原始大小: {raw_size / 1024 / 1024:.1f} MB")

    print(f"[2/4] 壓縮 + base64 編碼...")
    with open(CACHE_PATH, "rb") as f:
        raw = f.read()
    compressed = zlib.compress(raw, level=9)
    print(f"  壓縮後: {len(compressed) / 1024 / 1024:.1f} MB（{len(compressed)/raw_size*100:.0f}%）")
    encoded = base64.b64encode(compressed).decode()
    print(f"  base64 後: {len(encoded) / 1024 / 1024:.1f} MB")

    print(f"[3/4] 切 chunks（每檔 {CHUNK_SIZE/1000:.0f} KB）...")
    chunks = [encoded[i:i+CHUNK_SIZE] for i in range(0, len(encoded), CHUNK_SIZE)]
    n_chunks = len(chunks)
    print(f"  共 {n_chunks} 個 chunk")

    # Manifest 檔
    from datetime import datetime, timezone, timedelta
    TW_TZ = timezone(timedelta(hours=8))
    manifest = {
        "format": "zlib + base64",
        "n_chunks": n_chunks,
        "chunk_size": CHUNK_SIZE,
        "raw_size": raw_size,
        "compressed_size": len(compressed),
        "encoded_size": len(encoded),
        "backup_time": datetime.now(TW_TZ).isoformat(),
        "filename_pattern": "chunk_{:03d}.b64",
        "restore_script": "restore_cache_from_gist.py",
    }

    print(f"[4/4] PATCH Gist {GIST_ID}（分 {n_chunks // 5 + 1} 批推）...")
    # Gist API 一次 PATCH 多檔有 size limit，分批推
    BATCH = 5
    files_to_push = {"manifest.json": json.dumps(manifest, indent=2)}
    for i, chunk in enumerate(chunks):
        files_to_push[f"chunk_{i:03d}.b64"] = chunk
        if len(files_to_push) >= BATCH or i == len(chunks) - 1:
            try:
                status = patch_gist(GIST_ID, files_to_push)
                print(f"  batch {i+1}/{n_chunks}: status {status}")
                files_to_push = {}
                time.sleep(2)  # rate limit buffer
            except Exception as e:
                print(f"  ❌ batch {i+1} fail: {e}")
                # 推到一半失敗 → 停下來
                sys.exit(1)

    print()
    print(f"✅ 備份完成！Gist: https://gist.github.com/{GIST_ID}")
    print(f"   原始 {raw_size/1024/1024:.0f} MB → Gist {len(encoded)/1024/1024:.0f} MB（{n_chunks} chunks）")
    print(f"   還原方法：python restore_cache_from_gist.py")


if __name__ == "__main__":
    main()
