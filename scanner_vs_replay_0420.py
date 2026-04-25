"""對比 scanner.py 跟 cpu_replay 在 04-20 對達邁/聯茂/希華算的 score"""
import os, sys, json, types, urllib.request, importlib.util, base64
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from gpu_cupy_evolve import precompute, download_data

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER_LOCAL = os.path.join(HERE, "_tmp_scanner.py")

# 拉最新 scanner.py
print("從 GitHub 抓最新 scanner.py ...")
url = "https://api.github.com/repos/williamyang2000727-commits/stock-web-app/contents/scanner.py"
r = urllib.request.urlopen(urllib.request.Request(url), timeout=30)
d = json.loads(r.read())
content = base64.b64decode(d["content"])
with open(SCANNER_LOCAL, "wb") as f:
    f.write(content)
print(f"  下載完成 → {SCANNER_LOCAL}")

# import scanner
spec = importlib.util.spec_from_file_location("scanner_tmp", SCANNER_LOCAL)
scanner_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner_mod)
compute_indicators = scanner_mod.compute_indicators
score_stock = scanner_mod.score_stock
print("✅ scanner.py 載入成功")


def fetch_gist():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


# 載 raw cache（不切 1500 tail，給 compute_indicators 全期）
data = download_data()
strategy = fetch_gist()
p = strategy.get("params", strategy)

target_date = pd.Timestamp("2026-04-20").normalize()

# 對達邁/聯茂/希華 算 scanner score
print(f"\n=== scanner.compute_indicators + score_stock at 04-20 ===")
focus = {"3645.TW": "達邁", "6213.TW": "聯茂", "2484.TW": "希華"}
for tk, name in focus.items():
    if tk not in data:
        print(f"  {tk}: not in data")
        continue
    df = data[tk]
    # 切到 04-20 為止
    df_dates_norm = df.index.tz_localize(None).normalize() if df.index.tz else df.index.normalize()
    mask = df_dates_norm <= target_date
    df_to_d = df[mask]
    if len(df_to_d) < 100:
        print(f"  {tk} ({name}): 資料不足")
        continue
    try:
        # compute_indicators 簽名是 (c, h, lo, vol, open_arr=None, ...)
        c = df_to_d["Close"].values.astype(np.float64)
        h = df_to_d["High"].values.astype(np.float64)
        lo = df_to_d["Low"].values.astype(np.float64)
        vol = df_to_d["Volume"].values.astype(np.float64)
        opn = df_to_d["Open"].values.astype(np.float64) if "Open" in df_to_d.columns else None
        ind = compute_indicators(c, h, lo, vol, opn) if opn is not None else compute_indicators(c, h, lo, vol)
        if ind is None:
            print(f"  {tk} ({name}): compute_indicators returned None")
            continue
        sc = score_stock(ind, p)
        print(f"\n  {tk} ({name}):")
        print(f"    scanner score = {sc}")
        # 印關鍵指標
        for k in ["rsi", "bb_pos", "vol_ratio", "k_val", "macd_hist", "williams_r",
                  "near_high", "vol_up_days", "week52_pos", "up_days", "adx", "atr_pct",
                  "bias", "consecutive_green_days", "gap_pct", "above_ma60",
                  "vol_gt_yesterday", "obv_rising_3", "obv_rising_5", "obv_rising_10",
                  "is_green", "squeeze_fire", "new_high_60", "mom_accel"]:
            v = ind.get(k, "missing")
            print(f"    {k:25s}: {v}")
        # mom_3
        mom_days = int(p.get("momentum_days", 5))
        print(f"    momentum_{mom_days}: {ind.get(f'momentum_{mom_days}', 'missing')}")
        # ma3
        ma_fw = int(p.get("ma_fast_w", 5))
        print(f"    ma{ma_fw}: {ind.get(f'ma{ma_fw}', 'missing')}")
        print(f"    price (close): {ind.get('price', 'missing')}")
    except Exception as e:
        print(f"  {tk} ({name}): error {e}")
        import traceback; traceback.print_exc()

# cpu_replay mirror score（之前算出聯茂 27、達邁 23）
print(f"\n\n=== cpu_replay (mirror 完整版) at 04-20 已知 ===")
print(f"  聯茂(6213): 27")
print(f"  希華(2484): 26")
print(f"  達邁(3645): 23")
print(f"\n  → 如果 scanner 算的達邁 > 聯茂，就是 scanner.py 有 bug")
