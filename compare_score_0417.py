"""
比較 cpu_replay 和 scanner.py 在 04-17 對達邁/聯茂/希華 算的 score
看是不是兩邊邏輯有差
"""
import os, sys, json, types, urllib.request, pickle, importlib.util
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd

from gpu_cupy_evolve import precompute, download_data

# 從 GitHub 抓 scanner.py 到本機 tmp，再 import
HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER_LOCAL = os.path.join(HERE, "_tmp_scanner.py")
if not os.path.exists(SCANNER_LOCAL):
    print("從 GitHub 抓 scanner.py ...")
    try:
        # repo: williamyang2000727-commits/stock-web-app
        url = "https://api.github.com/repos/williamyang2000727-commits/stock-web-app/contents/scanner.py"
        r = urllib.request.urlopen(urllib.request.Request(url), timeout=30)
        d = json.loads(r.read())
        import base64
        content = base64.b64decode(d["content"])
        with open(SCANNER_LOCAL, "wb") as f:
            f.write(content)
        print(f"  下載完成 → {SCANNER_LOCAL}")
    except Exception as e:
        print(f"❌ 下載 scanner.py 失敗：{e}")
        sys.exit(1)

# import scanner from _tmp_scanner.py
HAS_SCANNER = False
try:
    spec = importlib.util.spec_from_file_location("scanner_tmp", SCANNER_LOCAL)
    scanner_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner_mod)
    compute_indicators = scanner_mod.compute_indicators
    score_stock = scanner_mod.score_stock
    HAS_SCANNER = True
    print("✅ scanner.py 載入成功")
except Exception as e:
    print(f"❌ scanner.py 載入失敗：{e}")
    import traceback; traceback.print_exc()


def fetch_gist_strategy():
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


# 載 cache + strategy
data = download_data()
strategy = fetch_gist_strategy()
p = strategy.get("params", strategy)
print(f"Strategy: {strategy.get('score',0):.3f}")

TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}

# 用 precompute 算指標（這是 cpu_replay 用的）
pre = precompute(data_t)
tickers = pre["tickers"]
dates = pre["dates"]

# 找 04-17 的 day index
target_d = None
for i, d in enumerate(dates):
    d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    if d_str == "2026-04-17":
        target_d = i; break
if target_d is None:
    print("❌ cache 沒 04-17")
    sys.exit(1)
print(f"04-17 = day index {target_d} / total {len(dates)}")

# 對 3 檔股印指標 + cpu_replay 算 score（gpu_cupy_evolve 內含的 score 公式）
focus_tickers = ["3645.TW", "6213.TW", "2484.TW"]

print(f"\n=== 4/17 三檔股的 cpu_replay 用指標 ===")
keys_to_print = ["close", "rsi", "bb_pos", "vol_ratio", "k_val", "macd_hist",
                 "williams_r", "near_high", "vol_up_days", "week52_pos", "up_days",
                 "adx", "atr_pct", "bias", "obv_rising_3", "obv_rising_5", "obv_rising_10",
                 "squeeze_fire", "new_high_60", "mom_accel", "is_green"]

for tk in focus_tickers:
    if tk not in tickers:
        print(f"\n  {tk}: NOT IN tickers")
        continue
    si = tickers.index(tk)
    print(f"\n  {tk} (si={si}) day {target_d} (4/17):")
    for k in keys_to_print:
        if k in pre:
            v = pre[k]
            if hasattr(v, "ndim") and v.ndim == 2:
                val = v[si, target_d]
                # mom 系列特殊處理
                print(f"    {k:20s}: {val}")

# 用 scanner.compute_indicators 對同一檔算指標（看跟 precompute 是否一致）
if HAS_SCANNER:
    print(f"\n\n=== 4/17 用 scanner.compute_indicators 算指標 ===")
    for tk in focus_tickers:
        if tk not in data_t:
            continue
        df = data_t[tk]
        # 取到 4/17 為止的歷史
        df_dates_norm = df.index.tz_localize(None).normalize() if df.index.tz else df.index.normalize()
        mask = df_dates_norm <= pd.Timestamp("2026-04-17")
        df_to_417 = df[mask]
        if len(df_to_417) < 100:
            print(f"  {tk}: 資料不足")
            continue
        try:
            ind = compute_indicators(df_to_417)
            if ind is None:
                print(f"  {tk}: compute_indicators returned None")
                continue
            print(f"\n  {tk} (scanner.compute_indicators):")
            for k in ["close", "rsi", "bb_pos", "vol_ratio", "k_val", "macd_hist", "williams_r",
                      "near_high", "vol_up_days", "week52_pos", "up_days", "adx", "atr_pct", "bias"]:
                v = ind.get(k, "missing")
                print(f"    {k:20s}: {v}")
            # 算 score
            sc = score_stock(ind, p)
            print(f"    SCORE = {sc}")
        except Exception as e:
            print(f"  {tk}: error {e}")

# cpu_replay 內建 score 算法（mirror kernel）— 也算一遍
print(f"\n\n=== 4/17 用 gpu_cupy_evolve cpu_replay 邏輯算 score ===")


def cpu_score(pre, p, si, day):
    """Mirror kernel scoring (gpu_cupy_evolve.py line 408-548)"""
    sc = 0
    # 拿出指標
    rsi = pre["rsi"][si, day]
    bb_pos = pre["bb_pos"][si, day]
    vol_ratio = pre["vol_ratio"][si, day]
    k_val = pre["k_val"][si, day]
    macd_hist = pre["macd_hist"][si, day]
    macd_line = pre["macd_line"][si, day]
    macd_hist_prev = pre["macd_hist"][si, day-1] if day > 0 else 0
    williams_r = pre["williams_r"][si, day]
    near_high = pre["near_high"][si, day]
    week52_pos = pre.get("week52_pos", np.zeros((1,1)))[si, day] if "week52_pos" in pre else 0
    vol_up_days = pre.get("vol_up_days", np.zeros((1,1)))[si, day] if "vol_up_days" in pre else 0
    up_days = pre.get("up_days", np.zeros((1,1)))[si, day] if "up_days" in pre else 0
    adx = pre.get("adx", np.zeros((1,1)))[si, day] if "adx" in pre else 0
    atr_pct = pre.get("atr_pct", np.zeros((1,1)))[si, day] if "atr_pct" in pre else 0
    bias = pre.get("bias", np.zeros((1,1)))[si, day] if "bias" in pre else 0
    is_green = pre["is_green"][si, day]
    squeeze_fire = pre.get("squeeze_fire", np.zeros((1,1)))[si, day] if "squeeze_fire" in pre else 0
    new_high_60 = pre.get("new_high_60", np.zeros((1,1)))[si, day] if "new_high_60" in pre else 0
    mom_accel = pre.get("mom_accel", np.zeros((1,1)))[si, day] if "mom_accel" in pre else 0
    obv_rising_3 = pre.get("obv_rising_3", np.zeros((1,1)))[si, day] if "obv_rising_3" in pre else 0
    obv_rising_5 = pre.get("obv_rising_5", np.zeros((1,1)))[si, day] if "obv_rising_5" in pre else 0
    obv_rising_10 = pre.get("obv_rising_10", np.zeros((1,1)))[si, day] if "obv_rising_10" in pre else 0

    # mom_days 動態
    mom_days = int(p.get("momentum_days", 5))
    mom_key = f"mom_{mom_days}"
    mom_val = pre.get(mom_key, np.zeros((1,1)))[si, day] if mom_key in pre else 0

    # ma_fast_w 動態
    ma_fw = int(p.get("ma_fast_w", 5))
    close = pre["close"][si, day]
    ma_fast_arr = pre.get(f"ma_{ma_fw}")
    ma_fast = ma_fast_arr[si, day] if ma_fast_arr is not None else 0

    # 計算
    if int(p.get("w_rsi",0)) > 0 and rsi >= p.get("rsi_th", 55): sc += int(p.get("w_rsi",0))
    if int(p.get("w_bb",0)) > 0 and bb_pos >= p.get("bb_th", 0.7): sc += int(p.get("w_bb",0))
    if int(p.get("w_vol",0)) > 0 and vol_ratio >= p.get("vol_th", 3): sc += int(p.get("w_vol",0))
    if int(p.get("w_ma",0)) > 0 and close > ma_fast: sc += int(p.get("w_ma",0))
    if int(p.get("w_wr",0)) > 0 and williams_r >= p.get("wr_th", -30): sc += int(p.get("w_wr",0))
    if int(p.get("w_mom",0)) > 0 and mom_val >= p.get("mom_th", 3): sc += int(p.get("w_mom",0))
    if int(p.get("w_near_high",0)) > 0 and abs(near_high) <= p.get("near_high_pct", 10): sc += int(p.get("w_near_high",0))
    if int(p.get("w_squeeze",0)) > 0 and squeeze_fire == 1: sc += int(p.get("w_squeeze",0))
    if int(p.get("w_new_high",0)) > 0 and new_high_60 == 1: sc += int(p.get("w_new_high",0))
    if int(p.get("w_adx",0)) > 0 and adx >= p.get("adx_th", 25): sc += int(p.get("w_adx",0))
    if int(p.get("w_atr",0)) > 0 and atr_pct >= p.get("atr_min", 2.0): sc += int(p.get("w_atr",0))
    if int(p.get("w_bias",0)) > 0 and 0 <= bias <= p.get("bias_max", 15): sc += int(p.get("w_bias",0))
    if int(p.get("w_up_days",0)) > 0 and up_days >= p.get("up_days_min", 3): sc += int(p.get("w_up_days",0))
    if int(p.get("w_week52",0)) > 0 and week52_pos >= p.get("week52_min", 0.7): sc += int(p.get("w_week52",0))
    if int(p.get("w_vol_up_days",0)) > 0 and vol_up_days >= p.get("vol_up_days_min", 3): sc += int(p.get("w_vol_up_days",0))
    if int(p.get("w_mom_accel",0)) > 0 and mom_accel >= p.get("mom_accel_min", 2): sc += int(p.get("w_mom_accel",0))
    # KD
    if int(p.get("w_kd",0)) > 0 and k_val >= p.get("kd_th", 50): sc += int(p.get("w_kd",0))
    # MACD
    if int(p.get("w_macd",0)) > 0:
        mm = int(p.get("macd_mode", 2))
        ok = (macd_hist > 0 and macd_hist_prev <= 0) if mm == 0 else (macd_line > 0 if mm == 1 else macd_hist > 0)
        if ok: sc += int(p.get("w_macd",0))
    # OBV
    if int(p.get("w_obv",0)) > 0 and (obv_rising_3 or obv_rising_5 or obv_rising_10):
        sc += int(p.get("w_obv",0))

    return sc


for tk in focus_tickers:
    if tk not in tickers:
        continue
    si = tickers.index(tk)
    sc = cpu_score(pre, p, si, target_d)
    print(f"  {tk} cpu_replay score = {sc}")

# top 排名
print(f"\n=== 4/17 全 universe 的 cpu_replay score 排名 top 30 ===")
all_scores = []
for si in range(len(tickers)):
    sc = cpu_score(pre, p, si, target_d)
    all_scores.append((sc, tickers[si], si))
all_scores.sort(reverse=True)
for rank, (sc, tk, si) in enumerate(all_scores[:30], 1):
    flag = "⭐" if tk in focus_tickers else ""
    print(f"  rank {rank:>3}: {tk:>10s} score={sc:>3} {flag}")
