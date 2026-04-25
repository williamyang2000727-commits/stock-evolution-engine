"""
追蹤 cpu_replay 04-15 ~ 04-22 每天 best_si 是誰
直接修改 gpu_cupy_evolve.cpu_replay 加 instrumentation
"""
import os, sys, json, types, urllib.request
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from gpu_cupy_evolve import precompute, download_data


def fetch_gist_strategy():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_gist_strategy()
p = strategy.get("params", strategy)
print(f"Strategy: {strategy.get('score',0):.3f}")
print(f"buy_threshold: {p.get('buy_threshold', '?')}")

TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data_t)
tickers = pre["tickers"]
dates = pre["dates"]
n = len(tickers)

# 找 4/15-4/22 對應的 day index
target_days = {}
for i, d in enumerate(dates):
    d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    if d_str in ["2026-04-15", "2026-04-16", "2026-04-17", "2026-04-21", "2026-04-22"]:
        target_days[d_str] = i

print(f"Day indices: {target_days}")
print(f"  Note: 4/18 is Saturday, not in cache")
print()

# 自己算 universe（top100 by vol_ratio at day-1，匹配 cpu_replay 邏輯）
# cpu_replay 用 vol_prev (前一天 volume) 排前 100
vol_prev = pre.get("vol_prev")
if vol_prev is None:
    print("❌ pre['vol_prev'] missing")
    sys.exit(1)


def cpu_score(pre, p, si, day):
    """Mirror kernel scoring (gpu_cupy_evolve.py line 408-548)"""
    sc = 0
    rsi = pre["rsi"][si, day]
    bb_pos = pre["bb_pos"][si, day]
    vol_ratio = pre["vol_ratio"][si, day]
    k_val = pre["k_val"][si, day]
    macd_hist = pre["macd_hist"][si, day]
    macd_line = pre["macd_line"][si, day]
    macd_hist_prev = pre["macd_hist"][si, day-1] if day > 0 else 0
    williams_r = pre["williams_r"][si, day]
    near_high = pre["near_high"][si, day]
    week52_pos = pre.get("week52_pos", np.zeros((n, len(dates))))[si, day]
    vol_up_days = pre.get("vol_up_days", np.zeros((n, len(dates))))[si, day]
    up_days = pre.get("up_days", np.zeros((n, len(dates))))[si, day]
    adx = pre.get("adx", np.zeros((n, len(dates))))[si, day]
    atr_pct = pre.get("atr_pct", np.zeros((n, len(dates))))[si, day]
    bias = pre.get("bias", np.zeros((n, len(dates))))[si, day]
    is_green = pre["is_green"][si, day]
    squeeze_fire = pre.get("squeeze_fire", np.zeros((n, len(dates))))[si, day]
    new_high_60 = pre.get("new_high_60", np.zeros((n, len(dates))))[si, day]
    mom_accel = pre.get("mom_accel", np.zeros((n, len(dates))))[si, day]
    obv_rising_3 = pre.get("obv_rising_3", np.zeros((n, len(dates))))[si, day]
    obv_rising_5 = pre.get("obv_rising_5", np.zeros((n, len(dates))))[si, day]
    obv_rising_10 = pre.get("obv_rising_10", np.zeros((n, len(dates))))[si, day]

    mom_days = int(p.get("momentum_days", 5))
    mom_key = f"mom_{mom_days}"
    mom_val = pre.get(mom_key, np.zeros((n, len(dates))))[si, day] if mom_key in pre else 0

    ma_fw = int(p.get("ma_fast_w", 5))
    close = pre["close"][si, day]
    ma_fast_arr = pre.get(f"ma_{ma_fw}")
    ma_fast = ma_fast_arr[si, day] if ma_fast_arr is not None else 0

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
    if int(p.get("w_kd",0)) > 0 and k_val >= p.get("kd_th", 50): sc += int(p.get("w_kd",0))
    if int(p.get("w_macd",0)) > 0:
        mm = int(p.get("macd_mode", 2))
        ok = (macd_hist > 0 and macd_hist_prev <= 0) if mm == 0 else (macd_line > 0 if mm == 1 else macd_hist > 0)
        if ok: sc += int(p.get("w_macd",0))
    if int(p.get("w_obv",0)) > 0 and (obv_rising_3 or obv_rising_5 or obv_rising_10):
        sc += int(p.get("w_obv",0))
    return sc


# 對每天，列 top 100 universe 內 score 最高的 5 檔 + 達邁/聯茂/希華 的 vol rank + score
buy_threshold = int(p.get("buy_threshold", 14))
print(f"\n要 score >= {buy_threshold} 才能進場")
print()

for d_str in sorted(target_days.keys()):
    day = target_days[d_str]
    print(f"━━━ Day {d_str} (index {day}) ━━━")
    # universe = top100 vol
    vol_at = vol_prev[:, day] if day > 0 else vol_prev[:, day]
    order = np.argsort(-vol_at)
    top100_idx = order[:100]
    top100_set = set(top100_idx.tolist())

    # 焦點股 vol rank
    for tk in ["3645.TW", "6213.TW", "2484.TW"]:
        if tk not in tickers:
            continue
        si = tickers.index(tk)
        vol_rank = list(order).index(si) + 1
        sc = cpu_score(pre, p, si, day)
        in_uni = "✅in" if si in top100_set else "❌out"
        print(f"  {tk} vol_rank=#{vol_rank} {in_uni} score={sc}")

    # universe 內 top 5 score
    in_uni_scores = [(cpu_score(pre, p, si, day), tickers[si], si) for si in top100_idx]
    in_uni_scores.sort(reverse=True)
    print(f"  Universe top 5:")
    for rank, (sc, tk, si) in enumerate(in_uni_scores[:5], 1):
        flag = "🎯" if sc >= buy_threshold else ""
        print(f"    #{rank} {tk:>10s} score={sc} {flag}")
    print()
