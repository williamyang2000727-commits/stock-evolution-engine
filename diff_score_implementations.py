"""
Mirror cpu_replay 真實 _score_stock 公式（line 114-154）
逐行對比我手寫版差在哪
"""
import os, sys, json, types, urllib.request
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
import gpu_cupy_evolve as base
from gpu_cupy_evolve import precompute, download_data


def fetch_gist_strategy():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_gist_strategy()
p = strategy.get("params", strategy)

TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data_t)
tickers = pre["tickers"]
dates = pre["dates"]
n = len(tickers)
nd = len(dates)

# 找 day index
day_0417 = None
for i, d in enumerate(dates):
    if pd.Timestamp(d).strftime("%Y-%m-%d") == "2026-04-17":
        day_0417 = i; break
print(f"04-17 day index: {day_0417}")

# 取出 cpu_replay 用的所有 array（mirror line 114 之前的綁定）
rsi = pre["rsi"]
bb_pos = pre["bb_pos"]
vol_ratio = pre["vol_ratio"]
close = pre["close"]
macd_line = pre["macd_line"]
macd_hist = pre["macd_hist"]
k_val = pre["k_val"]
d_val = pre["d_val"]
williams_r = pre["williams_r"]
near_high = pre["near_high"]
squeeze_fire = pre.get("squeeze_fire")
new_high_60 = pre.get("new_high_60")
adx_arr = pre.get("adx")
bias_arr = pre.get("bias")
obv_rising_arr = pre.get("obv_rising")  # ⚠️ 注意 cpu_replay 用 obv_rising 不是 obv_rising_3/5/10
atr_pct_arr = pre.get("atr_pct")
sector_hot = pre.get("sector_hot")
up_days_arr = pre.get("up_days")
week52_arr = pre.get("week52_pos")
vol_up_days_arr = pre.get("vol_up_days")
mom_accel_arr = pre.get("mom_accel")

# cpu_replay 真實邏輯 (line 1378-1381)：從 pre["ma_d"] / pre["mom_d"] dict 拿
ma_fw = int(p.get("ma_fast_w", 5))
mom_days = int(p.get("momentum_days", 5))
maf = pre["ma_d"].get(ma_fw, pre["ma_d"][5])
mom = pre["mom_d"].get(mom_days, pre["mom_d"][5])
print(f"ma_fast_w={ma_fw} → maf shape={maf.shape}")
print(f"momentum_days={mom_days} → mom shape={mom.shape}")


def score_mirror(si, day):
    """逐行 mirror cpu_replay._score_stock (gpu_cupy_evolve.py line 114-154)"""
    d = day
    sc = 0.0
    if int(p.get("w_rsi",0))>0 and rsi[si,d]>=p.get("rsi_th",55): sc+=int(p["w_rsi"])
    if int(p.get("w_bb",0))>0 and bb_pos[si,d]>=p.get("bb_th",0.7): sc+=int(p["w_bb"])
    if int(p.get("w_vol",0))>0 and vol_ratio[si,d]>=p.get("vol_th",3): sc+=int(p["w_vol"])
    if int(p.get("w_ma",0))>0 and close[si,d]>maf[si,d]: sc+=int(p["w_ma"])
    if int(p.get("w_macd",0))>0:
        mm=int(p.get("macd_mode",2)); ok=False
        if mm==0 and d>=1 and macd_hist[si,d]>0 and macd_hist[si,d-1]<=0: ok=True
        elif mm==1 and macd_line[si,d]>0: ok=True
        elif mm==2 and macd_hist[si,d]>0: ok=True
        if ok: sc+=int(p["w_macd"])
    if int(p.get("w_kd",0))>0:
        ok=k_val[si,d]>=p.get("kd_th",50)
        if ok and p.get("kd_cross",0) and d>=1: ok=k_val[si,d]>d_val[si,d] and k_val[si,d-1]<=d_val[si,d-1]
        if ok: sc+=int(p["w_kd"])
    if int(p.get("w_wr",0))>0 and williams_r[si,d]>=p.get("wr_th",-30): sc+=int(p["w_wr"])
    if int(p.get("w_mom",0))>0 and mom[si,d]>=p.get("mom_th",3): sc+=int(p["w_mom"])
    if int(p.get("w_near_high",0))>0 and abs(near_high[si,d])<=p.get("near_high_pct",10): sc+=int(p["w_near_high"])
    if int(p.get("w_squeeze",0))>0 and squeeze_fire is not None and squeeze_fire[si,d]>0.5: sc+=int(p["w_squeeze"])
    if int(p.get("w_new_high",0))>0 and new_high_60 is not None and new_high_60[si,d]>0.5: sc+=int(p["w_new_high"])
    if int(p.get("w_adx",0))>0 and adx_arr is not None and adx_arr[si,d]>=p.get("adx_th",25): sc+=int(p["w_adx"])
    if int(p.get("w_bias",0))>0 and bias_arr is not None and bias_arr[si,d]>=0 and bias_arr[si,d]<=p.get("bias_max",15): sc+=int(p["w_bias"])
    # ⭐ key: cpu_replay 用 obv_rising（單一 array），不是 obv_rising_3/5/10
    if int(p.get("w_obv",0))>0 and obv_rising_arr is not None and obv_rising_arr[si,d]>0.5: sc+=int(p["w_obv"])
    if int(p.get("w_atr",0))>0 and atr_pct_arr is not None and atr_pct_arr[si,d]>=p.get("atr_min",2): sc+=int(p["w_atr"])
    if int(p.get("w_sector_flow",0))>0 and sector_hot is not None and sector_hot[si,d]<p.get("sector_flow_topn",3): sc+=int(p["w_sector_flow"])
    if int(p.get("w_up_days",0))>0 and up_days_arr is not None and up_days_arr[si,d]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
    if int(p.get("w_week52",0))>0 and week52_arr is not None and week52_arr[si,d]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
    if int(p.get("w_vol_up_days",0))>0 and vol_up_days_arr is not None and vol_up_days_arr[si,d]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
    if int(p.get("w_mom_accel",0))>0 and mom_accel_arr is not None and mom_accel_arr[si,d]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
    return sc


# 用 mirror 算 04-17 universe top 10
top100_mask = pre["top100_mask"]
in_uni_si = np.where(top100_mask[:, day_0417] >= 0.5)[0]
scores = [(score_mirror(si, day_0417), tickers[si], si) for si in in_uni_si]
scores.sort(reverse=True)

print(f"\n=== 04-17 Universe top 15 (mirror cpu_replay 真實公式) ===")
for rank, (sc, tk, si) in enumerate(scores[:15], 1):
    flag = "⭐" if tk in ["3645.TW", "6213.TW", "2484.TW"] else ""
    print(f"  #{rank:>2} {tk:>10s} score={sc:>5.1f} {flag}")

# 看達邁/聯茂/希華
print(f"\n=== 焦點股 04-17 score (mirror) ===")
for tk in ["3645.TW", "6213.TW", "2484.TW"]:
    if tk in tickers:
        si = tickers.index(tk)
        sc = score_mirror(si, day_0417)
        in_uni = "in" if si in set(in_uni_si.tolist()) else "out"
        print(f"  {tk} ({in_uni}): score = {sc}")

# 對比 04-21 訊號日（用於 04-22 進場）
day_0421 = None
for i, d in enumerate(dates):
    if pd.Timestamp(d).strftime("%Y-%m-%d") == "2026-04-21":
        day_0421 = i; break

if day_0421:
    print(f"\n=== 04-21 Universe top 15 (給 04-22 進場用) ===")
    in_uni_si_421 = np.where(top100_mask[:, day_0421] >= 0.5)[0]
    scores_421 = [(score_mirror(si, day_0421), tickers[si], si) for si in in_uni_si_421]
    scores_421.sort(reverse=True)
    for rank, (sc, tk, si) in enumerate(scores_421[:15], 1):
        flag = "⭐" if tk in ["3645.TW", "6213.TW", "2484.TW"] else ""
        print(f"  #{rank:>2} {tk:>10s} score={sc:>5.1f} {flag}")
