r"""驗證 04-24 那天 cpu_replay 真公式 universe top 5（直接從 cache 重算）"""
import os, sys, json, types, urllib.request
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "stock-evolution"))
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from gpu_cupy_evolve import precompute, download_data, get_name


def fetch_strategy():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"),
        timeout=30,
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_strategy()
p = strategy.get("params", strategy)

# 用固定起點 2020-01-02
FIXED_START = pd.Timestamp("2020-01-02").normalize()
data_t = {}
for k, v in data.items():
    idx = v.index
    idx_naive = idx.tz_localize(None) if hasattr(idx, "tz") and idx.tz is not None else idx
    if idx_naive.normalize()[0] > FIXED_START:
        continue
    mask = idx_naive.normalize() >= FIXED_START
    df = v[mask]
    if len(df) >= 100:
        data_t[k] = df
print(f"{len(data_t)} stocks 通過")
pre = precompute(data_t)
tickers = pre["tickers"]
dates = pre["dates"]

# 找 04-24 day index
day_0424 = None
for i, d in enumerate(dates):
    if pd.Timestamp(d).strftime("%Y-%m-%d") == "2026-04-24":
        day_0424 = i; break
print(f"04-24 = day index {day_0424} / 末日 {pd.Timestamp(dates[-1]).date()}")
print()

# Bind arrays
rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
close = pre["close"]; macd_line = pre["macd_line"]; macd_hist = pre["macd_hist"]
k_val = pre["k_val"]; d_val = pre["d_val"]
williams_r = pre["williams_r"]; near_high = pre["near_high"]
is_green = pre.get("is_green"); gap = pre.get("gap"); ma60 = pre.get("ma60")
vol_prev = pre.get("vol_prev")
squeeze_fire = pre.get("squeeze_fire"); new_high_60 = pre.get("new_high_60")
adx_arr = pre.get("adx"); bias_arr = pre.get("bias")
obv_rising_arr = pre.get("obv_rising"); atr_pct_arr = pre.get("atr_pct")
up_days_arr = pre.get("up_days"); week52_arr = pre.get("week52_pos")
vol_up_days_arr = pre.get("vol_up_days"); mom_accel_arr = pre.get("mom_accel")
ma_fw = int(p.get("ma_fast_w", 5))
mom_days = int(p.get("momentum_days", 5))
maf = pre["ma_d"].get(ma_fw, pre["ma_d"][5])
mom = pre["mom_d"].get(mom_days, pre["mom_d"][5])


def score(si, d):
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
    if int(p.get("w_obv",0))>0 and obv_rising_arr is not None and obv_rising_arr[si,d]>0.5: sc+=int(p["w_obv"])
    if int(p.get("w_atr",0))>0 and atr_pct_arr is not None and atr_pct_arr[si,d]>=p.get("atr_min",2): sc+=int(p["w_atr"])
    if int(p.get("w_up_days",0))>0 and up_days_arr is not None and up_days_arr[si,d]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
    if int(p.get("w_week52",0))>0 and week52_arr is not None and week52_arr[si,d]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
    if int(p.get("w_vol_up_days",0))>0 and vol_up_days_arr is not None and vol_up_days_arr[si,d]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
    if int(p.get("w_mom_accel",0))>0 and mom_accel_arr is not None and mom_accel_arr[si,d]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
    cg = int(p.get("consecutive_green", 0))
    if cg >= 1 and is_green is not None:
        ok = True
        for g in range(cg):
            if d - g < 0 or is_green[si, d-g] != 1: ok = False; break
        if ok: sc += 1
    if p.get("gap_up", 0) and gap is not None and gap[si, d] >= 1.0: sc += 1
    if p.get("above_ma60", 0) and ma60 is not None and close[si, d] >= ma60[si, d]: sc += 1
    if p.get("vol_gt_yesterday", 0) and d >= 1 and vol_prev is not None and vol_ratio[si, d] > vol_prev[si, d]: sc += 1
    return sc


# 04-24 universe top 10
top100_mask = pre["top100_mask"]
in_uni = np.where(top100_mask[:, day_0424] >= 0.5)[0]
print(f"04-24 universe: {len(in_uni)} stocks")
print()

scores = []
for si in in_uni:
    sc = score(si, day_0424)
    scores.append((sc, si))
scores.sort(reverse=True)

print(f"{'Rank':>4} {'Ticker':>10} {'Name':>10} {'Score':>6} {'Close':>10} {'VolRatio':>9}")
print("-" * 65)
for rank, (sc, si) in enumerate(scores[:15], 1):
    tk = tickers[si]
    name = get_name(tk)
    cl = float(close[si, day_0424])
    vr = float(vol_ratio[si, day_0424])
    print(f"  #{rank:>2}  {tk:>10} {name:>10} {sc:>6.1f} {cl:>10.2f} {vr:>9.2f}")

# 跟 scan_results 比較
print()
print("=== 對比 scan_results.json buy_signals top 5 ===")
print("  #1 全科 (3209.TW) 27")
print("  #2 威盛 (2388.TW) 25")
print("  #3 良維 (6290.TWO) 25")
print("  #4 聯發科 (2454.TW) 24")
print("  #5 臻鼎-KY (4958.TW) 22")
print()
print("如果 mirror score 算的 top 跟上面一致 → write_web_data 沒 bug")
print("如果不一致 → write_web_data 計算錯誤 / 用了不同 day index")
