"""04-20 是訊號日（週一），看那天 universe + 焦點股 score"""
import os, sys, json, types, urllib.request
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
import gpu_cupy_evolve as base
from gpu_cupy_evolve import precompute, download_data


def fetch_gist():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_gist()
p = strategy.get("params", strategy)

TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data_t)
tickers = pre["tickers"]
dates = pre["dates"]
n = len(tickers)
nd = len(dates)

# bind arrays
rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
close = pre["close"]; macd_line = pre["macd_line"]; macd_hist = pre["macd_hist"]
k_val = pre["k_val"]; d_val = pre["d_val"]
williams_r = pre["williams_r"]; near_high = pre["near_high"]
squeeze_fire = pre.get("squeeze_fire"); new_high_60 = pre.get("new_high_60")
adx_arr = pre.get("adx"); bias_arr = pre.get("bias")
obv_rising_arr = pre.get("obv_rising"); atr_pct_arr = pre.get("atr_pct")
sector_hot = pre.get("sector_hot")
up_days_arr = pre.get("up_days"); week52_arr = pre.get("week52_pos")
vol_up_days_arr = pre.get("vol_up_days"); mom_accel_arr = pre.get("mom_accel")

ma_fw = int(p.get("ma_fast_w", 5))
mom_days = int(p.get("momentum_days", 5))
maf = pre["ma_d"].get(ma_fw, pre["ma_d"][5])
mom = pre["mom_d"].get(mom_days, pre["mom_d"][5])


def score_mirror(si, day):
    d = day; sc = 0.0
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
    if int(p.get("w_sector_flow",0))>0 and sector_hot is not None and sector_hot[si,d]<p.get("sector_flow_topn",3): sc+=int(p["w_sector_flow"])
    if int(p.get("w_up_days",0))>0 and up_days_arr is not None and up_days_arr[si,d]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
    if int(p.get("w_week52",0))>0 and week52_arr is not None and week52_arr[si,d]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
    if int(p.get("w_vol_up_days",0))>0 and vol_up_days_arr is not None and vol_up_days_arr[si,d]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
    if int(p.get("w_mom_accel",0))>0 and mom_accel_arr is not None and mom_accel_arr[si,d]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
    return sc


# 04-20 = index 1495
day_0420 = 1495
print(f"=== 04-20 (週一) Universe top 20 ===")
top100_mask = pre["top100_mask"]
in_uni_si = np.where(top100_mask[:, day_0420] >= 0.5)[0]
print(f"  Universe size: {len(in_uni_si)}")

scores = [(score_mirror(si, day_0420), tickers[si], si) for si in in_uni_si]
scores.sort(reverse=True)
for rank, (sc, tk, si) in enumerate(scores[:20], 1):
    flag = "⭐" if tk in ["3645.TW", "6213.TW", "2484.TW"] else ""
    print(f"  #{rank:>2} {tk:>10s} score={sc:>5.1f} {flag}")

print(f"\n=== 焦點股 04-20 score ===")
for tk in ["3645.TW", "6213.TW", "2484.TW"]:
    if tk in tickers:
        si = tickers.index(tk)
        sc = score_mirror(si, day_0420)
        in_uni = "in" if si in set(in_uni_si.tolist()) else "out"
        print(f"  {tk} ({in_uni}): score = {sc}")

# 04-20 那天有沒有持倉中（陽程 / 創威）→ 賣出條件觸發 → 同日換股
# 撈 cpu_replay 真實 trades，看 04-20 / 04-21 的 sell 邏輯
trades = base.cpu_replay(pre, p)
print(f"\n=== 04-20 / 04-21 賣出與買入 ===")
for t in trades:
    sd = t.get("sell_date", "")
    bd = t.get("buy_date", "")
    if sd in ["2026-04-20", "2026-04-21"]:
        print(f"  SELL {sd}: {t.get('name','')} ({t.get('ticker','')}) reason={t.get('reason','')}")
    if bd in ["2026-04-20", "2026-04-21", "2026-04-22"]:
        print(f"  BUY  {bd}: {t.get('name','')} ({t.get('ticker','')}) buy_p={t.get('buy_price','')}")
