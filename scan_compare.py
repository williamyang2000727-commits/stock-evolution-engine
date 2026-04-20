"""比較 89.90 和 103(1pos) 策略在今天的買入排名差異
用法：python scan_compare.py
"""
import sys, types, json, math
import numpy as np
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, download_data, get_name
import urllib.request, os

# 讀策略
HERE = os.path.dirname(os.path.abspath(__file__))

# 89.90 from Gist
r = urllib.request.urlopen("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2", timeout=15)
p89 = json.loads(json.load(r)["files"]["best_strategy.json"]["content"])["params"]

# 103 from backup
p103_path = os.path.join(HERE, "strategy_103_1pos_backup.json")
if os.path.exists(p103_path):
    p103 = json.load(open(p103_path, encoding="utf-8"))["params"]
else:
    print("strategy_103_1pos_backup.json not found!"); sys.exit(1)

# 載入資料
data = download_data()
_lens = [len(v) for v in data.values()]
TARGET = 1500 if sum(1 for l in _lens if l >= 1500) >= 500 else 900
data = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data)
ns, nd = pre["n_stocks"], pre["n_days"]
day = nd - 1
print(f"Date: {pre['dates'][day].date()} | {ns} stocks x {nd} days")

tickers = pre["tickers"]; close = pre["close"]
rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
macd_hist = pre["macd_hist"]; macd_line = pre["macd_line"]
k_val = pre["k_val"]; d_val = pre["d_val"]; williams_r = pre["williams_r"]
is_green = pre["is_green"]; gap = pre["gap"]; near_high = pre["near_high"]
vol_prev = pre["vol_prev"]; squeeze_fire = pre["squeeze_fire"]
new_high_60 = pre["new_high_60"]; adx_arr = pre["adx"]; bias_arr = pre["bias"]
obv_rising = pre["obv_rising"]; atr_pct = pre["atr_pct"]
top100_mask = pre["top100_mask"]; ma60 = pre["ma60"]
up_days = pre.get("up_days"); week52 = pre.get("week52_pos")
vol_up_days = pre.get("vol_up_days"); mom_accel = pre.get("mom_accel")

def scan_with(p):
    maf = pre["ma_d"].get(int(p.get("ma_fast_w", 5)), pre["ma_d"][5])
    mom = pre["mom_d"].get(int(p.get("momentum_days", 5)), pre["mom_d"][5])
    bt = p.get("buy_threshold", 8)
    results = []
    for si in range(ns):
        if top100_mask[si, day] < 0.5: continue
        sc = 0.0
        if int(p.get("w_rsi",0))>0 and rsi[si,day]>=p.get("rsi_th",55): sc+=int(p["w_rsi"])
        if int(p.get("w_bb",0))>0 and bb_pos[si,day]>=p.get("bb_th",0.7): sc+=int(p["w_bb"])
        if int(p.get("w_vol",0))>0 and vol_ratio[si,day]>=p.get("vol_th",3): sc+=int(p["w_vol"])
        if int(p.get("w_ma",0))>0 and close[si,day]>maf[si,day]: sc+=int(p["w_ma"])
        if int(p.get("w_macd",0))>0:
            mm=int(p.get("macd_mode",2))
            if mm==0 and day>=1 and macd_hist[si,day]>0 and macd_hist[si,day-1]<=0: sc+=int(p["w_macd"])
            elif mm==1 and macd_line[si,day]>0: sc+=int(p["w_macd"])
            elif mm==2 and macd_hist[si,day]>0: sc+=int(p["w_macd"])
        if int(p.get("w_kd",0))>0:
            ok=k_val[si,day]>=p.get("kd_th",50)
            if ok and p.get("kd_cross",0) and day>=1: ok=k_val[si,day]>d_val[si,day] and k_val[si,day-1]<=d_val[si,day-1]
            if ok: sc+=int(p["w_kd"])
        if int(p.get("w_wr",0))>0 and williams_r[si,day]>=p.get("wr_th",-30): sc+=int(p["w_wr"])
        if int(p.get("w_mom",0))>0 and mom[si,day]>=p.get("mom_th",3): sc+=int(p["w_mom"])
        if int(p.get("w_near_high",0))>0 and abs(near_high[si,day])<=p.get("near_high_pct",10): sc+=int(p["w_near_high"])
        if int(p.get("w_squeeze",0))>0 and squeeze_fire[si,day]>0.5: sc+=int(p["w_squeeze"])
        if int(p.get("w_new_high",0))>0 and new_high_60[si,day]>0.5: sc+=int(p["w_new_high"])
        if int(p.get("w_adx",0))>0 and adx_arr[si,day]>=p.get("adx_th",25): sc+=int(p["w_adx"])
        if int(p.get("w_bias",0))>0 and bias_arr[si,day]>=0 and bias_arr[si,day]<=p.get("bias_max",15): sc+=int(p["w_bias"])
        if int(p.get("w_obv",0))>0 and obv_rising[si,day]>0.5: sc+=int(p["w_obv"])
        if int(p.get("w_atr",0))>0 and atr_pct[si,day]>=p.get("atr_min",2): sc+=int(p["w_atr"])
        if int(p.get("w_up_days",0))>0 and up_days is not None and up_days[si,day]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
        if int(p.get("w_week52",0))>0 and week52 is not None and week52[si,day]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
        if int(p.get("w_vol_up_days",0))>0 and vol_up_days is not None and vol_up_days[si,day]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
        if int(p.get("w_mom_accel",0))>0 and mom_accel is not None and mom_accel[si,day]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
        cg=int(p.get("consecutive_green",0))
        if cg>=1:
            ok=True
            for g in range(cg):
                if day-g<0 or is_green[si,day-g]!=1: ok=False; break
            if ok: sc+=1
        if p.get("gap_up",0) and gap[si,day]>=1.0: sc+=1
        if p.get("above_ma60",0) and close[si,day]>=ma60[si,day]: sc+=1
        if p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]>vol_prev[si,day]: sc+=1
        if sc >= bt:
            results.append((int(sc), round(float(vol_ratio[si,day]),1), tickers[si], float(close[si,day])))
    results.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return results

for name, p in [("89.90 (2pos)", p89), ("103 (1pos)", p103)]:
    cands = scan_with(p)
    print(f"\n{'='*60}")
    print(f"  {name} — Top 10 (threshold={p.get('buy_threshold')})")
    print(f"{'='*60}")
    for i, (sc, vr, tk, cl) in enumerate(cands[:10]):
        print(f"  #{i+1:>2} {tk:>10} {get_name(tk):>8} score={sc:>2} vr={vr:>4} close={cl:>7.1f}")
    print(f"  Total signals: {len(cands)}")

print("\nDone!")
