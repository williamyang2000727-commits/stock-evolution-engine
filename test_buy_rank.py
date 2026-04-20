"""測試：買第 N 名的績效差異（用現有策略，只改買入排名）
用法：python test_buy_rank.py
"""
import sys, types, json, math
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, download_data, get_name, REASON_NAMES
import numpy as np

# 讀 89.90 策略
import urllib.request
r = urllib.request.urlopen("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2", timeout=15)
strategy = json.loads(json.load(r)["files"]["best_strategy.json"]["content"])
p = strategy["params"]

# 載入資料
data = download_data()
_lens = [len(v) for v in data.values()]
TARGET_DAYS = 1500 if sum(1 for l in _lens if l >= 1500) >= 500 else 900
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
print(f"Stocks: {len(data)} | Days: {TARGET_DAYS}")
pre = precompute(data)
ns, nd = pre["n_stocks"], pre["n_days"]
print(f"Period: {pre['dates'][0].date()} ~ {pre['dates'][-1].date()}")

# 修改版 cpu_replay：可選買第幾名
def cpu_replay_rank(pre, p, buy_rank=1):
    """buy_rank=1 買第一名（原版），buy_rank=2 買第二名，以此類推"""
    ns, nd = pre["n_stocks"], pre["n_days"]
    tickers = pre["tickers"]; dates = pre["dates"]; close = pre["close"]
    top100_mask = pre.get("top100_mask")
    rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
    macd_hist = pre["macd_hist"]; macd_line = pre["macd_line"]
    k_val = pre["k_val"]; d_val = pre["d_val"]; williams_r = pre["williams_r"]
    is_green = pre["is_green"]; gap = pre["gap"]; near_high = pre["near_high"]
    vol_prev = pre["vol_prev"]
    squeeze_fire = pre["squeeze_fire"]; new_high_60 = pre["new_high_60"]
    adx_arr = pre["adx"]; bias_arr = pre["bias"]; obv_rising_arr = pre["obv_rising"]
    atr_pct_arr = pre["atr_pct"]
    opn = pre.get("open"); market_bull = pre.get("market_bull")
    sector_hot = pre.get("sector_hot")
    up_days_arr = pre.get("up_days"); week52_arr = pre.get("week52_pos")
    vol_up_days_arr = pre.get("vol_up_days"); mom_accel_arr = pre.get("mom_accel")
    maf = pre["ma_d"].get(int(p.get("ma_fast_w", 5)), pre["ma_d"][5])
    mas = pre["ma_d"].get(int(p.get("ma_slow_w", 20)), pre["ma_d"][20])
    ma60 = pre["ma60"]
    mom = pre["mom_d"].get(int(p.get("momentum_days", 5)), pre["mom_d"][5])

    max_pos = int(p.get("max_positions", 2))
    hold_si = [-1]*3; hold_bp = [0.0]*3; hold_pk = [0.0]*3; hold_bd = [0]*3
    n_holding = 0; trades = []

    def _score(si, day):
        sc = 0.0
        if int(p.get("w_rsi",0))>0 and rsi[si,day]>=p.get("rsi_th",55): sc+=int(p["w_rsi"])
        if int(p.get("w_bb",0))>0 and bb_pos[si,day]>=p.get("bb_th",0.7): sc+=int(p["w_bb"])
        if int(p.get("w_vol",0))>0 and vol_ratio[si,day]>=p.get("vol_th",3): sc+=int(p["w_vol"])
        if int(p.get("w_ma",0))>0 and close[si,day]>maf[si,day]: sc+=int(p["w_ma"])
        if int(p.get("w_macd",0))>0:
            mm=int(p.get("macd_mode",2)); ok=False
            if mm==0 and day>=1 and macd_hist[si,day]>0 and macd_hist[si,day-1]<=0: ok=True
            elif mm==1 and macd_line[si,day]>0: ok=True
            elif mm==2 and macd_hist[si,day]>0: ok=True
            if ok: sc+=int(p["w_macd"])
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
        if int(p.get("w_obv",0))>0 and obv_rising_arr[si,day]>0.5: sc+=int(p["w_obv"])
        if int(p.get("w_atr",0))>0 and atr_pct_arr[si,day]>=p.get("atr_min",2): sc+=int(p["w_atr"])
        if int(p.get("w_up_days",0))>0 and up_days_arr is not None and up_days_arr[si,day]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
        if int(p.get("w_week52",0))>0 and week52_arr is not None and week52_arr[si,day]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
        if int(p.get("w_vol_up_days",0))>0 and vol_up_days_arr is not None and vol_up_days_arr[si,day]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
        if int(p.get("w_mom_accel",0))>0 and mom_accel_arr is not None and mom_accel_arr[si,day]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
        cg=int(p.get("consecutive_green",0))
        if cg>=1:
            ok=True
            for g in range(cg):
                if day-g<0 or is_green[si,day-g]!=1: ok=False; break
            if ok: sc+=1
        if p.get("gap_up",0) and gap[si,day]>=1.0: sc+=1
        if p.get("above_ma60",0) and close[si,day]>=ma60[si,day]: sc+=1
        if p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]>vol_prev[si,day]: sc+=1
        return sc

    for day in range(60, nd-1):
        # SELL
        for h in range(max_pos):
            if hold_si[h]<0: continue
            si=hold_si[h]; cur=float(close[si,day]); dh=day-hold_bd[h]
            ret=(cur/hold_bp[h]-1)*100
            if dh<1: continue
            if cur>hold_pk[h]: hold_pk[h]=cur
            sell=False; reason=0
            eff_stop=p["stop_loss"]
            _is_be = p.get("use_breakeven",0) and (hold_pk[h]/hold_bp[h]-1)*100>=p.get("breakeven_trigger",20)
            if _is_be: eff_stop=0
            if ret<=eff_stop: sell=True; reason=14 if _is_be else 2
            if not sell and p.get("use_take_profit",1) and ret>=p["take_profit"]: sell=True; reason=1
            if not sell and p.get("trailing_stop",0)>0 and hold_pk[h]>hold_bp[h]:
                if (cur/hold_pk[h]-1)*100<=-p["trailing_stop"]: sell=True; reason=4
            if not sell and p.get("use_profit_lock",0):
                pg=(hold_pk[h]/hold_bp[h]-1)*100
                if pg>=p.get("lock_trigger",30) and ret<p.get("lock_floor",10): sell=True; reason=11
            if not sell and dh>=int(p["hold_days"]): sell=True; reason=0
            if sell and day+1<nd:
                sp=float(opn[si,day+1]) if opn is not None else float(close[si,day])
                if np.isnan(sp) or sp<=0: sp=float(close[si,day])
                ar=(sp/hold_bp[h]-1)*100-0.585
                trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                    "buy_date":str(dates[hold_bd[h]].date()),"sell_date":str(dates[day+1].date()),
                    "buy_price":round(hold_bp[h],2),"sell_price":round(sp,2),
                    "return":round(ar,2),"days":day+1-hold_bd[h],"reason":REASON_NAMES[min(reason,len(REASON_NAMES)-1)]})
                hold_si[h]=-1; n_holding-=1

        # BUY — 改成買第 N 名
        if n_holding<max_pos and day+1<nd and (market_bull is None or market_bull[day]>0.5):
            candidates = []
            buy_th=p.get("buy_threshold",5)
            held_set=set(hh for hh in hold_si if hh>=0)
            for si in range(ns):
                if top100_mask is not None and top100_mask[si,day]<0.5: continue
                if si in held_set: continue
                sc = _score(si, day)
                vr = float(vol_ratio[si,day])
                if sc>=buy_th:
                    candidates.append((sc, vr, si))
            candidates.sort(key=lambda x: (-x[0], -x[1]))

            # 買第 buy_rank 名（1=原版，2=第二名）
            target_idx = buy_rank - 1
            if len(candidates) > target_idx:
                _, _, best_si = candidates[target_idx]
                for h in range(max_pos):
                    if hold_si[h]<0:
                        hold_si[h]=best_si; hold_bp[h]=float(close[best_si,day+1])
                        hold_pk[h]=hold_bp[h]; hold_bd[h]=day+1; n_holding+=1; break

    for h in range(max_pos):
        if hold_si[h]>=0:
            si=hold_si[h]; cur=float(close[si,nd-1])
            trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                "buy_date":str(dates[hold_bd[h]].date()),"sell_date":"",
                "buy_price":round(hold_bp[h],2),"sell_price":round(cur,2),
                "return":round((cur/hold_bp[h]-1)*100-0.585,2),"days":nd-1-hold_bd[h],"reason":"持有中"})
    return sorted(trades, key=lambda x: x["buy_date"])

# 跑比較
print()
print("=" * 75)
print(f"{'排名':<6} {'筆數':>5} {'總報酬%':>8} {'平均%':>6} {'勝率%':>6} {'MaxDD%':>7}")
print("=" * 75)

for rank in [1, 2, 3]:
    trades = cpu_replay_rank(pre, p, buy_rank=rank)
    trades = [t for t in trades if not math.isnan(t.get("return", 0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    n = len(completed)
    if n == 0:
        print(f"  #{rank}     NO TRADES")
        continue
    rets = [t["return"] for t in completed]
    total = sum(rets)
    avg = total / n
    wins = sum(1 for r in rets if r > 0)
    wr = wins / n * 100
    run_dd = 0; max_dd = 0
    for r in rets:
        if r < 0: run_dd += r
        else: run_dd = 0
        if run_dd < max_dd: max_dd = run_dd
    print(f"  #{rank}   {n:>5} {total:>8.1f} {avg:>6.1f} {wr:>6.1f} {max_dd:>7.1f}")

print("=" * 75)
print("Done!")
