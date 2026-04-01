"""
驗證 cpu_replay 和 realistic_test 邏輯是否完全一致
用同一份資料 + 同一組策略跑兩邊，逐筆比對
"""
import numpy as np
import json, os, sys, types

mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name

# === 載入資料和策略 ===
data = download_data()
data = {k:v for k,v in data.items() if len(v) >= 400}
with open("best_strategy.json", encoding="utf-8") as f:
    strategy = json.load(f)
p = strategy["params"]

all_tickers = sorted(data.keys())
min_len = min(len(data[t]) for t in all_tickers)
n_stocks = len(all_tickers)
print(f"股票：{n_stocks} 檔 | 天數：{min_len}")

# === 方法 A: cpu_replay ===
pre = precompute(data)
trades_a = cpu_replay(p, pre)
trades_a.sort(key=lambda x: x["buy_date"])

# === 方法 B: realistic_test 邏輯 ===
nd = min_len
close = np.zeros((n_stocks, nd), dtype=np.float32)
high = np.zeros((n_stocks, nd), dtype=np.float32)
low = np.zeros((n_stocks, nd), dtype=np.float32)
opn = np.zeros((n_stocks, nd), dtype=np.float32)
volume = np.zeros((n_stocks, nd), dtype=np.float32)
dates = None
for si, t in enumerate(all_tickers):
    h = data[t]
    close[si] = h["Close"].values[-nd:].astype(np.float32)
    high[si] = h["High"].values[-nd:].astype(np.float32)
    low[si] = h["Low"].values[-nd:].astype(np.float32)
    opn[si] = h["Open"].values[-nd:].astype(np.float32)
    volume[si] = h["Volume"].values[-nd:].astype(np.float32)
    if dates is None: dates = h.index[-nd:]

# 指標（跟 realistic_test.py 一模一樣）
delta = np.diff(close, axis=1)
gain = np.where(delta > 0, delta, 0)
loss_a_arr = np.where(delta < 0, -delta, 0)
ag = np.zeros_like(close); al = np.zeros_like(close)
for i in range(14, nd):
    if i == 14: ag[:, i] = np.mean(gain[:, :14], axis=1); al[:, i] = np.mean(loss_a_arr[:, :14], axis=1)
    else: ag[:, i] = (ag[:, i-1]*13 + gain[:, i-1])/14; al[:, i] = (al[:, i-1]*13 + loss_a_arr[:, i-1])/14
rs = np.where(al > 0, ag/al, 100)
rsi = (100 - 100/(1+rs)).astype(np.float32)
ma_d = {}
for w in [3, 5, 10, 15, 20, 30, 60]:
    ma = np.zeros_like(close)
    for i in range(w, nd): ma[:, i] = np.mean(close[:, i-w:i], axis=1)
    ma_d[w] = ma.astype(np.float32)
bb_std = np.zeros_like(close)
for i in range(20, nd): bb_std[:, i] = np.std(close[:, i-20:i], axis=1)
bb_u = ma_d[20] + 2*bb_std; bb_l = ma_d[20] - 2*bb_std; bb_r = bb_u - bb_l
bb_pos = np.where(bb_r > 0, (close - bb_l)/bb_r, 0.5).astype(np.float32)
vol_ma = np.zeros_like(volume)
for i in range(20, nd): vol_ma[:, i] = np.mean(volume[:, i-20:i], axis=1)
vol_ratio = np.where(vol_ma > 0, volume/vol_ma, 1).astype(np.float32)
vol_prev = np.zeros_like(vol_ratio); vol_prev[:, 1:] = vol_ratio[:, :-1]
e12 = np.zeros_like(close); e26 = np.zeros_like(close)
e12[:, 0] = close[:, 0]; e26[:, 0] = close[:, 0]
for i in range(1, nd):
    e12[:, i] = e12[:, i-1]*(1-2/13) + close[:, i]*2/13
    e26[:, i] = e26[:, i-1]*(1-2/27) + close[:, i]*2/27
ml_arr = (e12 - e26).astype(np.float32)
ms = np.zeros_like(close); ms[:, 0] = ml_arr[:, 0]
for i in range(1, nd): ms[:, i] = ms[:, i-1]*(1-2/10) + ml_arr[:, i]*2/10
mh = (ml_arr - ms).astype(np.float32)
l9 = np.zeros_like(close); h9 = np.zeros_like(close)
for i in range(9, nd):
    l9[:, i] = np.min(low[:, i-9:i+1], axis=1); h9[:, i] = np.max(high[:, i-9:i+1], axis=1)
rsv = np.where((h9-l9) > 0, (close-l9)/(h9-l9)*100, 50)
kv = np.zeros_like(close); dv = np.zeros_like(close); kv[:, 0] = 50; dv[:, 0] = 50
for i in range(1, nd):
    kv[:, i] = kv[:, i-1]*2/3 + rsv[:, i]*1/3
    dv[:, i] = dv[:, i-1]*2/3 + kv[:, i]*1/3
mom_d = {}
for d in [3, 5, 10]:
    m = np.zeros_like(close); m[:, d:] = (close[:, d:]/close[:, :-d]-1)*100
    mom_d[d] = m.astype(np.float32)
ig = (close > opn).astype(np.float32)
gp = np.zeros_like(close); gp[:, 1:] = (opn[:, 1:]/close[:, :-1]-1)*100
l14 = np.zeros_like(close); h14 = np.zeros_like(close)
for i in range(14, nd):
    l14[:, i] = np.min(low[:, i-14:i+1], axis=1); h14[:, i] = np.max(high[:, i-14:i+1], axis=1)
wr = np.where((h14-l14) > 0, (h14-close)/(h14-l14)*-100, -50).astype(np.float32)
h20 = np.zeros_like(close)
for i in range(20, nd): h20[:, i] = np.max(high[:, i-20:i+1], axis=1)
nh = np.where(h20 > 0, (close/h20-1)*100, 0).astype(np.float32)
tr = np.zeros_like(close)
tr[:, 1:] = np.maximum(high[:, 1:]-low[:, 1:], np.maximum(np.abs(high[:, 1:]-close[:, :-1]), np.abs(low[:, 1:]-close[:, :-1])))
atr = np.zeros_like(close)
for i in range(1, nd):
    if i <= 14: atr[:, i] = np.mean(tr[:, 1:min(i+1, 15)], axis=1)
    else: atr[:, i] = (atr[:, i-1]*13 + tr[:, i])/14
kc_u = ma_d[20] + 1.5*atr; kc_l = ma_d[20] - 1.5*atr
sq_on = ((ma_d[20]+2*bb_std < kc_u) & (ma_d[20]-2*bb_std > kc_l)).astype(np.float32)
squeeze_fire = np.zeros_like(close)
squeeze_fire[:, 1:] = ((sq_on[:, :-1]==1) & (sq_on[:, 1:]==0)).astype(np.float32)
squeeze_fire = (squeeze_fire * (mh > 0)).astype(np.float32)
h60 = np.zeros_like(close)
for i in range(60, nd): h60[:, i] = np.max(high[:, i-60:i], axis=1)
new_high_60 = (close > h60).astype(np.float32)
plus_dm = np.zeros_like(close); minus_dm = np.zeros_like(close)
for i in range(1, nd):
    up = high[:, i]-high[:, i-1]; dn = low[:, i-1]-low[:, i]
    plus_dm[:, i] = np.where((up > dn) & (up > 0), up, 0)
    minus_dm[:, i] = np.where((dn > up) & (dn > 0), dn, 0)
atr14 = np.zeros_like(close); sp_v = np.zeros_like(close); sm_v = np.zeros_like(close)
for i in range(14, nd):
    if i == 14:
        atr14[:, i] = np.mean(tr[:, 1:15], axis=1)
        sp_v[:, i] = np.mean(plus_dm[:, 1:15], axis=1)
        sm_v[:, i] = np.mean(minus_dm[:, 1:15], axis=1)
    else:
        atr14[:, i] = (atr14[:, i-1]*13 + tr[:, i])/14
        sp_v[:, i] = (sp_v[:, i-1]*13 + plus_dm[:, i])/14
        sm_v[:, i] = (sm_v[:, i-1]*13 + minus_dm[:, i])/14
pdi = np.where(atr14 > 0, sp_v/atr14*100, 0)
mdi = np.where(atr14 > 0, sm_v/atr14*100, 0)
dx = np.where((pdi+mdi) > 0, np.abs(pdi-mdi)/(pdi+mdi)*100, 0)
adx = np.zeros_like(close)
for i in range(28, nd):
    if i == 28: adx[:, i] = np.mean(dx[:, 14:29], axis=1)
    else: adx[:, i] = (adx[:, i-1]*13 + dx[:, i])/14
bias = np.where(ma_d[20] > 0, (close-ma_d[20])/ma_d[20]*100, 0).astype(np.float32)
atr_pct = np.where(close > 0, atr/close*100, 0).astype(np.float32)
price_sign = np.sign(np.diff(close, axis=1))
vol_signed = np.zeros_like(close); vol_signed[:, 1:] = volume[:, 1:]*price_sign
obv = np.cumsum(vol_signed, axis=1).astype(np.float64)
obv_rising = np.zeros_like(close)
for d in [3, 5, 10]:
    r = np.zeros_like(close); r[:, d:] = (obv[:, d:] > obv[:, :-d]).astype(np.float32)
    obv_rising += r
obv_rising = (obv_rising > 0).astype(np.float32)

ma_fast_w = int(p.get("ma_fast_w", 5))
ma_slow_w = int(p.get("ma_slow_w", 20))
mom_days_val = int(p.get("momentum_days", 5))
maf = ma_d.get(ma_fast_w, ma_d[5])
mas = ma_d.get(ma_slow_w, ma_d[20])
ma60 = ma_d[60]
mom = mom_d.get(mom_days_val, mom_d[5])

max_pos = int(p.get("max_positions", 1))
if max_pos < 1: max_pos = 1
if max_pos > 3: max_pos = 3
hold_si = [-1]*3; hold_bp = [0.0]*3; hold_pk = [0.0]*3; hold_bd = [0]*3
n_holding = 0
trades_b = []
TXCOST = 0.585
REASON_NAMES = ["到期","停利","停損","RSI超買","移動停利","MACD死叉","KD死叉","量縮","跌破均線","停滯出場","漸進停利","鎖利出場","動量反轉","換股"]

def _score(si, day):
    sc = 0.0
    if int(p.get("w_rsi",0))>0 and rsi[si,day]>=p.get("rsi_th",55): sc+=int(p["w_rsi"])
    if int(p.get("w_bb",0))>0 and bb_pos[si,day]>=p.get("bb_th",0.7): sc+=int(p["w_bb"])
    if int(p.get("w_vol",0))>0 and vol_ratio[si,day]>=p.get("vol_th",3): sc+=int(p["w_vol"])
    if int(p.get("w_ma",0))>0 and close[si,day]>maf[si,day]: sc+=int(p["w_ma"])
    if int(p.get("w_macd",0))>0:
        mm=int(p.get("macd_mode",2)); ok=False
        if mm==0 and day>=1 and mh[si,day]>0 and mh[si,day-1]<=0: ok=True
        elif mm==1 and ml_arr[si,day]>0: ok=True
        elif mm==2 and mh[si,day]>0: ok=True
        if ok: sc+=int(p["w_macd"])
    if int(p.get("w_kd",0))>0:
        ok=kv[si,day]>=p.get("kd_th",50)
        if ok and p.get("kd_cross",0) and day>=1: ok=kv[si,day]>dv[si,day] and kv[si,day-1]<=dv[si,day-1]
        if ok: sc+=int(p["w_kd"])
    if int(p.get("w_wr",0))>0 and wr[si,day]>=p.get("wr_th",-30): sc+=int(p["w_wr"])
    if int(p.get("w_mom",0))>0 and mom[si,day]>=p.get("mom_th",3): sc+=int(p["w_mom"])
    if int(p.get("w_near_high",0))>0 and abs(nh[si,day])<=p.get("near_high_pct",10): sc+=int(p["w_near_high"])
    if int(p.get("w_squeeze",0))>0 and squeeze_fire[si,day]>0.5: sc+=int(p["w_squeeze"])
    if int(p.get("w_new_high",0))>0 and new_high_60[si,day]>0.5: sc+=int(p["w_new_high"])
    if int(p.get("w_adx",0))>0 and adx[si,day]>=p.get("adx_th",25): sc+=int(p["w_adx"])
    if int(p.get("w_bias",0))>0 and bias[si,day]>=0 and bias[si,day]<=p.get("bias_max",15): sc+=int(p["w_bias"])
    if int(p.get("w_obv",0))>0 and obv_rising[si,day]>0.5: sc+=int(p["w_obv"])
    if int(p.get("w_atr",0))>0 and atr_pct[si,day]>=p.get("atr_min",2): sc+=int(p["w_atr"])
    cg=int(p.get("consecutive_green",0))
    if cg>=1:
        ok=True
        for g in range(cg):
            if day-g<0 or ig[si,day-g]!=1: ok=False; break
        if ok: sc+=1
    if p.get("gap_up",0) and gp[si,day]>=1.0: sc+=1
    if p.get("above_ma60",0) and close[si,day]>=ma60[si,day]: sc+=1
    if p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]>vol_prev[si,day]: sc+=1
    return sc

for day in range(60, nd - 1):
    day_vol = volume[:, day]
    top100_idx = set(np.argsort(day_vol)[-100:])
    # Phase 1: 賣出
    for h in range(3):
        if hold_si[h] < 0: continue
        si = hold_si[h]; cur = float(close[si, day]); dh = day - hold_bd[h]
        ret = (cur/hold_bp[h] - 1)*100
        if dh < 1: continue
        if cur > hold_pk[h]: hold_pk[h] = cur
        sell = False; reason = 0
        eff_stop = p["stop_loss"]
        if p.get("use_breakeven",0) and (hold_pk[h]/hold_bp[h]-1)*100>=p.get("breakeven_trigger",20): eff_stop=0
        if ret<=eff_stop: sell=True; reason=2
        if not sell and p.get("use_take_profit",1) and ret>=p["take_profit"]: sell=True; reason=1
        if not sell and p.get("trailing_stop",0)>0 and hold_pk[h]>hold_bp[h]:
            if (cur/hold_pk[h]-1)*100<=-p["trailing_stop"]: sell=True; reason=4
        if not sell and p.get("use_rsi_sell",1) and rsi[si,day]>=p.get("rsi_sell",90): sell=True; reason=3
        if not sell and p.get("use_macd_sell",0) and day>=1:
            if mh[si,day]<0 and mh[si,day-1]>=0: sell=True; reason=5
        if not sell and p.get("use_kd_sell",0) and day>=1:
            if kv[si,day]<dv[si,day] and kv[si,day-1]>=dv[si,day-1]: sell=True; reason=6
        if not sell and p.get("sell_vol_shrink",0)>0 and dh>=2 and vol_ratio[si,day]<p["sell_vol_shrink"]: sell=True; reason=7
        sbm=int(p.get("sell_below_ma",0))
        if not sell and sbm>0:
            mc=0
            if sbm==1: mc=maf[si,day]
            elif sbm==2: mc=mas[si,day]
            elif sbm==3: mc=ma60[si,day]
            if mc>0 and cur<mc: sell=True; reason=8
        if not sell and p.get("use_stagnation_exit",0) and dh>=int(p.get("stagnation_days",10)) and ret<p.get("stagnation_min_ret",5): sell=True; reason=9
        hd_half=int(p["hold_days"])//2
        if not sell and p.get("use_time_decay",0) and dh>=hd_half and ret<(dh-hd_half)*p.get("ret_per_day",0.5): sell=True; reason=10
        if not sell and p.get("use_profit_lock",0):
            pg=(hold_pk[h]/hold_bp[h]-1)*100
            if pg>=p.get("lock_trigger",30) and ret<p.get("lock_floor",10): sell=True; reason=11
        if not sell and p.get("use_mom_exit",0) and dh>=10 and mom[si,day]<-p.get("mom_exit_th",2): sell=True; reason=12
        if not sell and dh>=int(p["hold_days"]): sell=True; reason=0
        if sell and day+1<nd:
            sell_price = float(opn[si, day+1])
            actual_ret = (sell_price/hold_bp[h]-1)*100 - TXCOST
            actual_days = day+1-hold_bd[h]
            trades_b.append({"ticker":all_tickers[si],"name":get_name(all_tickers[si]),
                "buy_date":str(dates[hold_bd[h]].date()),"sell_date":str(dates[day+1].date()),
                "buy_price":round(hold_bp[h],2),"sell_price":round(sell_price,2),
                "return":round(actual_ret,2),"days":actual_days,
                "reason":REASON_NAMES[min(reason,len(REASON_NAMES)-1)]})
            hold_si[h]=-1; n_holding-=1
    # Phase 1.5: 換股
    um = int(p.get("upgrade_margin", 0))
    if um > 0 and n_holding >= max_pos and day+1 < nd:
        cand_si=-1; cand_sc=0
        held_set=set(hh for hh in hold_si if hh>=0)
        for si in top100_idx:
            if si in held_set: continue
            sc=_score(si,day)
            if sc>=p.get("buy_threshold",5) and sc>cand_sc: cand_si=si; cand_sc=sc
        if cand_si>=0:
            weakest_h=-1; weakest_sc=9999
            for h in range(max_pos):
                if hold_si[h]<0: continue
                sc=_score(hold_si[h],day)
                if sc<weakest_sc: weakest_sc=sc; weakest_h=h
            if weakest_h>=0 and cand_sc-weakest_sc>=um:
                si=hold_si[weakest_h]
                sell_price=float(opn[si,day+1])
                actual_ret=(sell_price/hold_bp[weakest_h]-1)*100 - TXCOST
                actual_days=day+1-hold_bd[weakest_h]
                trades_b.append({"ticker":all_tickers[si],"name":get_name(all_tickers[si]),
                    "buy_date":str(dates[hold_bd[weakest_h]].date()),"sell_date":str(dates[day+1].date()),
                    "buy_price":round(hold_bp[weakest_h],2),"sell_price":round(sell_price,2),
                    "return":round(actual_ret,2),"days":actual_days,"reason":"換股"})
                hold_si[weakest_h]=-1; n_holding-=1
    # Phase 2: 買入
    if n_holding < max_pos and day+1 < nd:
        best_si=-1; best_sc=0
        held_set=set(hh for hh in hold_si if hh>=0)
        for si in top100_idx:
            if si in held_set: continue
            sc=_score(si,day)
            if sc>=p.get("buy_threshold",5) and sc>best_sc: best_si=si; best_sc=sc
        if best_si>=0:
            for h in range(max_pos):
                if hold_si[h]<0:
                    hold_si[h]=best_si; hold_bp[h]=float(close[best_si,day+1])
                    hold_pk[h]=hold_bp[h]; hold_bd[h]=day+1; n_holding+=1; break

trades_b.sort(key=lambda x: x["buy_date"])

# === 比對 ===
print(f"\n{'='*60}")
print(f"cpu_replay：{len(trades_a)} 筆 | realistic：{len(trades_b)} 筆")
print(f"{'='*60}")

max_len = max(len(trades_a), len(trades_b))
match = 0; diff = 0
for i in range(max_len):
    a = trades_a[i] if i < len(trades_a) else None
    b = trades_b[i] if i < len(trades_b) else None
    if a and b and a["ticker"]==b["ticker"] and a["buy_date"]==b["buy_date"] and a["sell_date"]==b["sell_date"] and abs(a["return"]-b["return"])<0.1:
        match += 1
    else:
        diff += 1
        a_str = f"{a['name']}({a['ticker']}) {a['buy_date']}→{a['sell_date']} {a['return']:+.2f}% {a['reason']}" if a else "---"
        b_str = f"{b['name']}({b['ticker']}) {b['buy_date']}→{b['sell_date']} {b['return']:+.2f}% {b['reason']}" if b else "---"
        print(f"\n⚠️ 差異 #{diff}:")
        print(f"  cpu_replay: {a_str}")
        print(f"  realistic:  {b_str}")

print(f"\n{'='*60}")
print(f"一致：{match} 筆 | 差異：{diff} 筆")
if diff == 0:
    print("✅ 完全一致！GPU 和 realistic 結果相同。")
else:
    print(f"❌ 有 {diff} 筆不同，需要繼續抓 bug。")

total_a = sum(t["return"] for t in trades_a)
total_b = sum(t["return"] for t in trades_b)
print(f"\ncpu_replay 總報酬：{total_a:.1f}%")
print(f"realistic  總報酬：{total_b:.1f}%")
print(f"差距：{abs(total_a-total_b):.1f}%")

input("\n按 Enter 結束...")
