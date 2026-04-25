"""實驗：加入 3 個新指標，測試是否能突破 89.90
新指標：
1. MFI (Money Flow Index) — 量價合一的 RSI，比單獨看 RSI + volume 更精準
2. CMF (Chaikin Money Flow) — 機構累積/出貨指標，比 OBV 更乾淨
3. ATR Contraction (ATR5/ATR20) — 波動收縮 = 即將爆發，跟 squeeze 不同維度

測法：以 89.90 為基底，加入新指標 scoring，看勝率/報酬能否提升
用法：python experiment_new_indicators.py
"""
import sys, types, json, math
import numpy as np

mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name

# === Load data ===
print("Loading data...")
data = download_data()
_lens = [len(v) for v in data.values()]
_n_1500 = sum(1 for l in _lens if l >= 1500)
TARGET_DAYS = 1500 if _n_1500 >= 500 else 900
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
print(f"Stocks: {len(data)} | Days: {TARGET_DAYS}")

# === Precompute (standard) ===
print("Precomputing standard indicators...")
pre = precompute(data)
ns, nd = pre["n_stocks"], pre["n_days"]
print(f"Period: {pre['dates'][0].date()} ~ {pre['dates'][-1].date()} | {ns} stocks x {nd} days")

# === Compute NEW indicators ===
print("Computing new indicators (MFI, CMF, ATR contraction)...")
close = pre["close"]
high = np.zeros_like(close)
low = np.zeros_like(close)
volume = np.zeros_like(close)
tickers = list(data.keys())
ml = nd
for si, t in enumerate(tickers):
    h = data[t]
    high[si] = h["High"].values[-ml:].astype(np.float32)
    low[si] = h["Low"].values[-ml:].astype(np.float32)
    volume[si] = h["Volume"].values[-ml:].astype(np.float32)

# --- MFI (Money Flow Index, 14-period) ---
# Typical Price = (H + L + C) / 3
# Raw MF = TP * Volume
# MFI = 100 - 100/(1 + positive_mf_sum/negative_mf_sum)
tp = (high + low + close) / 3.0
raw_mf = tp * volume
mfi = np.full((ns, ml), 50.0, dtype=np.float32)
for i in range(15, ml):
    pos_mf = np.zeros(ns)
    neg_mf = np.zeros(ns)
    for j in range(i-13, i+1):
        up = tp[:, j] > tp[:, j-1]
        pos_mf += np.where(up, raw_mf[:, j], 0)
        neg_mf += np.where(~up, raw_mf[:, j], 0)
    ratio = np.where(neg_mf > 0, pos_mf / neg_mf, 100.0)
    mfi[:, i] = (100 - 100 / (1 + ratio)).astype(np.float32)
print(f"  MFI: done (sample last day mean={mfi[:, -1].mean():.1f})")

# --- CMF (Chaikin Money Flow, 20-period) ---
# CLV = ((C - L) - (H - C)) / (H - L)  (close location value, -1 to +1)
# CMF = sum(CLV * volume, 20) / sum(volume, 20)
hl_range = high - low
clv = np.where(hl_range > 0, ((close - low) - (high - close)) / hl_range, 0.0)
clv_vol = clv * volume
cmf = np.zeros((ns, ml), dtype=np.float32)
for i in range(20, ml):
    vol_sum = volume[:, i-19:i+1].sum(axis=1)
    clv_vol_sum = clv_vol[:, i-19:i+1].sum(axis=1)
    cmf[:, i] = np.where(vol_sum > 0, clv_vol_sum / vol_sum, 0).astype(np.float32)
print(f"  CMF: done (sample last day mean={cmf[:, -1].mean():.3f})")

# --- ATR Contraction (ATR5 / ATR20, <0.8 = 壓縮即將爆發) ---
tr = np.zeros_like(close)
tr[:, 1:] = np.maximum(high[:, 1:] - low[:, 1:],
    np.maximum(np.abs(high[:, 1:] - close[:, :-1]), np.abs(low[:, 1:] - close[:, :-1])))
atr5 = np.zeros_like(close)
atr20 = np.zeros_like(close)
for i in range(5, ml):
    atr5[:, i] = tr[:, i-4:i+1].mean(axis=1)
for i in range(20, ml):
    atr20[:, i] = tr[:, i-19:i+1].mean(axis=1)
atr_ratio = np.where(atr20 > 0, atr5 / atr20, 1.0).astype(np.float32)
print(f"  ATR ratio: done (sample last day mean={atr_ratio[:, -1].mean():.2f})")

# === Modified cpu_replay with new indicators ===
def cpu_replay_with_new(pre, p, mfi_arr, cmf_arr, atr_ratio_arr,
                         w_mfi=0, mfi_th=70,
                         w_cmf=0, cmf_th=0.1,
                         w_atr_contract=0, atr_contract_th=0.8):
    """Run cpu_replay but inject new indicator scores into buy decision."""
    # We'll monkey-patch: run normal cpu_replay logic but with modified scoring
    # Strategy: copy 89.90's cpu_replay but add new indicator checks in buy phase

    ns, nd = pre["n_stocks"], pre["n_days"]
    tickers_list = pre["tickers"]; dates = pre["dates"]; close_arr = pre["close"]
    top100_mask = pre.get("top100_mask")
    rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
    macd_hist = pre["macd_hist"]; macd_line = pre["macd_line"]
    k_val = pre["k_val"]; d_val = pre["d_val"]; williams_r = pre["williams_r"]
    is_green = pre["is_green"]; gap = pre["gap"]; near_high = pre["near_high"]
    vol_prev = pre["vol_prev"]
    squeeze_fire = pre["squeeze_fire"]; new_high_60 = pre["new_high_60"]
    adx_arr = pre["adx"]; bias_arr = pre["bias"]; obv_rising_arr = pre["obv_rising"]
    atr_pct_arr = pre["atr_pct"]
    opn = pre.get("open")
    market_bull = pre.get("market_bull")
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

    def _score_stock(si, day):
        sc = 0.0
        if int(p.get("w_rsi",0))>0 and rsi[si,day]>=p.get("rsi_th",55): sc+=int(p["w_rsi"])
        if int(p.get("w_bb",0))>0 and bb_pos[si,day]>=p.get("bb_th",0.7): sc+=int(p["w_bb"])
        if int(p.get("w_vol",0))>0 and vol_ratio[si,day]>=p.get("vol_th",3): sc+=int(p["w_vol"])
        if int(p.get("w_ma",0))>0 and close_arr[si,day]>maf[si,day]: sc+=int(p["w_ma"])
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
        if p.get("above_ma60",0) and close_arr[si,day]>=ma60[si,day]: sc+=1
        if p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]>vol_prev[si,day]: sc+=1
        # === NEW INDICATORS ===
        if w_mfi > 0 and mfi_arr[si, day] >= mfi_th: sc += w_mfi
        if w_cmf > 0 and cmf_arr[si, day] >= cmf_th: sc += w_cmf
        if w_atr_contract > 0 and atr_ratio_arr[si, day] <= atr_contract_th: sc += w_atr_contract
        return sc

    for day in range(60, nd-1):
        # SELL (identical to cpu_replay)
        for h in range(max_pos):
            if hold_si[h]<0: continue
            si=hold_si[h]; cur=float(close_arr[si,day]); dh=day-hold_bd[h]
            ret=(cur/hold_bp[h]-1)*100
            if dh<1: continue
            if cur>hold_pk[h]: hold_pk[h]=cur
            sell=False; reason=""
            eff_stop=p["stop_loss"]
            _is_be = p.get("use_breakeven",0) and (hold_pk[h]/hold_bp[h]-1)*100>=p.get("breakeven_trigger",20)
            if _is_be: eff_stop=0
            if ret<=eff_stop: sell=True; reason="保本出場" if _is_be else "停損"
            if not sell and p.get("use_take_profit",1) and ret>=p["take_profit"]: sell=True; reason="停利"
            if not sell and p.get("trailing_stop",0)>0 and hold_pk[h]>hold_bp[h]:
                if (cur/hold_pk[h]-1)*100<=-p["trailing_stop"]: sell=True; reason="移動停利"
            if not sell and p.get("use_profit_lock",0):
                pg=(hold_pk[h]/hold_bp[h]-1)*100
                if pg>=p.get("lock_trigger",30) and ret<p.get("lock_floor",10): sell=True; reason="鎖利"
            if not sell and dh>=int(p["hold_days"]): sell=True; reason="到期"
            if sell and day+1<nd:
                sp=float(opn[si,day+1]) if opn is not None else float(close_arr[si,day])
                if np.isnan(sp) or sp<=0: sp=float(close_arr[si,day])
                ar=(sp/hold_bp[h]-1)*100-0.585
                trades.append({"ticker":tickers_list[si],"name":get_name(tickers_list[si]),
                    "buy_date":str(dates[hold_bd[h]].date()),"sell_date":str(dates[day+1].date()),
                    "buy_price":round(hold_bp[h],2),"sell_price":round(sp,2),
                    "return":round(ar,2),"days":day+1-hold_bd[h],"reason":reason})
                hold_si[h]=-1; n_holding-=1

        # BUY
        if n_holding<max_pos and day+1<nd and (market_bull is None or market_bull[day]>0.5):
            best_si=-1; best_sc=0; best_vol=0
            buy_th=p.get("buy_threshold",5)
            held_set=set(hh for hh in hold_si if hh>=0)
            for si in range(ns):
                if top100_mask is not None and top100_mask[si,day]<0.5: continue
                if si in held_set: continue
                sc = _score_stock(si, day)
                vr = float(vol_ratio[si,day])
                if sc>=buy_th and (sc>best_sc or (sc==best_sc and vr>best_vol)):
                    best_si=si; best_sc=sc; best_vol=vr
            if best_si>=0:
                for h in range(max_pos):
                    if hold_si[h]<0:
                        hold_si[h]=best_si; hold_bp[h]=float(close_arr[best_si,day+1])
                        hold_pk[h]=hold_bp[h]; hold_bd[h]=day+1; n_holding+=1; break

    # Holdings
    for h in range(max_pos):
        if hold_si[h]>=0:
            si=hold_si[h]; cur=float(close_arr[si,nd-1])
            trades.append({"ticker":tickers_list[si],"name":get_name(tickers_list[si]),
                "buy_date":str(dates[hold_bd[h]].date()),"sell_date":"",
                "buy_price":round(hold_bp[h],2),"sell_price":round(cur,2),
                "return":round((cur/hold_bp[h]-1)*100-0.585,2),"days":nd-1-hold_bd[h],"reason":"持有中"})
    return sorted(trades, key=lambda x: x["buy_date"])

# === 89.90 base params ===
BASE = {"w_rsi": 3.0, "rsi_th": 70.0, "w_bb": 3.0, "bb_th": 0.95, "w_vol": 0.0, "vol_th": 2.5, "w_ma": 2.0, "w_macd": 3.0, "macd_mode": 0.0, "w_kd": 2.0, "kd_th": 80.0, "kd_cross": 0.0, "w_wr": 0.0, "wr_th": -50.0, "w_mom": 3.0, "mom_th": 8.0, "w_near_high": 2.0, "near_high_pct": 10.0, "w_squeeze": 0.0, "w_new_high": 1.0, "w_adx": 2.0, "adx_th": 40.0, "consecutive_green": 1.0, "gap_up": 1.0, "above_ma60": 0.0, "vol_gt_yesterday": 0.0, "buy_threshold": 8.0, "stop_loss": -20.0, "use_take_profit": 1.0, "take_profit": 40.0, "trailing_stop": 20.0, "use_rsi_sell": 0.0, "rsi_sell": 70.0, "use_macd_sell": 0.0, "use_kd_sell": 0.0, "sell_vol_shrink": 0.0, "sell_below_ma": 0.0, "hold_days": 30.0, "w_bias": 1.0, "bias_max": 5.0, "use_stagnation_exit": 0.0, "stagnation_days": 7.0, "stagnation_min_ret": 3.0, "use_breakeven": 1.0, "breakeven_trigger": 10.0, "w_obv": 2.0, "obv_rising_days": 3.0, "w_atr": 1.0, "atr_min": 3.0, "use_time_decay": 0.0, "ret_per_day": 0.8, "use_profit_lock": 1.0, "lock_trigger": 20.0, "lock_floor": 3.0, "use_mom_exit": 0.0, "mom_exit_th": 3.0, "upgrade_margin": 0.0, "max_positions": 2.0, "w_sector_flow": 0.0, "sector_flow_topn": 8.0, "w_up_days": 2.0, "up_days_min": 5.0, "w_week52": 1.0, "week52_min": 0.7, "w_vol_up_days": 1.0, "vol_up_days_min": 2.0, "w_mom_accel": 2.0, "mom_accel_min": 0.0, "ma_fast_w": 3, "ma_slow_w": 15, "momentum_days": 3}

# === 實驗組合 ===
EXPERIMENTS = [
    # baseline (no new indicators)
    {"name": "89.90 baseline", "w_mfi": 0, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 0, "atr_contract_th": 0.8},
    # MFI only
    {"name": "MFI+1 (>=60)", "w_mfi": 1, "mfi_th": 60, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "MFI+2 (>=60)", "w_mfi": 2, "mfi_th": 60, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "MFI+2 (>=70)", "w_mfi": 2, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "MFI+3 (>=70)", "w_mfi": 3, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "MFI+2 (>=80)", "w_mfi": 2, "mfi_th": 80, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 0, "atr_contract_th": 0.8},
    # CMF only
    {"name": "CMF+1 (>=0.05)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 1, "cmf_th": 0.05, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "CMF+2 (>=0.05)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.05, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "CMF+2 (>=0.10)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.10, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "CMF+3 (>=0.10)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 3, "cmf_th": 0.10, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "CMF+2 (>=0.15)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.15, "w_atr_contract": 0, "atr_contract_th": 0.8},
    # ATR contraction only
    {"name": "ATRc+1 (<=0.85)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 1, "atr_contract_th": 0.85},
    {"name": "ATRc+2 (<=0.85)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 2, "atr_contract_th": 0.85},
    {"name": "ATRc+2 (<=0.75)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 2, "atr_contract_th": 0.75},
    {"name": "ATRc+3 (<=0.75)", "w_mfi": 0, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 3, "atr_contract_th": 0.75},
    # Combos (best of each)
    {"name": "MFI2+CMF2", "w_mfi": 2, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.10, "w_atr_contract": 0, "atr_contract_th": 0.8},
    {"name": "MFI2+ATRc2", "w_mfi": 2, "mfi_th": 70, "w_cmf": 0, "cmf_th": 0.1, "w_atr_contract": 2, "atr_contract_th": 0.80},
    {"name": "CMF2+ATRc2", "w_mfi": 0, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.10, "w_atr_contract": 2, "atr_contract_th": 0.80},
    {"name": "ALL3 (MFI2+CMF2+ATRc2)", "w_mfi": 2, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.10, "w_atr_contract": 2, "atr_contract_th": 0.80},
    {"name": "ALL3 heavy (3+3+3)", "w_mfi": 3, "mfi_th": 70, "w_cmf": 3, "cmf_th": 0.10, "w_atr_contract": 3, "atr_contract_th": 0.80},
    # Combos with higher buy_threshold (because new indicators add score, threshold should rise)
    {"name": "ALL3+bt12", "w_mfi": 2, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.10, "w_atr_contract": 2, "atr_contract_th": 0.80, "bt_override": 12},
    {"name": "ALL3+bt14", "w_mfi": 2, "mfi_th": 70, "w_cmf": 2, "cmf_th": 0.10, "w_atr_contract": 2, "atr_contract_th": 0.80, "bt_override": 14},
    {"name": "ALL3 heavy+bt14", "w_mfi": 3, "mfi_th": 70, "w_cmf": 3, "cmf_th": 0.10, "w_atr_contract": 3, "atr_contract_th": 0.80, "bt_override": 14},
    {"name": "ALL3 heavy+bt16", "w_mfi": 3, "mfi_th": 70, "w_cmf": 3, "cmf_th": 0.10, "w_atr_contract": 3, "atr_contract_th": 0.80, "bt_override": 16},
]

# === Run experiments ===
print()
print("=" * 90)
print(f"{'實驗名稱':<22} | {'筆數':>4} {'總報酬%':>8} {'平均%':>6} {'勝率%':>6} {'MaxDD%':>7} | vs 89.90")
print("=" * 90)

for i, exp in enumerate(EXPERIMENTS):
    p = dict(BASE)
    if "bt_override" in exp:
        p["buy_threshold"] = exp["bt_override"]

    trades = cpu_replay_with_new(pre, p, mfi, cmf, atr_ratio,
                                  w_mfi=exp["w_mfi"], mfi_th=exp["mfi_th"],
                                  w_cmf=exp["w_cmf"], cmf_th=exp["cmf_th"],
                                  w_atr_contract=exp["w_atr_contract"], atr_contract_th=exp["atr_contract_th"])
    trades = [t for t in trades if not math.isnan(t.get("return", 0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    n = len(completed)

    if n < 5:
        print(f"  {exp['name']:<20} | NO TRADES")
        continue

    rets = [t["return"] for t in completed]
    total_r = sum(rets)
    avg_r = total_r / n
    wins = sum(1 for r in rets if r > 0)
    wr = wins / n * 100
    run_dd = 0; max_dd = 0
    for r in rets:
        if r < 0: run_dd += r
        else: run_dd = 0
        if run_dd < max_dd: max_dd = run_dd

    # Compare to baseline
    delta_wr = wr - 69.4
    delta_total = total_r - 2134.5
    if delta_wr > 2 and delta_total > -200:
        verdict = f"WR+{delta_wr:.1f} <<<BETTER"
    elif delta_wr > 0 and delta_total > 0:
        verdict = f"WR+{delta_wr:.1f} RET+{delta_total:.0f}"
    elif delta_wr > 0:
        verdict = f"WR+{delta_wr:.1f} RET{delta_total:.0f}"
    elif delta_wr < -3:
        verdict = "WORSE"
    else:
        verdict = f"WR{delta_wr:+.1f} RET{delta_total:+.0f}"

    print(f"  {exp['name']:<20} | {n:>4} {total_r:>8.1f} {avg_r:>6.1f} {wr:>6.1f} {max_dd:>7.1f} | {verdict}")

print("=" * 90)
print()
print("Done!")
