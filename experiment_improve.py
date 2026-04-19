"""實驗：解決 89.90 三個弱點
1. 勝率要更高 → 提高 buy_threshold（更挑剔才進場）
2. +40% 就停利太保守 → 拉高 take_profit 讓贏家跑
3. 買入看運氣 → 加 upgrade_margin（主動賣弱換強，不被動等觸發才買）

以 89.90 為基底，測 48 組組合。
用法：python experiment_improve.py
"""
import sys, types, json, math
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data

# 89.90 基底
BASE = {"w_rsi": 3.0, "rsi_th": 70.0, "w_bb": 3.0, "bb_th": 0.95, "w_vol": 0.0, "vol_th": 2.5, "w_ma": 2.0, "w_macd": 3.0, "macd_mode": 0.0, "w_kd": 2.0, "kd_th": 80.0, "kd_cross": 0.0, "w_wr": 0.0, "wr_th": -50.0, "w_mom": 3.0, "mom_th": 8.0, "w_near_high": 2.0, "near_high_pct": 10.0, "w_squeeze": 0.0, "w_new_high": 1.0, "w_adx": 2.0, "adx_th": 40.0, "consecutive_green": 1.0, "gap_up": 1.0, "above_ma60": 0.0, "vol_gt_yesterday": 0.0, "buy_threshold": 8.0, "stop_loss": -20.0, "use_take_profit": 1.0, "take_profit": 40.0, "trailing_stop": 20.0, "use_rsi_sell": 0.0, "rsi_sell": 70.0, "use_macd_sell": 0.0, "use_kd_sell": 0.0, "sell_vol_shrink": 0.0, "sell_below_ma": 0.0, "hold_days": 30.0, "w_bias": 1.0, "bias_max": 5.0, "use_stagnation_exit": 0.0, "stagnation_days": 7.0, "stagnation_min_ret": 3.0, "use_breakeven": 1.0, "breakeven_trigger": 10.0, "w_obv": 2.0, "obv_rising_days": 3.0, "w_atr": 1.0, "atr_min": 3.0, "use_time_decay": 0.0, "ret_per_day": 0.8, "use_profit_lock": 1.0, "lock_trigger": 20.0, "lock_floor": 3.0, "use_mom_exit": 0.0, "mom_exit_th": 3.0, "upgrade_margin": 0.0, "max_positions": 2.0, "w_sector_flow": 0.0, "sector_flow_topn": 8.0, "w_up_days": 2.0, "up_days_min": 5.0, "w_week52": 1.0, "week52_min": 0.7, "w_vol_up_days": 1.0, "vol_up_days_min": 2.0, "w_mom_accel": 2.0, "mom_accel_min": 0.0, "ma_fast_w": 3, "ma_slow_w": 15, "momentum_days": 3}

# === 實驗變數 ===
# 1. buy_threshold: 越高 = 越挑剔 = 更少交易但更高品質
BUY_TH = [8, 10, 12, 14]
# 2. take_profit: 越高 = 讓贏家跑更遠（40 是現狀）
TAKE_PROFIT = [40, 80, 150]
# 3. upgrade_margin: >0 = 主動賣弱換強（0 是現狀）
UPGRADE = [0, 5, 7, 10]
# 同時配合調整 breakeven_trigger（太早保本會讓 trailing 沒機會跑）
# take_profit 拉高時，breakeven_trigger 也要拉高，不然還沒到 80% 就被保本踢出
BREAKEVEN_MAP = {40: 10, 80: 20, 150: 30}

print("Loading data...")
data = download_data()
_lens = [len(v) for v in data.values()]
_n_1500 = sum(1 for l in _lens if l >= 1500)
TARGET_DAYS = 1500 if _n_1500 >= 500 else 900
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
print(f"Stocks: {len(data)} | Days: {TARGET_DAYS}")
print("Precomputing...")
pre = precompute(data)
print(f"Period: {pre['dates'][0].date()} ~ {pre['dates'][-1].date()}")
print()

# === 跑實驗 ===
results = []
total_exp = len(BUY_TH) * len(TAKE_PROFIT) * len(UPGRADE)
done = 0

for bt in BUY_TH:
    for tp in TAKE_PROFIT:
        for um in UPGRADE:
            p = dict(BASE)
            p["buy_threshold"] = bt
            p["take_profit"] = tp
            p["upgrade_margin"] = um
            # 配合調整 breakeven
            p["breakeven_trigger"] = BREAKEVEN_MAP.get(tp, 10)
            # take_profit 高時，lock_trigger 也要跟著高（不然鎖利太早）
            if tp >= 80:
                p["lock_trigger"] = 40
                p["lock_floor"] = 10
            if tp >= 150:
                p["lock_trigger"] = 60
                p["lock_floor"] = 20

            trades = cpu_replay(pre, p)
            trades = [t for t in trades if not math.isnan(t.get("return", 0))]
            completed = [t for t in trades if t.get("reason") != "持有中"]
            n = len(completed)
            done += 1

            if n < 10:
                results.append({"bt": bt, "tp": tp, "um": um, "n": n, "total": 0, "avg": 0, "wr": 0, "dd": 0})
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

            results.append({"bt": bt, "tp": tp, "um": um, "n": n, "total": total_r, "avg": avg_r, "wr": wr, "dd": max_dd})

            if done % 8 == 0:
                print(f"  進度 {done}/{total_exp}...", flush=True)

# === 排序輸出 ===
print()
print("=" * 85)
print(f"{'BuyTh':>5} {'TP':>4} {'UpgM':>4} | {'筆數':>4} {'總報酬%':>8} {'平均%':>6} {'勝率%':>6} {'MaxDD%':>7} | 評語")
print("=" * 85)

# 先按勝率排，同勝率按總報酬排
results.sort(key=lambda x: (x["wr"], x["total"]), reverse=True)

base_wr = 69.4
base_total = 2134.5
base_avg = 15.9

for r in results:
    # 評語
    tags = []
    if r["wr"] > base_wr + 3: tags.append("WR++")
    elif r["wr"] > base_wr: tags.append("WR+")
    if r["total"] > base_total: tags.append("RET+")
    if r["avg"] > base_avg + 3: tags.append("AVG++")
    elif r["avg"] > base_avg: tags.append("AVG+")
    if r["dd"] > -30: tags.append("SAFE")
    if r["wr"] < 60: tags.append("WEAK")
    if r["n"] < 50: tags.append("FEW")
    tag_str = " ".join(tags) if tags else ""

    marker = ">>>" if r["wr"] > base_wr and r["total"] > base_total * 0.8 else "   "
    print(f"{marker} {r['bt']:>3} {r['tp']:>4} {r['um']:>4} | {r['n']:>4} {r['total']:>8.1f} {r['avg']:>6.1f} {r['wr']:>6.1f} {r['dd']:>7.1f} | {tag_str}")

print("=" * 85)
print()
print("基準 89.90: bt=8 tp=40 um=0 | 134筆 2134.5% avg15.9% wr69.4%")
print()
print(">>> = 勝率超越 89.90 且總報酬 ≥ 80%（值得上線的候選）")
print()
print("Done!")
