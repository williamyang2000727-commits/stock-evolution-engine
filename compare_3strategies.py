"""三策略 1500 天公平比較（189 vs 88.60 vs 89.90）
用法：python compare_3strategies.py
"""
import sys, types, json, math

mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

from gpu_cupy_evolve import precompute, cpu_replay, download_data

STRATEGIES = {
    "189": {"w_rsi": 1.0, "rsi_th": 65.0, "w_bb": 0.0, "bb_th": 0.7, "w_vol": 0.0, "vol_th": 3.5, "w_ma": 3.0, "w_macd": 3.0, "macd_mode": 0.0, "w_kd": 2.0, "kd_th": 80.0, "kd_cross": 0.0, "w_wr": 0.0, "wr_th": -30.0, "w_mom": 2.0, "mom_th": 8.0, "w_near_high": 0.0, "near_high_pct": 5.0, "w_squeeze": 0.0, "w_new_high": 2.0, "w_adx": 2.0, "adx_th": 40.0, "consecutive_green": 1.0, "gap_up": 1.0, "above_ma60": 0.0, "vol_gt_yesterday": 0.0, "buy_threshold": 14.0, "stop_loss": -12.0, "use_take_profit": 1.0, "take_profit": 40.0, "trailing_stop": 15.0, "use_rsi_sell": 0.0, "rsi_sell": 75.0, "use_macd_sell": 0.0, "use_kd_sell": 0.0, "sell_vol_shrink": 0.0, "sell_below_ma": 3.0, "hold_days": 30.0, "w_bias": 1.0, "bias_max": 8.0, "use_stagnation_exit": 0.0, "stagnation_days": 15.0, "stagnation_min_ret": 5.0, "use_breakeven": 1.0, "breakeven_trigger": 10.0, "w_obv": 2.0, "obv_rising_days": 10.0, "w_atr": 2.0, "atr_min": 3.0, "use_time_decay": 0.0, "ret_per_day": 1.5, "use_profit_lock": 0.0, "lock_trigger": 40.0, "lock_floor": 8.0, "use_mom_exit": 0.0, "mom_exit_th": 5.0, "upgrade_margin": 0.0, "max_positions": 2.0, "w_sector_flow": 0.0, "sector_flow_topn": 3.0, "w_up_days": 2.0, "up_days_min": 5.0, "w_week52": 2.0, "week52_min": 0.6, "w_vol_up_days": 0.0, "vol_up_days_min": 2.0, "w_mom_accel": 1.0, "mom_accel_min": 8.0, "ma_fast_w": 5, "ma_slow_w": 60, "momentum_days": 3},
    "88.60": {"w_rsi": 3.0, "rsi_th": 65.0, "w_bb": 1.0, "bb_th": 0.9, "w_vol": 0.0, "vol_th": 3.5, "w_ma": 3.0, "w_macd": 3.0, "macd_mode": 0.0, "w_kd": 2.0, "kd_th": 80.0, "kd_cross": 0.0, "w_wr": 3.0, "wr_th": -40.0, "w_mom": 3.0, "mom_th": 8.0, "w_near_high": 2.0, "near_high_pct": 10.0, "w_squeeze": 0.0, "w_new_high": 2.0, "w_adx": 2.0, "adx_th": 40.0, "consecutive_green": 1.0, "gap_up": 1.0, "above_ma60": 1.0, "vol_gt_yesterday": 0.0, "buy_threshold": 6.0, "stop_loss": -20.0, "use_take_profit": 1.0, "take_profit": 40.0, "trailing_stop": 15.0, "use_rsi_sell": 0.0, "rsi_sell": 70.0, "use_macd_sell": 0.0, "use_kd_sell": 0.0, "sell_vol_shrink": 0.0, "sell_below_ma": 0.0, "hold_days": 30.0, "w_bias": 2.0, "bias_max": 3.0, "use_stagnation_exit": 0.0, "stagnation_days": 5.0, "stagnation_min_ret": 1.0, "use_breakeven": 1.0, "breakeven_trigger": 10.0, "w_obv": 1.0, "obv_rising_days": 10.0, "w_atr": 3.0, "atr_min": 3.0, "use_time_decay": 0.0, "ret_per_day": 1.5, "use_profit_lock": 0.0, "lock_trigger": 40.0, "lock_floor": 10.0, "use_mom_exit": 0.0, "mom_exit_th": 3.0, "upgrade_margin": 0.0, "max_positions": 2.0, "w_sector_flow": 0.0, "sector_flow_topn": 1.0, "w_up_days": 2.0, "up_days_min": 5.0, "w_week52": 3.0, "week52_min": 0.7, "w_vol_up_days": 1.0, "vol_up_days_min": 3.0, "w_mom_accel": 1.0, "mom_accel_min": 0.0, "ma_fast_w": 3, "ma_slow_w": 15, "momentum_days": 3},
    "89.90": {"w_rsi": 3.0, "rsi_th": 70.0, "w_bb": 3.0, "bb_th": 0.95, "w_vol": 0.0, "vol_th": 2.5, "w_ma": 2.0, "w_macd": 3.0, "macd_mode": 0.0, "w_kd": 2.0, "kd_th": 80.0, "kd_cross": 0.0, "w_wr": 0.0, "wr_th": -50.0, "w_mom": 3.0, "mom_th": 8.0, "w_near_high": 2.0, "near_high_pct": 10.0, "w_squeeze": 0.0, "w_new_high": 1.0, "w_adx": 2.0, "adx_th": 40.0, "consecutive_green": 1.0, "gap_up": 1.0, "above_ma60": 0.0, "vol_gt_yesterday": 0.0, "buy_threshold": 8.0, "stop_loss": -20.0, "use_take_profit": 1.0, "take_profit": 40.0, "trailing_stop": 20.0, "use_rsi_sell": 0.0, "rsi_sell": 70.0, "use_macd_sell": 0.0, "use_kd_sell": 0.0, "sell_vol_shrink": 0.0, "sell_below_ma": 0.0, "hold_days": 30.0, "w_bias": 1.0, "bias_max": 5.0, "use_stagnation_exit": 0.0, "stagnation_days": 7.0, "stagnation_min_ret": 3.0, "use_breakeven": 1.0, "breakeven_trigger": 10.0, "w_obv": 2.0, "obv_rising_days": 3.0, "w_atr": 1.0, "atr_min": 3.0, "use_time_decay": 0.0, "ret_per_day": 0.8, "use_profit_lock": 1.0, "lock_trigger": 20.0, "lock_floor": 3.0, "use_mom_exit": 0.0, "mom_exit_th": 3.0, "upgrade_margin": 0.0, "max_positions": 2.0, "w_sector_flow": 0.0, "sector_flow_topn": 8.0, "w_up_days": 2.0, "up_days_min": 5.0, "w_week52": 1.0, "week52_min": 0.7, "w_vol_up_days": 1.0, "vol_up_days_min": 2.0, "w_mom_accel": 2.0, "mom_accel_min": 0.0, "ma_fast_w": 3, "ma_slow_w": 15, "momentum_days": 3},
}

# === Load data ===
print("Loading data...")
data = download_data()
_lens = [len(v) for v in data.values()]
_n_1500 = sum(1 for l in _lens if l >= 1500)
if _n_1500 >= 500:
    TARGET_DAYS = 1500
else:
    TARGET_DAYS = 900
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
print(f"Stocks: {len(data)} | Days: {TARGET_DAYS}")

# === Precompute ===
print("Precomputing indicators...")
pre = precompute(data)
print(f"Period: {pre['dates'][0].date()} ~ {pre['dates'][-1].date()}")
print()

# === Run ===
print("=" * 70)
print(f"{'策略':<8} {'筆數':>5} {'總報酬%':>8} {'平均%':>6} {'勝率%':>6} {'MaxDD%':>7} {'avg天':>5} {'持有':>3}")
print("=" * 70)

for name, p in STRATEGIES.items():
    trades = cpu_replay(pre, p)
    trades = [t for t in trades if not math.isnan(t.get("return", 0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    holding = [t for t in trades if t.get("reason") == "持有中"]
    n = len(completed)
    if n == 0:
        print(f"{name:<8} NO TRADES")
        continue
    rets = [t["return"] for t in completed]
    total = sum(rets)
    avg = total / n
    wins = sum(1 for r in rets if r > 0)
    wr = wins / n * 100
    # MaxDD (consecutive loss sum)
    run_dd = 0; max_dd = 0
    for r in rets:
        if r < 0:
            run_dd += r
        else:
            run_dd = 0
        if run_dd < max_dd:
            max_dd = run_dd
    avg_days = sum(t["days"] for t in completed) / n
    print(f"{name:<8} {n:>5} {total:>8.1f} {avg:>6.1f} {wr:>6.1f} {max_dd:>7.1f} {avg_days:>5.1f} {len(holding):>3}")

    # Per-year breakdown
    years = {}
    for t in completed:
        y = t["buy_date"][:4]
        if y not in years:
            years[y] = []
        years[y].append(t)
    print(f"         ", end="")
    for y in sorted(years.keys()):
        ts = years[y]
        yn = len(ts)
        yw = sum(1 for t in ts if t["return"] > 0)
        ya = sum(t["return"] for t in ts) / yn
        print(f" {y}:{yn}筆/{yw/yn*100:.0f}%/avg{ya:.1f}%", end="")
    print()

    # Holdings
    if holding:
        for h in holding:
            print(f"          持有: {h['name']} {h['ticker']} 買{h['buy_date']} @{h['buy_price']}")
    print("-" * 70)

print()
print("Done!")
