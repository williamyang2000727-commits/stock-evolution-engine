"""公平比較兩個策略（用 Web 完全一樣的指標計算方式）
指標：總報酬、CAGR、Max Drawdown（equity curve）、Sharpe、勝率、盈虧比、Profit Factor
用法：python compare_fair.py
"""
import sys, types, json, math
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data
from datetime import date as _date

# === 兩個策略 ===
STRATEGY_89 = {"name": "89.90 (2pos)", "params": {"w_rsi": 3.0, "rsi_th": 70.0, "w_bb": 3.0, "bb_th": 0.95, "w_vol": 0.0, "vol_th": 2.5, "w_ma": 2.0, "w_macd": 3.0, "macd_mode": 0.0, "w_kd": 2.0, "kd_th": 80.0, "kd_cross": 0.0, "w_wr": 0.0, "wr_th": -50.0, "w_mom": 3.0, "mom_th": 8.0, "w_near_high": 2.0, "near_high_pct": 10.0, "w_squeeze": 0.0, "w_new_high": 1.0, "w_adx": 2.0, "adx_th": 40.0, "consecutive_green": 1.0, "gap_up": 1.0, "above_ma60": 0.0, "vol_gt_yesterday": 0.0, "buy_threshold": 8.0, "stop_loss": -20.0, "use_take_profit": 1.0, "take_profit": 40.0, "trailing_stop": 20.0, "use_rsi_sell": 0.0, "rsi_sell": 70.0, "use_macd_sell": 0.0, "use_kd_sell": 0.0, "sell_vol_shrink": 0.0, "sell_below_ma": 0.0, "hold_days": 30.0, "w_bias": 1.0, "bias_max": 5.0, "use_stagnation_exit": 0.0, "stagnation_days": 7.0, "stagnation_min_ret": 3.0, "use_breakeven": 1.0, "breakeven_trigger": 10.0, "w_obv": 2.0, "obv_rising_days": 3.0, "w_atr": 1.0, "atr_min": 3.0, "use_time_decay": 0.0, "ret_per_day": 0.8, "use_profit_lock": 1.0, "lock_trigger": 20.0, "lock_floor": 3.0, "use_mom_exit": 0.0, "mom_exit_th": 3.0, "upgrade_margin": 0.0, "max_positions": 2.0, "w_sector_flow": 0.0, "sector_flow_topn": 8.0, "w_up_days": 2.0, "up_days_min": 5.0, "w_week52": 1.0, "week52_min": 0.7, "w_vol_up_days": 1.0, "vol_up_days_min": 2.0, "w_mom_accel": 2.0, "mom_accel_min": 0.0, "ma_fast_w": 3, "ma_slow_w": 15, "momentum_days": 3}}

# 從 pending 讀新策略
import os
HERE = os.path.dirname(os.path.abspath(__file__))
_pending_path = None
for f in ["pending_push.json", "pending_push.json.pushed"]:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        _pending_path = p; break
if _pending_path:
    _pd = json.load(open(_pending_path, encoding="utf-8"))
    STRATEGY_NEW = {"name": f"NEW {_pd.get('score',0):.1f} ({int(_pd['params'].get('max_positions',2))}pos)", "params": _pd["params"]}
else:
    print("No pending_push.json found"); sys.exit(1)

STRATEGIES = [STRATEGY_89, STRATEGY_NEW]

# === Load data ===
print("Loading data...")
data = download_data()
_lens = [len(v) for v in data.values()]
_n_1500 = sum(1 for l in _lens if l >= 1500)
TARGET_DAYS = 1500 if _n_1500 >= 500 else 900
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
print(f"Stocks: {len(data)} | Days: {TARGET_DAYS}")
print("Precomputing...")
pre = precompute(data)
first_date = pre["dates"][0].date()
last_date = pre["dates"][-1].date()
print(f"Period: {first_date} ~ {last_date}")
print()

# === Run + Web-style metrics ===
print("=" * 80)
print(f"{'':>22} | {'89.90 (2pos)':>14} | {'NEW (1pos)':>14} | {'diff':>8}")
print("=" * 80)

results = []
for strat in STRATEGIES:
    p = strat["params"]
    trades = cpu_replay(pre, p)
    trades = [t for t in trades if not math.isnan(t.get("return", 0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    holding = [t for t in trades if t.get("reason") == "持有中"]
    n = len(completed)
    if n == 0:
        results.append(None); continue

    rets = [t["return"] for t in completed]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    bt_total = sum(rets)
    win_rate = len(wins) / n * 100
    avg_ret = bt_total / n
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    avg_hold = sum(t.get("days", 0) for t in completed) / n
    max_positions = int(p.get("max_positions", 2))

    # === CAGR (Web formula) ===
    pos_size = 1.0 / max(max_positions, 1)
    portfolio_growth = 1 + (bt_total * pos_size) / 100
    all_buy_dates = sorted([t.get("buy_date", "") for t in trades if t.get("buy_date")])
    first_trade = _date.fromisoformat(all_buy_dates[0]) if all_buy_dates else first_date
    years = max((last_date - first_trade).days / 365.25, 1.0)
    cagr = (portfolio_growth ** (1 / years) - 1) * 100 if portfolio_growth > 0 else 0

    # === Max Drawdown (Web formula: equity curve × pos_size) ===
    equity = 1.0
    peak_eq = 1.0
    max_dd = 0
    for r in rets:
        equity *= (1 + r * pos_size / 100)
        if equity <= 0: equity = 0.0001
        peak_eq = max(peak_eq, equity)
        dd = (equity / peak_eq - 1) * 100
        max_dd = min(max_dd, dd)

    # === Sharpe (Web formula) ===
    if n >= 2:
        mean_r = sum(rets) / n
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / (n - 1))
        if avg_hold > 1:
            trades_per_year = 252 / avg_hold
        else:
            trades_per_year = n / years
        sharpe = (mean_r / std_r) * math.sqrt(trades_per_year) if std_r > 0 else 0
    else:
        sharpe = 0

    # === Profit Factor + Win/Loss Ratio ===
    total_win = sum(wins)
    total_loss = abs(sum(losses)) if losses else 0
    pf = total_win / total_loss if total_loss > 0 else float('inf')
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    results.append({
        "name": strat["name"], "n": n, "total": bt_total, "avg": avg_ret,
        "wr": win_rate, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
        "pf": pf, "wl": wl_ratio, "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_hold": avg_hold, "holding": len(holding), "max_pos": max_positions,
        "years": years, "portfolio_growth": portfolio_growth,
    })

# === Print comparison ===
r0, r1 = results[0], results[1]

def row(label, v0, v1, fmt=".1f", higher_better=True):
    s0 = f"{v0:{fmt}}" if isinstance(v0, float) else str(v0)
    s1 = f"{v1:{fmt}}" if isinstance(v1, float) else str(v1)
    diff = v1 - v0
    if abs(diff) < 0.05:
        marker = "  ="
    elif (diff > 0) == higher_better:
        marker = " <<<"
    else:
        marker = ""
    d_str = f"{diff:+{fmt}}" if isinstance(diff, float) else f"{diff:+d}"
    print(f"  {label:>20} | {s0:>14} | {s1:>14} | {d_str:>8}{marker}")

row("trades", r0["n"], r1["n"], "d", False)
row("sum total %", r0["total"], r1["total"])
row("avg / trade %", r0["avg"], r1["avg"])
row("win rate %", r0["wr"], r1["wr"])
row("CAGR %", r0["cagr"], r1["cagr"])
row("Max Drawdown %", r0["max_dd"], r1["max_dd"], ".1f", False)
row("Sharpe", r0["sharpe"], r1["sharpe"], ".2f")
row("Profit Factor", r0["pf"], r1["pf"], ".2f")
row("W/L Ratio", r0["wl"], r1["wl"], ".2f")
row("avg win %", r0["avg_win"], r1["avg_win"])
row("avg loss %", r0["avg_loss"], r1["avg_loss"], ".1f", False)
row("avg hold days", r0["avg_hold"], r1["avg_hold"], ".0f", False)
row("max_positions", r0["max_pos"], r1["max_pos"], "d", False)

print("=" * 80)
print()
print(f"  Portfolio growth:  89.90={r0['portfolio_growth']:.2f}x  NEW={r1['portfolio_growth']:.2f}x  (pos-adjusted, {r0['years']:.1f} years)")
print()

# Per-year comparison
print("=" * 80)
print("  分年勝率比較")
print("=" * 80)
for strat, res in zip(STRATEGIES, results):
    trades = cpu_replay(pre, strat["params"])
    trades = [t for t in trades if not math.isnan(t.get("return", 0)) and t.get("reason") != "持有中"]
    years_d = {}
    for t in trades:
        y = t["buy_date"][:4]
        if y not in years_d: years_d[y] = []
        years_d[y].append(t)
    print(f"  {res['name']}:")
    for y in sorted(years_d.keys()):
        ts = years_d[y]
        yn = len(ts)
        yw = sum(1 for t in ts if t["return"] > 0)
        ya = sum(t["return"] for t in ts) / yn
        print(f"    {y}: {yn:>3}p  wr {yw/yn*100:>5.0f}%  avg {ya:>+6.1f}%")
    print()

print("Done!")
