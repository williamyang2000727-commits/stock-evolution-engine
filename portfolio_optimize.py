"""
V43 Portfolio of Strategies — Efficient Frontier
用法：C:\\stock-evolution> python portfolio_optimize.py

V1-V42 全敗（33 種「找新策略」方向）→ 89.90 是 GLOBAL OPTIMUM
**剩唯一沒試的方向**：不找新策略，找最佳資金配置

設計：
  把 89.90 / 103 / 88.60 當 3 個 building blocks
  跑 1000 個 weight combinations [w1, w2, w3] s.t. sum=1
  對每個 weight 算 portfolio (total, wr, MaxDD, Sharpe)
  畫 efficient frontier（最大 Sharpe / 最小 DD / 最大 total）

學術依據：
  Markowitz (1952) Modern Portfolio Theory
  Sharpe (1966) Sharpe Ratio
  Roncalli (2014) Risk Parity portfolios

預期：
  發現「89.90 純跑 100% 風險最高」
  Mix 89.90 60% + 103 40% 可能 Sharpe 更高 / DD 更小
  → 換得失：total 略降 但 risk-adjusted 更好

策略來源：
  89.90: GPU Gist c1bef892（生產線上）
  103:   strategy_103_1pos_backup.json（Windows 本地）
  88.60: 從 Mac 備份還原 OR 89.90 的近親
"""
import os, sys, json, pickle, time, random
import numpy as np

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
STRATEGY_103_PATH = os.path.join(USER_SE, "strategy_103_1pos_backup.json")


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def load_strategy_103():
    if not os.path.exists(STRATEGY_103_PATH):
        print(f"  ⚠️ {STRATEGY_103_PATH} 不存在，跳過 103")
        return None
    with open(STRATEGY_103_PATH) as f:
        d = json.load(f)
    return d.get("params", d)


def normalize_params(p):
    """補足 PARAMS_SPACE 中 strategy 缺漏的 keys（同 nsga2 做法）"""
    p = dict(p)
    for k, opts in base.PARAMS_SPACE.items():
        if k not in p:
            p[k] = opts[0]
    if "ma_fast_w" not in p: p["ma_fast_w"] = 5
    if "ma_slow_w" not in p: p["ma_slow_w"] = 20
    if "momentum_days" not in p: p["momentum_days"] = 5
    return p


def get_trades_for_strategy(name, params, pre):
    """跑 cpu_replay 拿 trades，轉成「日期 → return」timeline"""
    print(f"  跑 {name} cpu_replay...")
    params = normalize_params(params)
    trades = base.cpu_replay(pre, params)
    completed = [t for t in trades if t.get("sell_date") and t.get("reason") != "持有中"]
    n = len(completed)
    if n == 0:
        return None
    rets = np.array([t.get("return", 0) for t in completed])
    total = rets.sum()
    wr = (rets > 0).mean() * 100
    avg = rets.mean()
    print(f"    {name}: n={n}, total {total:+.0f}%, wr {wr:.1f}%, avg {avg:+.2f}%")
    return {
        "name": name,
        "params": params,
        "trades": completed,
        "n": n, "total": float(total), "wr": float(wr), "avg": float(avg),
    }


def trades_to_daily_pnl(trades, dates_arr, max_pos):
    """
    把 trades list 轉成 daily P&L array（per dollar of capital, divided by max_pos）
    daily[date_idx] = sum of (return / max_pos / hold_days) over trades held that day
    這個近似讓「不同策略」可以在 portfolio 層級疊加
    """
    n_days = len(dates_arr)
    daily = np.zeros(n_days, dtype=np.float64)
    date_to_idx = {str(d.date() if hasattr(d, 'date') else d)[:10]: i
                   for i, d in enumerate(dates_arr)}

    for t in trades:
        bd = t.get("buy_date", "")
        sd = t.get("sell_date", "")
        ret = t.get("return", 0)
        if bd not in date_to_idx or sd not in date_to_idx:
            continue
        bi = date_to_idx[bd]
        si = date_to_idx[sd]
        hold = max(1, si - bi)
        # 平均分配到持倉日
        per_day_pnl = (ret / max_pos) / hold
        daily[bi:si + 1] += per_day_pnl

    return daily


def portfolio_stats(daily_pnl_combined):
    """從 combined daily P&L 算 portfolio stats"""
    if len(daily_pnl_combined) == 0:
        return None
    cum = np.cumsum(daily_pnl_combined)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min())
    total = float(cum[-1])
    n_active_days = (daily_pnl_combined != 0).sum()
    if daily_pnl_combined.std() > 0:
        # daily Sharpe → annualize sqrt(252)
        sharpe = float(daily_pnl_combined.mean() / daily_pnl_combined.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    return {
        "total": total, "max_dd": max_dd,
        "sharpe": sharpe, "n_active_days": int(n_active_days),
    }


def main():
    print("=" * 80)
    print("V43 Portfolio of Strategies — Efficient Frontier")
    print("=" * 80)

    # === 1. 載 cache ===
    print(f"\n[1/4] 載 cache + precompute...")
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    dates = pre["dates"]

    # === 2. 跑 89.90 + 103（其他策略可後加）===
    print(f"\n[2/4] 跑各策略 cpu_replay...")
    strategies = []

    s1 = get_trades_for_strategy("89.90", fetch_gist_strategy(), pre)
    if s1:
        strategies.append(s1)

    s103 = load_strategy_103()
    if s103:
        s2 = get_trades_for_strategy("103_1pos", s103, pre)
        if s2:
            strategies.append(s2)

    if len(strategies) < 2:
        print(f"\n❌ 只有 {len(strategies)} 個策略可用，需要至少 2 個做 portfolio")
        print(f"   89.90: GPU Gist (always available)")
        print(f"   103:   {STRATEGY_103_PATH} (需要在 Windows)")
        return

    # === 3. 算 daily P&L per strategy ===
    print(f"\n[3/4] 轉 daily P&L timeline...")
    daily_pnls = {}
    for s in strategies:
        max_pos = int(s["params"].get("max_positions", 2))
        daily = trades_to_daily_pnl(s["trades"], dates, max_pos)
        daily_pnls[s["name"]] = daily
        active = (daily != 0).sum()
        print(f"  {s['name']}: max_pos={max_pos}, active days {active}/{len(daily)}, "
              f"daily mean {daily.mean()*100:+.3f}%, std {daily.std()*100:.3f}%")

    # === 4. Monte Carlo weight sweep ===
    n_strat = len(strategies)
    n_samples = 1000
    print(f"\n[4/4] Monte Carlo {n_samples} weights × {n_strat} strategies...")

    results = []
    daily_arrs = [daily_pnls[s["name"]] for s in strategies]
    daily_matrix = np.stack(daily_arrs)  # (n_strat, n_days)

    # 個別 baseline (w=[1,0,0...])
    baselines = []
    for i, s in enumerate(strategies):
        w = np.zeros(n_strat); w[i] = 1.0
        port = (w[:, None] * daily_matrix).sum(axis=0)
        st = portfolio_stats(port)
        baselines.append({"name": s["name"], "weights": w.tolist(), **st})
        print(f"  baseline {s['name']}: total {st['total']:+.0f}%, "
              f"DD {st['max_dd']:.1f}%, Sharpe {st['sharpe']:.3f}")

    # Random weights (Dirichlet for uniform on simplex)
    print(f"\n  Sweep random weights...")
    for _ in range(n_samples):
        w = np.random.dirichlet(np.ones(n_strat))
        port = (w[:, None] * daily_matrix).sum(axis=0)
        st = portfolio_stats(port)
        results.append({"weights": w.tolist(), **st})

    # === 5. Efficient frontier ===
    print()
    print("=" * 80)
    print(f"📊 Efficient Frontier ({n_strat} strategies × {n_samples} weights)")
    print("=" * 80)

    # Best by 3 criteria
    best_total = max(results, key=lambda r: r["total"])
    best_dd = max(results, key=lambda r: r["max_dd"])  # max_dd is negative, max(closer to 0)
    best_sharpe = max(results, key=lambda r: r["sharpe"])

    def fmt_w(w):
        return "[" + ", ".join(f"{x:.2f}" for x in w) + "]"

    print(f"\n  Strategies: {[s['name'] for s in strategies]}")
    print()
    print(f"  {'Type':<28} {'weights':<28} {'total':<10} {'MaxDD':<10} {'Sharpe':<8}")
    print(f"  {'-' * 95}")
    for b in baselines:
        print(f"  baseline {b['name']:<20} {fmt_w(b['weights']):<28} "
              f"{b['total']:<+10.0f} {b['max_dd']:<10.1f} {b['sharpe']:<8.3f}")
    print(f"  {'-' * 95}")
    print(f"  best total                   {fmt_w(best_total['weights']):<28} "
          f"{best_total['total']:<+10.0f} {best_total['max_dd']:<10.1f} {best_total['sharpe']:<8.3f}")
    print(f"  best MaxDD (closest to 0)    {fmt_w(best_dd['weights']):<28} "
          f"{best_dd['total']:<+10.0f} {best_dd['max_dd']:<10.1f} {best_dd['sharpe']:<8.3f}")
    print(f"  best Sharpe                  {fmt_w(best_sharpe['weights']):<28} "
          f"{best_sharpe['total']:<+10.0f} {best_sharpe['max_dd']:<10.1f} {best_sharpe['sharpe']:<8.3f}")

    # 看 risk-parity（DD 大幅降但 total 接受少量損失）的點
    s1_baseline = baselines[0]  # 89.90 baseline
    print(f"\n  vs 89.90 baseline (total {s1_baseline['total']:+.0f}%, DD {s1_baseline['max_dd']:.1f}%, Sharpe {s1_baseline['sharpe']:.3f})")
    print()

    # 找「DD 改善 ≥ 30% AND total 損失 ≤ 20%」的 risk-adjusted alpha
    risk_alphas = []
    for r in results:
        dd_imp = r["max_dd"] - s1_baseline["max_dd"]  # 越正越好（max_dd 是負數，往 0 移動）
        tot_loss_pct = (r["total"] - s1_baseline["total"]) / abs(s1_baseline["total"]) * 100
        if dd_imp >= 5 and tot_loss_pct >= -20:  # DD 改善 ≥ 5%, total 損失 ≤ 20%
            risk_alphas.append({**r, "dd_imp": dd_imp, "tot_loss_pct": tot_loss_pct})

    if risk_alphas:
        risk_alphas.sort(key=lambda r: -r["sharpe"])
        print(f"  🟢 Risk-adjusted alpha 配置（DD↑≥5%, total 損失 ≤20%, sort by Sharpe）：")
        print(f"  {'weights':<28} {'total':<10} {'MaxDD':<10} {'Sharpe':<8} {'DD↑':<8} {'tot 損':<8}")
        for r in risk_alphas[:10]:
            print(f"  {fmt_w(r['weights']):<28} {r['total']:<+10.0f} {r['max_dd']:<10.1f} "
                  f"{r['sharpe']:<8.3f} {r['dd_imp']:<+8.1f} {r['tot_loss_pct']:<+8.1f}")

        best = risk_alphas[0]
        print(f"\n  🏆 推薦上線配置：weights={fmt_w(best['weights'])}")
        print(f"     total {best['total']:+.0f}% (vs 89.90 {s1_baseline['total']:+.0f}%, 損失 {best['tot_loss_pct']:+.1f}%)")
        print(f"     MaxDD {best['max_dd']:.1f}% (vs 89.90 {s1_baseline['max_dd']:.1f}%, 改善 {best['dd_imp']:+.1f}%)")
        print(f"     Sharpe {best['sharpe']:.3f} (vs 89.90 {s1_baseline['sharpe']:.3f}, 改善 {best['sharpe'] - s1_baseline['sharpe']:+.3f})")
    else:
        print(f"  🔴 沒有「DD↑≥5% AND total 損失 ≤20%」配置")
        print(f"     → 89.90 純跑就是 portfolio 層級的 optimal")
        print(f"     → 接受 89.90 final，5090 不需要做 portfolio")

    # === 存 ===
    out = os.path.join(USER_SE, "portfolio_results.json")
    with open(out, "w") as f:
        json.dump({
            "strategies": [s["name"] for s in strategies],
            "baselines": baselines,
            "best_total": best_total,
            "best_dd": best_dd,
            "best_sharpe": best_sharpe,
            "risk_alphas": risk_alphas[:20] if risk_alphas else [],
            "n_samples": n_samples,
        }, f, indent=2)
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
