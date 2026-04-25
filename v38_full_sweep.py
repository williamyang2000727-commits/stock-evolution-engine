"""
V38 完整 sweep — 放寬 Kronos next-day threshold 找頻率/勝率平衡
用法：C:\\stock-evolution> python v38_full_sweep.py

問題：V38 預設 next > 0.8 太嚴 → 一年 3.5 筆
真正應該調的是 **next-day threshold**，不是 ML threshold

Sweep 兩個維度：
  - Kronos next-day threshold: -2, -1, 0, 0.3, 0.5, 0.8, 1.0
  - 5d 規則：full median / >= 0 / 不限
  - 看每組 trade-off
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base
from metalabel_features import extract_features_for_trades

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
SANITY_CSV = os.path.join(USER_SE, "kronos_sanity_results.csv")


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def stats_of(rets):
    if len(rets) == 0:
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0}
    r = np.array(rets)
    return {
        "n": len(r),
        "wr": float((r > 0).mean() * 100),
        "total": float(r.sum()),
        "avg": float(r.mean()),
    }


def main():
    print("=" * 80)
    print("V38 完整 Sweep — 放寬 Kronos threshold 找頻率/勝率平衡")
    print("=" * 80)

    df_sanity = pd.read_csv(SANITY_CSV)

    print(f"\n跑 89.90 + 抽 features...")
    params = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]

    X19, _, keep_indices = extract_features_for_trades(pre, completed)
    trades_kept = [completed[i] for i in keep_indices]
    df_trades = pd.DataFrame([{
        "ticker": t.get("ticker"),
        "buy_date": t.get("buy_date"),
        "actual_return": float(t.get("return", 0)),
    } for t in trades_kept]).reset_index(drop=True)

    df_merged = df_trades.merge(
        df_sanity[["ticker", "buy_date", "pred_next_pct", "pred_5d_pct"]],
        on=["ticker", "buy_date"], how="inner"
    )
    rets_m = df_merged["actual_return"].values
    p_next = df_merged["pred_next_pct"].values
    p_5d = df_merged["pred_5d_pct"].values
    print(f"最終 {len(df_merged)} 筆")
    print(f"Kronos next 分布：min={p_next.min():.2f}, p25={np.percentile(p_next,25):.2f}, "
          f"median={np.median(p_next):.2f}, p75={np.percentile(p_next,75):.2f}, max={p_next.max():.2f}")
    print(f"Kronos 5d   分布：min={p_5d.min():.2f}, p25={np.percentile(p_5d,25):.2f}, "
          f"median={np.median(p_5d):.2f}, p75={np.percentile(p_5d,75):.2f}, max={p_5d.max():.2f}")

    n_years = 1500 / 250
    a_stats = stats_of(rets_m)

    # === Sweep ===
    print()
    print(f"{'設定':<35} {'n':<5} {'kept%':<7} {'wr':<7} {'avg':<8} {'total':<10} {'年化':<6} {'評價'}")
    print("-" * 105)

    print(f"{'A (89.90 全買)':<35} {a_stats['n']:<5} {100:<7.1f} {a_stats['wr']:<7.1f} "
          f"{a_stats['avg']:<+8.2f} {a_stats['total']:<+10.0f} {a_stats['n']/n_years:<6.1f} 基準")

    print()

    # 各種 next threshold × 5d 規則
    next_ths = [-2.0, -1.0, -0.5, 0.0, 0.3, 0.5, 0.8]
    fived_rules = [
        ("any", lambda p5d: np.ones(len(p5d), dtype=bool)),
        (">= 0", lambda p5d: p5d >= 0),
        (">= -5", lambda p5d: p5d >= -5),
        (">= median", lambda p5d: p5d > np.median(p5d)),
    ]

    results = []
    best_per_freq_band = {"low": None, "mid": None, "high": None}

    for next_th in next_ths:
        for rule_name, rule_fn in fived_rules:
            keep = (p_next > next_th) & rule_fn(p_5d)
            s = stats_of(rets_m[keep])
            if s["n"] < 5:
                continue
            kept_pct = s["n"] / a_stats["n"] * 100
            yearly = s["n"] / n_years

            label = f"next > {next_th:+.1f} | 5d {rule_name}"

            if yearly < 5:
                eval_str = "🔴 太低"
                band = "low"
            elif yearly < 10:
                eval_str = "🟡 偏低"
                band = "low"
            elif yearly < 16:
                eval_str = "🟢 合理"
                band = "mid"
            else:
                eval_str = "🟢 接近 89.90"
                band = "high"

            print(f"{label:<35} {s['n']:<5} {kept_pct:<7.1f} {s['wr']:<7.1f} "
                  f"{s['avg']:<+8.2f} {s['total']:<+10.0f} {yearly:<6.1f} {eval_str}")

            r = {
                "label": label, "next_th": next_th, "rule_5d": rule_name,
                "n": s["n"], "kept_pct": kept_pct, "wr": s["wr"],
                "avg": s["avg"], "total": s["total"], "yearly": yearly,
            }
            results.append(r)

            # 找各頻率帶的最佳 wr
            if band in best_per_freq_band:
                cur = best_per_freq_band[band]
                if cur is None or s["wr"] > cur["wr"]:
                    best_per_freq_band[band] = r
        print()

    # === 推薦 ===
    print("=" * 80)
    print("📊 各頻率帶最佳設定（按 wr 排）")
    print("=" * 80)

    for band, label in [("high", "🟢 高頻率（接近 89.90，每月 1.5+ 筆）"),
                         ("mid", "🟢 中頻率（每月 ~1 筆）"),
                         ("low", "🟡 低頻率（每 2-3 月 1 筆）")]:
        best = best_per_freq_band[band]
        if best is None:
            continue
        print(f"\n{label}")
        print(f"  設定：{best['label']}")
        print(f"  n={best['n']}, kept {best['kept_pct']:.1f}%, "
              f"wr {best['wr']:.1f}%, avg {best['avg']:+.2f}%, "
              f"total {best['total']:+.0f}%, 年化 {best['yearly']:.1f} 筆")
        print(f"  vs 89.90：wr {best['wr']-a_stats['wr']:+.1f}%, "
              f"total {best['total']-a_stats['total']:+.0f}%")

    out = os.path.join(USER_SE, "v38_full_sweep.json")
    with open(out, "w") as f:
        json.dump({"all": results, "best_per_band": best_per_freq_band, "baseline": a_stats},
                  f, indent=2, default=str)
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
