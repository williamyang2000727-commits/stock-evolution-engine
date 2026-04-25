"""
基本面 backfill — 看月營收 YoY filter 對 89.90 total return 的影響
用法：C:\\stock-evolution> python backfill_revenue.py

對比 Kronos backfill：
  Kronos sanity Spearman 看似贏，backfill 才知道 total -87%
  → revenue 也要做這步驗證

判定：
  - 不只看 wr / avg，要看 total return
  - 如果 total 跟 89.90 接近 OR 更高 → 真有 net alpha
  - 如果 total 砍很多 → 跟 Kronos 同下場（被自己騙）
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

YOY_CSV = os.path.join(USER_SE, "revenue_yoy_results.csv")


def stats_of(rets):
    if len(rets) == 0:
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    r = np.array(rets)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return {
        "n": len(r),
        "wr": float((r > 0).mean() * 100),
        "total": float(r.sum()),
        "avg": float(r.mean()),
        "max_dd": float(dd.min()) if len(dd) > 0 else 0.0,
        "sharpe": float(r.mean() / r.std()) if r.std() > 0 else 0.0,
    }


def main():
    print("=" * 80)
    print("基本面 Backfill — 月營收 YoY filter 對 total return 的影響")
    print("=" * 80)

    if not os.path.exists(YOY_CSV):
        print(f"❌ {YOY_CSV} 不存在，先跑 sanity_test_revenue.py")
        return

    df = pd.read_csv(YOY_CSV)
    print(f"\n讀 {len(df)} 筆 trade YoY 資料")

    yoy = df["yoy_pct"].values
    rets = df["actual_return"].values
    n_years = 6.0

    # === Sweep ===
    print()
    print(f"{'設定':<20} {'n':<5} {'kept%':<7} {'wr':<7} {'avg':<8} {'total':<10} {'sharpe':<8} {'年化':<6} {'評價'}")
    print("-" * 95)

    a_stats = stats_of(rets)
    print(f"{'A (89.90 全買)':<20} {a_stats['n']:<5} {100:<7.1f} {a_stats['wr']:<7.1f} "
          f"{a_stats['avg']:<+8.2f} {a_stats['total']:<+10.0f} {a_stats['sharpe']:<8.3f} "
          f"{a_stats['n']/n_years:<6.1f} 基準")
    print()

    sweep = [-20, -10, 0, 10, 20, 30, 50]
    results = []
    for th in sweep:
        keep = yoy > th
        if keep.sum() < 10:
            continue
        s = stats_of(rets[keep])
        kept_pct = s["n"] / a_stats["n"] * 100
        yearly = s["n"] / n_years

        # 評價：跟 89.90 比 wr 跟 total
        wr_vs = s["wr"] - a_stats["wr"]
        total_vs_pct = (s["total"] - a_stats["total"]) / abs(a_stats["total"]) * 100
        sharpe_vs = s["sharpe"] - a_stats["sharpe"]

        if wr_vs > 3 and total_vs_pct > -10:
            eval_str = "🟢🟢 wr 提升 + total 不損失"
        elif wr_vs > 3 and total_vs_pct > -30:
            eval_str = "🟢 wr 提升，total 略降"
        elif wr_vs > 0 and total_vs_pct > -50:
            eval_str = "🟡 邊際"
        else:
            eval_str = "🔴 net 負"

        label = f"YoY > {th:+>3}%"
        print(f"{label:<20} {s['n']:<5} {kept_pct:<7.1f} {s['wr']:<7.1f} "
              f"{s['avg']:<+8.2f} {s['total']:<+10.0f} {s['sharpe']:<8.3f} "
              f"{yearly:<6.1f} {eval_str}")

        results.append({
            "th": th, **s, "kept_pct": kept_pct, "yearly": yearly,
            "wr_vs": wr_vs, "total_vs_pct": total_vs_pct, "sharpe_vs": sharpe_vs,
        })

    # === 推薦 ===
    print()
    print("=" * 80)
    print("📊 推薦設定（vs 89.90）")
    print("=" * 80)

    # 找 wr 提升 >= 3% AND total 不損失 >= -10% 的
    good = [r for r in results if r["wr_vs"] > 3 and r["total_vs_pct"] > -10]
    if good:
        # Best by sharpe
        best_sharpe = max(good, key=lambda r: r["sharpe"])
        print(f"\n🟢🟢 真 net alpha 配置（wr ↑ AND total 不損失）：")
        print(f"   最佳 Sharpe: YoY > {best_sharpe['th']:+}%")
        print(f"   n={best_sharpe['n']}, kept {best_sharpe['kept_pct']:.1f}%, "
              f"wr {best_sharpe['wr']:.1f}% ({best_sharpe['wr_vs']:+.1f}%)")
        print(f"   avg {best_sharpe['avg']:+.2f}%, total {best_sharpe['total']:+.0f}% ({best_sharpe['total_vs_pct']:+.1f}%)")
        print(f"   Sharpe {best_sharpe['sharpe']:.3f} ({best_sharpe['sharpe_vs']:+.3f})")
        print(f"   年化 {best_sharpe['yearly']:.1f} 筆")
        print()
        print(f"   ✅ **這配置 wr 提升又不犧牲 total return → 真有 alpha**")
        print(f"   下一步：CPCV 驗證跨 regime 泛化（避免 V36 80/20 假象）")
    else:
        print(f"\n🔴 所有 threshold 都讓 total 損失 > 10%")
        print(f"   跟 Kronos 同下場 → 接受 89.90 final")

    out = os.path.join(USER_SE, "revenue_backfill_results.json")
    with open(out, "w") as f:
        json.dump({"baseline": a_stats, "sweep": results}, f, indent=2)
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
