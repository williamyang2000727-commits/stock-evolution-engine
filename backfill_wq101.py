"""
WQ101 Backfill — top alphas filter 對 89.90 total return 的影響
用法：C:\\stock-evolution> python backfill_wq101.py

對比 V38 Kronos / V39 月營收教訓：
  Kronos sanity Spearman 看似贏，backfill 才知道 total -87%
  V39 月營收 CPCV real 但 backfill total -34%
  → WQ101 也要做這步驗證

WQ101 CPCV 結果（已知）：
  a003 top10%: 15/15 path mean +30.79% p25 +24.40% (但 totΔ -472%)
  a026 top30%: 13/15 path mean +15.51% (totΔ -334%)
  top5 combo median: 15/15 path mean +10.89% (totΔ -162%)

判定：
  🟢🟢 wr↑ AND total 不損失 (>-10%) → 真 net alpha，值得上線/上 GPU
  🟢 wr↑ AND total 略降 (-10~-30%) → 換 Sharpe 邊際值得
  🟡 wr↑ AND total 大降 (-30~-50%) → 同 V39（不夠值得）
  🔴 跟 V38 Kronos 同下場（total -50%+）

如果都 🔴 → 5090 算力轉 Kronos fine-tune
如果有 🟢🟢 → 寫 GPU patch 加 alpha 進 PARAMS_SPACE 跑 5090
"""
import os, sys, json
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

ALPHAS_CSV = os.path.join(USER_SE, "wq101_alphas.csv")


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


def eval_filter(filt_rets, baseline):
    s = stats_of(filt_rets)
    kept_pct = s["n"] / baseline["n"] * 100 if baseline["n"] > 0 else 0
    n_years = 6.0
    yearly = s["n"] / n_years
    wr_vs = s["wr"] - baseline["wr"]
    total_vs_pct = (s["total"] - baseline["total"]) / abs(baseline["total"]) * 100 if baseline["total"] != 0 else 0
    sharpe_vs = s["sharpe"] - baseline["sharpe"]

    if wr_vs > 3 and total_vs_pct > -10:
        eval_str = "🟢🟢 wr↑ + total 不損失"
    elif wr_vs > 3 and total_vs_pct > -30:
        eval_str = "🟢 wr↑ total 略降"
    elif wr_vs > 0 and total_vs_pct > -50:
        eval_str = "🟡 邊際 (V39 同級)"
    else:
        eval_str = "🔴 net 負 (V38 Kronos 同下場)"

    return {
        **s, "kept_pct": kept_pct, "yearly": yearly,
        "wr_vs": wr_vs, "total_vs_pct": total_vs_pct, "sharpe_vs": sharpe_vs,
        "eval": eval_str,
    }


def print_row(label, r):
    print(f"  {label:<28} {r['n']:<5} {r['kept_pct']:<7.1f} {r['wr']:<7.1f} "
          f"{r['avg']:<+8.2f} {r['total']:<+9.0f} {r['sharpe']:<8.3f} "
          f"{r['yearly']:<6.1f} {r['wr_vs']:<+7.1f} {r['total_vs_pct']:<+8.1f} {r['eval']}")


def main():
    print("=" * 110)
    print("WQ101 Backfill — top alphas filter 對 89.90 total return 的影響")
    print("=" * 110)

    if not os.path.exists(ALPHAS_CSV):
        print(f"❌ {ALPHAS_CSV} 不存在")
        return

    df = pd.read_csv(ALPHAS_CSV)
    print(f"\n讀 {len(df)} 筆 trade × alpha")

    rets = df["actual_return"].values
    a_stats = stats_of(rets)
    print(f"\n89.90 baseline (此 116 筆)：")
    print(f"  n={a_stats['n']}, wr={a_stats['wr']:.1f}%, avg={a_stats['avg']:+.2f}%, "
          f"total={a_stats['total']:+.0f}%, Sharpe={a_stats['sharpe']:.3f}")

    # 算每個 alpha 的 Spearman 找方向
    from scipy import stats as scipy_stats
    import re
    alpha_pattern = re.compile(r"^a\d{3}$")
    all_alphas = [c for c in df.columns if alpha_pattern.match(c)]
    alpha_sps = []
    for name in all_alphas:
        v = df[name].values
        m = np.isfinite(v) & np.isfinite(rets)
        if m.sum() < 30: continue
        sp, _ = scipy_stats.spearmanr(v[m], rets[m])
        if not np.isfinite(sp): continue
        alpha_sps.append((name, float(sp)))
    alpha_sps.sort(key=lambda x: -abs(x[1]))
    top_alphas = [a for a in alpha_sps if abs(a[1]) >= 0.10]

    print(f"\n  Top {len(top_alphas)} alphas (|Sp|≥0.10): {[a[0] for a in top_alphas[:14]]}")

    print()
    print(f"  {'設定':<28} {'n':<5} {'kept%':<7} {'wr':<7} {'avg':<8} "
          f"{'total':<9} {'sharpe':<8} {'年化':<6} {'Δwr':<7} {'Δtot%':<8} {'評價'}")
    print(f"  {'-' * 110}")
    print_row("A (89.90 全買 baseline)", {**a_stats, "kept_pct": 100, "yearly": a_stats["n"]/6.0,
                                          "wr_vs": 0, "total_vs_pct": 0, "sharpe_vs": 0, "eval": "基準"})
    print()

    results = []

    # === 對 top 14 alphas 各自 sweep median / top30 / top20 / top10 ===
    for name, sp in top_alphas[:14]:
        sign = 1 if sp > 0 else -1
        vals = df[name].values
        valid = np.isfinite(vals) & np.isfinite(rets)
        if valid.sum() < 30: continue
        v_valid = vals[valid]
        r_valid = rets[valid]

        v_baseline = stats_of(r_valid)  # 此 alpha 有效範圍下的 baseline
        v_baseline["kept_pct"] = v_baseline["n"] / a_stats["n"] * 100

        # median split
        if sign > 0:
            keep_med = v_valid > np.median(v_valid)
            keep_t30 = v_valid > np.percentile(v_valid, 70)
            keep_t20 = v_valid > np.percentile(v_valid, 80)
            keep_t10 = v_valid > np.percentile(v_valid, 90)
        else:
            keep_med = v_valid < np.median(v_valid)
            keep_t30 = v_valid < np.percentile(v_valid, 30)
            keep_t20 = v_valid < np.percentile(v_valid, 20)
            keep_t10 = v_valid < np.percentile(v_valid, 10)

        for label_suffix, keep in [("median", keep_med), ("top30%", keep_t30),
                                     ("top20%", keep_t20), ("top10%", keep_t10)]:
            if keep.sum() < 5: continue
            r = eval_filter(r_valid[keep], a_stats)
            label = f"{name}({sp:+.3f}) {label_suffix}"
            print_row(label, r)
            results.append({"alpha": name, "sp": sp, "mode": label_suffix, **r})
        print()

    # === Top 5 combo (signed z-score sum) sweep ===
    print(f"  {'Top 5 combo':<28} (signed z-score sum)")
    print()

    if len(top_alphas) >= 5:
        top5 = top_alphas[:5]
        combo = np.zeros(len(df))
        valid_count = np.zeros(len(df))
        for name, sp in top5:
            v = df[name].values
            m = np.isfinite(v)
            if m.sum() < 10: continue
            mu = np.nanmean(v)
            sd = np.nanstd(v)
            if sd == 0: continue
            z = np.where(m, (v - mu) / sd, 0)
            sign = -1 if sp < 0 else 1
            combo += sign * z
            valid_count += m.astype(float)
        combo = np.where(valid_count > 0, combo / np.maximum(valid_count, 1), np.nan)
        valid = np.isfinite(combo) & np.isfinite(rets)
        c_valid = combo[valid]
        r_valid = rets[valid]

        for label_suffix, pct_keep in [("median", 50), ("top30%", 30), ("top20%", 20), ("top10%", 10)]:
            cutoff = np.percentile(c_valid, 100 - pct_keep)
            keep = c_valid > cutoff
            if keep.sum() < 5: continue
            r = eval_filter(r_valid[keep], a_stats)
            label = f"top5 combo {label_suffix}"
            print_row(label, r)
            results.append({"alpha": "top5_combo", "sp": None, "mode": label_suffix, **r})

    # === 推薦 ===
    print()
    print("=" * 110)
    print("📊 推薦設定（看「Δtot% 不損失太多」優先）")
    print("=" * 110)

    # 找 net 真 alpha
    real_alpha = [r for r in results if r["wr_vs"] > 3 and r["total_vs_pct"] > -10]
    margin_alpha = [r for r in results if r["wr_vs"] > 3 and r["total_vs_pct"] > -30 and r["total_vs_pct"] <= -10]
    weak_alpha = [r for r in results if r["wr_vs"] > 0 and r["total_vs_pct"] > -50 and r["total_vs_pct"] <= -30]

    print(f"\n  🟢🟢 真 net alpha (wr↑ AND total 不損失 ≥-10%): {len(real_alpha)} 個")
    if real_alpha:
        best = max(real_alpha, key=lambda r: r["sharpe"])
        print(f"     最佳 Sharpe: {best['alpha']} {best['mode']}")
        print(f"     n={best['n']}, kept {best['kept_pct']:.1f}%, wr {best['wr']:.1f}% ({best['wr_vs']:+.1f}%)")
        print(f"     total {best['total']:+.0f}% ({best['total_vs_pct']:+.1f}%), Sharpe {best['sharpe']:.3f}")
        print(f"     年化 {best['yearly']:.1f} 筆")
        print()
        print(f"     ✅ 真 alpha → 寫 GPU patch 加進 PARAMS_SPACE 跑 5090！")

    print(f"\n  🟢 邊際 alpha (wr↑ AND total -10~-30%): {len(margin_alpha)} 個")
    if margin_alpha:
        for r in sorted(margin_alpha, key=lambda r: -r["sharpe"])[:3]:
            print(f"     {r['alpha']} {r['mode']}: wr {r['wr_vs']:+.1f}%, totΔ {r['total_vs_pct']:+.1f}%, "
                  f"Sharpe {r['sharpe']:.3f}, 年化 {r['yearly']:.1f}")

    print(f"\n  🟡 弱 alpha (wr↑ AND total -30~-50%): {len(weak_alpha)} 個 (V39 月營收同級)")

    print()
    print("=" * 110)
    print("最終建議")
    print("=" * 110)

    if real_alpha:
        print(f"\n  🟢🟢 找到 {len(real_alpha)} 個真 net alpha → 寫 GPU patch 上 5090")
        print(f"     5090 evolutionary 用這些 alpha 當 buy gate，比 89.90 真實突破")
    elif margin_alpha:
        print(f"\n  🟢 {len(margin_alpha)} 個邊際 alpha (wr↑ total 略降)")
        print(f"     如果你重視 Sharpe / 風險調整 → 可加進 GPU 試")
        print(f"     如果你重視 absolute total return → 接受 89.90 final")
    else:
        print(f"\n  🔴 沒有 net alpha — WQ101 跟 V38 Kronos / V39 月營收同下場")
        print(f"     wr 提升的代價是 total 損失 → filter 對 89.90 net 負")
        print(f"\n  建議：5090 算力轉 Kronos fine-tune（真 deep learning, 不只 calibration head）")
        print(f"        Kronos fine-tune 是唯一沒嘗試過、5090 才能做的方向")

    out = os.path.join(USER_SE, "wq101_backfill_results.json")
    with open(out, "w") as f:
        json.dump({"baseline": a_stats, "results": results,
                   "real_alpha_count": len(real_alpha),
                   "margin_alpha_count": len(margin_alpha)}, f, indent=2, default=str)
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
