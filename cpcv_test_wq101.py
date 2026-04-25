"""
WQ101 CPCV LOO 驗證 — top alphas 跨 regime 泛化測試
用法：C:\\stock-evolution> python cpcv_test_wq101.py

V36 教訓：80/20 split 假象，必須 CPCV LOO
本 script 對 sanity 找出的 top alphas 做 15 path leave-one-path-out

輸入：wq101_alphas.csv（116 筆 × 37 alphas，由 sanity_test_wq101.py 產生）

對每個 alpha 跑 4 個 thresholds：
  - 留上半（>median, kept ~50%）
  - 留 top 30%（>p70, kept ~30%）
  - 留 top 50%（>p50, kept ~50%）
  - 留 bottom 30%（<p30, kept ~30，反向 alpha）

對 top 5 alphas equal-weight z-score combo 也跑 CPCV

判定（嚴於 sanity 假象）：
  🟢🟢🟢 strict: n_break ≥ 12 AND mean wr↑ ≥ 5% AND p25 ≥ 0
  🟢🟢 real:    n_break ≥ 10 AND mean wr↑ ≥ 4% AND p25 ≥ 0 AND n_pos ≥ 12
  🟢 marginal: n_break ≥ 7 AND mean wr↑ ≥ 3% AND p25 ≥ -1
  🔴 fail: 其他

對比 baseline:
  V38 zero-shot: n_break 11/14, mean +14.28%, p25 +5.58%
  V39 月營收 YoY > +20%: n_break 10/15, mean +5.68%, p25 +2.53%
"""
import os, sys, pickle, json
from itertools import combinations
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
ALPHAS_CSV = os.path.join(USER_SE, "wq101_alphas.csv")
N_GROUPS = 6
K_TEST = 2
WARMUP = 60


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def split_into_groups(n_days, warmup, n_groups):
    g_size = (n_days - warmup) // n_groups
    return [(warmup + i * g_size, warmup + (i + 1) * g_size if i < n_groups - 1 else n_days)
            for i in range(n_groups)]


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


def evaluate_alpha_cpcv(alpha_vals, rets, day_idx, n_days, alpha_name, alpha_sign,
                        test_combos, groups, mode="median", percentile=50):
    """
    對單一 alpha 跑 CPCV LOO
    mode:
      - "median": filter = vals > median (or < median if sign<0)
      - "topN": filter = vals > p(100-percentile) (or < p(percentile) if sign<0)
    回傳：per_path 統計 dict
    """
    per_path = []
    for pi, gi in enumerate(test_combos):
        ranges = [groups[g] for g in gi]
        in_test = np.zeros(len(rets), dtype=bool)
        for s, e in ranges:
            in_test |= (day_idx >= s) & (day_idx < e)
        if in_test.sum() < 5: continue

        rets_te = rets[in_test]
        vals_te = alpha_vals[in_test]

        # 過 NaN
        valid = np.isfinite(vals_te)
        if valid.sum() < 5: continue
        rets_te = rets_te[valid]
        vals_te = vals_te[valid]

        # 決定 keep 規則
        if mode == "median":
            cutoff = np.median(vals_te)
            keep = (vals_te > cutoff) if alpha_sign > 0 else (vals_te < cutoff)
        elif mode == "topN":
            if alpha_sign > 0:
                cutoff = np.percentile(vals_te, 100 - percentile)
                keep = vals_te > cutoff
            else:
                cutoff = np.percentile(vals_te, percentile)
                keep = vals_te < cutoff
        else:
            continue

        if keep.sum() < 3: continue

        raw_s = stats_of(rets_te)
        filt_s = stats_of(rets_te[keep])
        wr_imp = filt_s["wr"] - raw_s["wr"]
        kept_pct = filt_s["n"] / raw_s["n"] * 100
        total_imp = filt_s["total"] - raw_s["total"]
        per_path.append({
            "wr_imp": wr_imp, "kept_pct": kept_pct,
            "raw_wr": raw_s["wr"], "filt_wr": filt_s["wr"],
            "raw_total": raw_s["total"], "filt_total": filt_s["total"],
            "total_imp": total_imp,
        })
    return per_path


def grade(n_break, mean_wr_imp, p25, n_pos, n_valid):
    if n_break >= 12 and mean_wr_imp >= 5 and p25 >= 0:
        return "🟢🟢🟢 strict"
    elif n_break >= 10 and mean_wr_imp >= 4 and p25 >= 0 and n_pos >= 12:
        return "🟢🟢 real"
    elif n_break >= 7 and mean_wr_imp >= 3 and p25 >= -1:
        return "🟢 marginal"
    elif n_break >= 5:
        return "🟡 weak"
    else:
        return "🔴"


def report_alpha(per_path, label, n_total_paths):
    if len(per_path) < 5:
        return None
    wr_imps = np.array([p["wr_imp"] for p in per_path])
    kept_pcts = np.array([p["kept_pct"] for p in per_path])
    total_imps = np.array([p["total_imp"] for p in per_path])
    n_break = int((wr_imps >= 5).sum())
    n_pos = int((wr_imps > 0).sum())
    p25 = float(np.percentile(wr_imps, 25))
    mean_wr = float(wr_imps.mean())
    eval_str = grade(n_break, mean_wr, p25, n_pos, len(per_path))

    print(f"  {label:<22} {n_break:>2}/{len(per_path):<3} "
          f"mean {mean_wr:+5.2f}%  p25 {p25:+5.2f}%  "
          f"min {wr_imps.min():+5.2f}%  max {wr_imps.max():+5.2f}%  "
          f"n_pos {n_pos:>2}/{len(per_path):<3}  "
          f"kept {kept_pcts.mean():>5.1f}%  totΔ {total_imps.mean():+6.1f}%  {eval_str}")

    return {
        "label": label, "n_valid": len(per_path),
        "n_break": n_break, "n_pos": n_pos,
        "mean_wr_imp": mean_wr, "p25_wr_imp": p25,
        "min_wr_imp": float(wr_imps.min()), "max_wr_imp": float(wr_imps.max()),
        "median_wr_imp": float(np.median(wr_imps)),
        "mean_kept_pct": float(kept_pcts.mean()),
        "mean_total_imp": float(total_imps.mean()),
        "eval": eval_str,
    }


def main():
    print("=" * 100)
    print("WQ101 CPCV LOO — top alphas 跨 regime 驗證")
    print("=" * 100)

    if not os.path.exists(ALPHAS_CSV):
        print(f"❌ {ALPHAS_CSV} 不存在，先跑 sanity_test_wq101.py")
        return

    df = pd.read_csv(ALPHAS_CSV)
    print(f"\n[1/3] 讀 {len(df)} 筆 alpha 資料")

    # === 對齊 day_idx ===
    print(f"\n[2/3] 對齊 buy_date → day_idx...")
    params = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    n_days = pre["n_days"]
    dates = pre["dates"]
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}

    df["day_idx"] = df["buy_date"].map(date_to_day)
    df = df.dropna(subset=["day_idx"]).copy()
    df["day_idx"] = df["day_idx"].astype(int)
    print(f"  最終 {len(df)} 筆對齊")

    rets = df["actual_return"].values
    day_idx = df["day_idx"].values

    # 抓所有 alpha 欄位
    import re
    alpha_pattern = re.compile(r"^a\d{3}$")
    all_alphas = [c for c in df.columns if alpha_pattern.match(c)]

    # 從 sanity 排行讀 top 14 strong alphas（|Sp|≥0.10）
    # 重新算每個 alpha 的 Spearman 找 top 14
    from scipy import stats as scipy_stats
    alpha_sps = []
    for name in all_alphas:
        vals = df[name].values
        mask = np.isfinite(vals) & np.isfinite(rets)
        if mask.sum() < 30: continue
        sp, _ = scipy_stats.spearmanr(vals[mask], rets[mask])
        if not np.isfinite(sp): continue
        alpha_sps.append((name, float(sp)))
    alpha_sps.sort(key=lambda x: -abs(x[1]))

    top_alphas = [a for a in alpha_sps if abs(a[1]) >= 0.10]
    print(f"\n  Top alphas (|Sp|≥0.10): {len(top_alphas)} 個")

    # === CPCV LOO ===
    print(f"\n[3/3] CPCV LOO {N_GROUPS} groups, k={K_TEST}, total {len(list(combinations(range(N_GROUPS), K_TEST)))} paths")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))

    all_results = []

    # === Per-alpha CPCV ===
    print(f"\n{'─' * 100}")
    print(f"Per-alpha CPCV (median split)")
    print(f"{'─' * 100}")
    print(f"  {'Alpha (Sp)':<22} {'n_break':<7} {'mean':<7} {'p25':<7} "
          f"{'min':<7} {'max':<7} {'n_pos':<10} {'kept':<8} {'totΔ':<10} {'評價'}")
    print(f"  {'-' * 95}")

    for name, sp in top_alphas:
        sign = 1 if sp > 0 else -1
        vals = df[name].values
        per_path_med = evaluate_alpha_cpcv(vals, rets, day_idx, n_days, name, sign,
                                            test_combos, groups, mode="median")
        label = f"{name} (Sp{sp:+.3f})"
        r = report_alpha(per_path_med, label, len(test_combos))
        if r:
            r["alpha"] = name
            r["spearman"] = sp
            r["mode"] = "median"
            all_results.append(r)

    # === Top alpha 多 percentile sweep（看更嚴的 threshold 是否更強）===
    print(f"\n{'─' * 100}")
    print(f"Top 3 alphas — percentile threshold sweep (top 30% / top 20% / top 10%)")
    print(f"{'─' * 100}")
    print(f"  {'Alpha @ pct':<22} {'n_break':<7} {'mean':<7} {'p25':<7} "
          f"{'min':<7} {'max':<7} {'n_pos':<10} {'kept':<8} {'totΔ':<10} {'評價'}")
    print(f"  {'-' * 95}")

    for name, sp in top_alphas[:3]:
        sign = 1 if sp > 0 else -1
        vals = df[name].values
        for pct in [30, 20, 10]:
            per_path = evaluate_alpha_cpcv(vals, rets, day_idx, n_days, name, sign,
                                            test_combos, groups, mode="topN", percentile=pct)
            label = f"{name} top{pct}%"
            r = report_alpha(per_path, label, len(test_combos))
            if r:
                r["alpha"] = name
                r["spearman"] = sp
                r["mode"] = f"top{pct}"
                all_results.append(r)

    # === Top 5 combo z-score sum CPCV ===
    print(f"\n{'─' * 100}")
    print(f"Top 5 alphas combo (signed z-score sum)")
    print(f"{'─' * 100}")
    print(f"  {'Combo':<22} {'n_break':<7} {'mean':<7} {'p25':<7} "
          f"{'min':<7} {'max':<7} {'n_pos':<10} {'kept':<8} {'totΔ':<10} {'評價'}")
    print(f"  {'-' * 95}")

    if len(top_alphas) >= 5:
        top5 = top_alphas[:5]
        # 對全期算 z-score
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
        combo = np.where(valid_count > 0, combo / np.maximum(valid_count, 1), 0)
        combo = np.where(valid_count >= 3, combo, np.nan)

        # combo 已 signed，正方向，median split
        for mode_name, mode, pct in [("median", "median", 50),
                                       ("top30", "topN", 30),
                                       ("top20", "topN", 20)]:
            per_path = evaluate_alpha_cpcv(combo, rets, day_idx, n_days, "combo", 1,
                                            test_combos, groups, mode=mode, percentile=pct)
            label = f"top5 combo {mode_name}"
            r = report_alpha(per_path, label, len(test_combos))
            if r:
                r["alpha"] = "top5_combo"
                r["spearman"] = None
                r["mode"] = mode_name
                all_results.append(r)

    # === Summary ===
    print()
    print("=" * 100)
    print("📊 WQ101 CPCV 結論")
    print("=" * 100)

    print(f"\n【Baseline 對比】")
    print(f"  V38 Kronos zero-shot: n_break 11/14, mean +14.28%, p25 +5.58%, n_pos 13/14")
    print(f"  V39 月營收 YoY > +20%: n_break 10/15, mean +5.68%, p25 +2.53%, n_pos 14/15")

    # 找最強的
    strict_results = [r for r in all_results if "strict" in r["eval"]]
    real_results = [r for r in all_results if "real" in r["eval"]]
    marginal_results = [r for r in all_results if "marginal" in r["eval"]]

    print(f"\n【WQ101 結果】")
    print(f"  🟢🟢🟢 strict: {len(strict_results)} 個")
    print(f"  🟢🟢 real:    {len(real_results)} 個")
    print(f"  🟢 marginal: {len(marginal_results)} 個")
    print(f"  🔴 fail:     {len(all_results) - len(strict_results) - len(real_results) - len(marginal_results)} 個")

    if strict_results:
        best = max(strict_results, key=lambda r: r["mean_wr_imp"])
        print(f"\n  🏆 最強 (strict)：{best['label']}")
        print(f"     n_break {best['n_break']}/{best['n_valid']}, mean {best['mean_wr_imp']:+.2f}%, "
              f"p25 {best['p25_wr_imp']:+.2f}%, totΔ {best['mean_total_imp']:+.1f}%")
    elif real_results:
        best = max(real_results, key=lambda r: r["mean_wr_imp"])
        print(f"\n  🏆 最強 (real)：{best['label']}")
        print(f"     n_break {best['n_break']}/{best['n_valid']}, mean {best['mean_wr_imp']:+.2f}%, "
              f"p25 {best['p25_wr_imp']:+.2f}%, totΔ {best['mean_total_imp']:+.1f}%")
    elif marginal_results:
        best = max(marginal_results, key=lambda r: r["mean_wr_imp"])
        print(f"\n  最強 (marginal)：{best['label']}")
        print(f"     n_break {best['n_break']}/{best['n_valid']}, mean {best['mean_wr_imp']:+.2f}%, "
              f"p25 {best['p25_wr_imp']:+.2f}%, totΔ {best['mean_total_imp']:+.1f}%")

    print(f"\n【實盤建議】")
    if strict_results:
        print(f"  🟢🟢🟢 strict 級突破！直接寫 GPU patch 加進 PARAMS_SPACE 跑 5090")
        print(f"  推薦上線：5090 GPU evolutionary search 含這些 alpha 為 buy gate")
    elif real_results:
        print(f"  🟢🟢 real 級！可以加進 GPU PARAMS_SPACE 跑 5090，但留意 totΔ")
        if all(r["mean_total_imp"] < 0 for r in real_results):
            print(f"  ⚠️ 但所有 real 的 totΔ 都負 → 同 V39/Kronos 教訓（wr↑但 total↓）")
            print(f"     可能不值得整合（接受 89.90 final）")
    elif marginal_results:
        print(f"  🟡 marginal 級，邊緣突破 → 跟 V39 月營收同級不夠值得")
        print(f"     接受 89.90 final，5090 算力轉 Kronos fine-tune")
    else:
        print(f"  🔴 全部沒過 CPCV → sanity 是 multiple comparison 假象")
        print(f"     接受 89.90 final，5090 算力轉 Kronos fine-tune")

    # 存
    out_path = os.path.join(USER_SE, "wq101_cpcv_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_trades": len(df),
            "n_top_alphas": len(top_alphas),
            "results": all_results,
            "strict_count": len(strict_results),
            "real_count": len(real_results),
            "marginal_count": len(marginal_results),
        }, f, indent=2, default=str)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
