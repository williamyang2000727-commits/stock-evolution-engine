"""
基本面 CPCV LOO 驗證 — 月營收 YoY 跨 regime 泛化測試
用法：C:\\stock-evolution> python cpcv_test_revenue.py

V36 教訓：80/20 split 假象，必須 CPCV LOO
此 script 對月營收 filter 做 15 path leave-one-path-out

對比 baseline:
  - V38 zero-shot: n_break 11/14, mean +14.28%, p25 +5.58%
  - V38d: n_break 8/11, mean +18.02%, p25 +0.54%

判定（嚴格）:
  🟢🟢 strict: n_break >= 12 AND mean >= 5% AND p25 >= 0
  🟢 real: n_break >= 10 AND mean >= 4% AND p25 >= 0 AND n_pos >= 12
  🟡 marginal: n_break >= 7 AND mean >= 3% AND p25 >= -1
  🔴 fail: 其他
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
YOY_CSV = os.path.join(USER_SE, "revenue_yoy_results.csv")
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


def main():
    print("=" * 70)
    print("基本面 CPCV LOO — 月營收 YoY 跨 regime 驗證")
    print("=" * 70)

    if not os.path.exists(YOY_CSV):
        print(f"❌ {YOY_CSV} 不存在")
        return

    df_yoy = pd.read_csv(YOY_CSV)
    print(f"\n[1/3] 讀 {len(df_yoy)} 筆 YoY 資料")

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

    df_yoy["day_idx"] = df_yoy["buy_date"].map(date_to_day)
    df_yoy = df_yoy.dropna(subset=["day_idx"]).copy()
    df_yoy["day_idx"] = df_yoy["day_idx"].astype(int)
    print(f"  最終 {len(df_yoy)} 筆對齊")

    yoy = df_yoy["yoy_pct"].values
    rets = df_yoy["actual_return"].values
    day_idx = df_yoy["day_idx"].values

    # === CPCV LOO ===
    print(f"\n[3/3] CPCV LOO sweep over thresholds...")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    print(f"  CPCV {N_GROUPS} groups, k={K_TEST}, total {len(test_combos)} paths")
    print()

    print(f"{'Threshold':<15} {'n_break':<10} {'mean wr↑':<11} {'median':<10} {'p25':<10} {'min':<10} {'max':<10} {'n_pos':<10} {'kept%':<8} {'評價'}")
    print("-" * 110)

    best_overall = None
    for th in [-20, -10, 0, 10, 20, 30, 50]:
        per_path = []
        for pi, gi in enumerate(test_combos):
            ranges = [groups[g] for g in gi]
            in_test = np.zeros(len(df_yoy), dtype=bool)
            for s, e in ranges:
                in_test |= (day_idx >= s) & (day_idx < e)
            test_mask = in_test
            if test_mask.sum() < 5: continue

            rets_te = rets[test_mask]
            yoy_te = yoy[test_mask]
            keep = yoy_te > th
            if keep.sum() < 3: continue

            raw_s = stats_of(rets_te)
            filt_s = stats_of(rets_te[keep])
            wr_imp = filt_s["wr"] - raw_s["wr"]
            kept_pct = filt_s["n"] / raw_s["n"] * 100
            per_path.append({
                "wr_imp": wr_imp, "kept_pct": kept_pct,
                "raw_wr": raw_s["wr"], "filt_wr": filt_s["wr"],
                "raw_total": raw_s["total"], "filt_total": filt_s["total"],
            })

        if not per_path:
            continue

        wr_imps = np.array([p["wr_imp"] for p in per_path])
        kept_pcts = np.array([p["kept_pct"] for p in per_path])
        n_break = int((wr_imps >= 5).sum())
        n_pos = int((wr_imps > 0).sum())

        # 評價
        if n_break >= 12 and wr_imps.mean() >= 5 and np.percentile(wr_imps, 25) >= 0:
            eval_str = "🟢🟢🟢 strict"
        elif n_break >= 10 and wr_imps.mean() >= 4 and np.percentile(wr_imps, 25) >= 0 and n_pos >= 12:
            eval_str = "🟢🟢 real"
        elif n_break >= 7 and wr_imps.mean() >= 3 and np.percentile(wr_imps, 25) >= -1:
            eval_str = "🟢 marginal"
        elif n_break >= 5:
            eval_str = "🟡 weak"
        else:
            eval_str = "🔴"

        print(f"YoY > {th:+>3}%      {n_break:>2}/{len(per_path):<5}    "
              f"{wr_imps.mean():+5.2f}%    {np.median(wr_imps):+5.2f}%    "
              f"{np.percentile(wr_imps,25):+5.2f}%    {wr_imps.min():+5.2f}%    "
              f"{wr_imps.max():+5.2f}%    {n_pos}/{len(per_path):<5}    "
              f"{kept_pcts.mean():>5.1f}%   {eval_str}")

        result = {
            "th": th, "n_valid": len(per_path),
            "n_break": n_break, "n_pos": n_pos,
            "mean_wr_imp": float(wr_imps.mean()),
            "median_wr_imp": float(np.median(wr_imps)),
            "p25_wr_imp": float(np.percentile(wr_imps, 25)),
            "min_wr_imp": float(wr_imps.min()),
            "max_wr_imp": float(wr_imps.max()),
            "mean_kept_pct": float(kept_pcts.mean()),
        }
        if best_overall is None or (n_break, wr_imps.mean()) > (best_overall["n_break"], best_overall["mean_wr_imp"]):
            best_overall = result

    # === 結論 ===
    print()
    print("=" * 70)
    print("📊 基本面 CPCV 結果")
    print("=" * 70)

    print(f"\n【V38 zero-shot baseline】n_break 11/14, mean +14.28%, p25 +5.58%, n_pos 13/14")
    print(f"【V38d baseline】           n_break  8/11, mean +18.02%, p25 +0.54%, n_pos 8/11")

    if best_overall:
        print(f"\n【月營收 YoY best】th = {best_overall['th']}")
        print(f"  n_break = {best_overall['n_break']}/{best_overall['n_valid']}")
        print(f"  mean wr↑ = {best_overall['mean_wr_imp']:+.2f}%")
        print(f"  p25 wr↑ = {best_overall['p25_wr_imp']:+.2f}%")
        print(f"  n_pos = {best_overall['n_pos']}/{best_overall['n_valid']}")
        print(f"  保留率 = {best_overall['mean_kept_pct']:.1f}%")

        print(f"\n【vs V38 zero-shot】")
        if best_overall["mean_wr_imp"] >= 14.28 and best_overall["p25_wr_imp"] >= 5.58:
            print(f"  🟢🟢 月營收 strict 勝 V38 zero-shot！")
        elif best_overall["mean_wr_imp"] >= 14.28:
            print(f"  🟢 月營收 mean 贏 V38（p25 各有勝負）")
        elif best_overall["mean_wr_imp"] >= 8:
            print(f"  🟡 月營收 mean 略弱於 V38 但仍有 alpha")
        else:
            print(f"  🔴 月營收 mean 弱於 V38")

        # 實盤建議：基於 backfill 結果（YoY > 0% total -13% 但 wr +4%）
        print(f"\n【實盤建議】")
        if best_overall["mean_wr_imp"] >= 5 and best_overall["mean_kept_pct"] >= 60:
            print(f"  🟢 可實盤試（保留 >= 60% 不損失太多 total）")
            print(f"  推薦 YoY > 0%：kept 75%, total -13%, wr +4.3%, Sharpe +13%")
        elif best_overall["mean_wr_imp"] >= 5:
            print(f"  🟡 過 CPCV 但 kept 太低 → 實盤頻率不夠")
        else:
            print(f"  🔴 CPCV 沒過 → 接受 89.90 final")

    out = os.path.join(USER_SE, "revenue_cpcv_results.json")
    with open(out, "w") as f:
        json.dump(best_overall, f, indent=2, default=str)
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
