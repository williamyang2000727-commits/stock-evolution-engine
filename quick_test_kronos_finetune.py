"""
V38b Kronos Fine-tune 快速試水（10-15 分鐘）
用法：C:\\stock-evolution> python quick_test_kronos_finetune.py

目的：花 10-15 分鐘確認 fine-tune 方向有沒有用，再決定要不要花 4 小時做 full fine-tune

設計：
1. 不真的 fine-tune Kronos（要 4h），改用 lightweight 替代驗證：
   - 拿 zero-shot Kronos prediction（已有 kronos_sanity_results.csv 133 筆）
   - 訓練「單層 calibration head」修正 Kronos prediction 的 bias
   - 用 CPCV LOO 驗證有沒有比 zero-shot 強
2. 如果 calibration head 在 CPCV 上勝過 zero-shot → fine-tune 全 model 大概率也會贏
3. 如果 calibration head 都贏不了 → 不用做 full fine-tune（domain gap 太大）

原理：
- Kronos 是凍結的特徵抽取器
- 它的 raw prediction (pred_next, pred_5d) 可能有系統性 bias（比如台股預測偏多/偏空）
- 訓練一個 logistic regression 在 (pred_next, pred_5d, market_context...) 上預測「會不會贏」
- 等價於做「Meta-Labeling on Kronos features」
- 比 V36（hand-crafted features）強的可能性高，因為 Kronos features 本身就是強訊號

跟 V36 的差別：
- V36: 17 indicators + 2 market context = 19 hand-crafted features
- V38b: pred_next, pred_5d, pred_next×5d, market_context = ~5 Kronos-derived features
- 如果這個都失敗 → Kronos 真的沒泛化能力，fine-tune 也救不了
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
from itertools import combinations

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
SANITY_CSV = os.path.join(USER_SE, "kronos_sanity_results.csv")
N_GROUPS = 6
K_TEST = 2
WARMUP = 60


def split_into_groups(n_days, warmup, n_groups):
    g_size = (n_days - warmup) // n_groups
    return [(warmup + i * g_size, warmup + (i + 1) * g_size if i < n_groups - 1 else n_days)
            for i in range(n_groups)]


def stats_of(rets):
    if len(rets) == 0:
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0, "max_dd": 0.0}
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
    }


def main():
    print("=" * 70)
    print("V38b Kronos Calibration Head — Quick Test（10-15 分鐘）")
    print("=" * 70)
    print()
    print("目的：快速判斷 fine-tune 方向有沒有用")
    print("方法：訓練 calibration head 在 Kronos features 上 → CPCV 驗證")
    print()

    # === 1. 讀已有的 sanity CSV（V38 跑過了）===
    if not os.path.exists(SANITY_CSV):
        print(f"❌ {SANITY_CSV} 不存在")
        print(f"   請先跑 python sanity_test_kronos.py 拿 Kronos predictions")
        return

    df_sanity = pd.read_csv(SANITY_CSV)
    print(f"[1/5] 讀 sanity CSV：{len(df_sanity)} 筆 Kronos predictions ✅")

    # === 2. 對齊 buy_date 到 day index ===
    print(f"\n[2/5] 對齊 buy_date 到 CPCV groups...")
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

    df_sanity["day_idx"] = df_sanity["buy_date"].map(date_to_day)
    df_sanity = df_sanity.dropna(subset=["day_idx"]).copy()
    df_sanity["day_idx"] = df_sanity["day_idx"].astype(int)
    print(f"  對齊後 {len(df_sanity)} 筆")

    # === 3. 加 market context features ===
    print(f"\n[3/5] 加 market context features（大盤狀態）...")
    # pre dict 沒有 market_close key，自己從 close 算（v1 框架的大盤等權 = 所有股票每日 close 平均）
    close_arr = pre["close"]  # shape (n_stocks, n_days)
    market_close = np.nanmean(close_arr, axis=0)  # shape (n_days,)
    # 對每筆 trade，算 buy_date 當天的市場狀態
    market_features = []
    for di in df_sanity["day_idx"].values:
        if di < 20:
            market_features.append({"market_20d_ret": 0.0, "market_ma_ratio": 1.0})
            continue
        ret_20 = (market_close[di] / market_close[di - 20] - 1) * 100 if market_close[di - 20] > 0 else 0
        ma20 = np.mean(market_close[max(0, di-20):di])
        ma60 = np.mean(market_close[max(0, di-60):di])
        ma_ratio = ma20 / ma60 if ma60 > 0 else 1.0
        market_features.append({"market_20d_ret": float(ret_20), "market_ma_ratio": float(ma_ratio)})
    mf = pd.DataFrame(market_features)
    df_sanity = pd.concat([df_sanity.reset_index(drop=True), mf], axis=1)
    print(f"  features: pred_next_pct, pred_5d_pct, market_20d_ret, market_ma_ratio")

    # === 4. 設定 CPCV groups + 訓練 calibration head ===
    print(f"\n[4/5] CPCV LOO 訓練 calibration head（5 features → P(win)）...")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    print(f"  CPCV {N_GROUPS} groups, k={K_TEST}, total {len(test_combos)} paths")

    # Features 矩陣
    df_sanity["pred_next_x_5d"] = df_sanity["pred_next_pct"] * df_sanity["pred_5d_pct"]
    feature_cols = ["pred_next_pct", "pred_5d_pct", "pred_next_x_5d", "market_20d_ret", "market_ma_ratio"]
    X = df_sanity[feature_cols].values
    y = (df_sanity["actual_return"].values > 0).astype(int)
    print(f"  X shape: {X.shape}, y mean: {y.mean():.3f} (raw wr {y.mean()*100:.1f}%)")

    # CPCV LOO
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # 試多個 model + threshold
    model_specs = [
        ("LogReg", LogisticRegression(max_iter=500, C=1.0)),
        ("GB-shallow", GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)),
        ("GB-deeper", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
    ]

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

    print()
    best_overall = None

    for model_name, model_template in model_specs:
        print(f"  Model: {model_name}")
        for th in thresholds:
            per_path = []
            for pi, gi in enumerate(test_combos):
                ranges = [groups[g] for g in gi]
                in_test = np.zeros(len(df_sanity), dtype=bool)
                for s, e in ranges:
                    in_test |= (df_sanity["day_idx"].values >= s) & (df_sanity["day_idx"].values < e)

                test_mask = in_test
                train_mask = ~in_test

                if train_mask.sum() < 10 or test_mask.sum() < 5:
                    continue

                X_tr, X_te = X[train_mask], X[test_mask]
                y_tr, y_te = y[train_mask], y[test_mask]
                rets_te = df_sanity["actual_return"].values[test_mask]

                # Scale
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                # Train
                from sklearn.base import clone
                m = clone(model_template)
                try:
                    m.fit(X_tr_s, y_tr)
                    proba = m.predict_proba(X_te_s)[:, 1]
                except Exception:
                    continue

                # Filter
                keep = proba > th
                if keep.sum() < 3:
                    continue

                raw_stats = stats_of(rets_te)
                filt_stats = stats_of(rets_te[keep])
                wr_imp = filt_stats["wr"] - raw_stats["wr"]
                kept_pct = filt_stats["n"] / raw_stats["n"] * 100 if raw_stats["n"] > 0 else 0

                per_path.append({
                    "path_idx": pi,
                    "raw_wr": raw_stats["wr"],
                    "filt_wr": filt_stats["wr"],
                    "wr_imp": wr_imp,
                    "kept_pct": kept_pct,
                })

            if not per_path:
                continue

            wr_imps = np.array([p["wr_imp"] for p in per_path])
            kept_pcts = np.array([p["kept_pct"] for p in per_path])
            n_break = int((wr_imps >= 5).sum())
            n_pos = int((wr_imps > 0).sum())
            stats = {
                "model": model_name,
                "th": th,
                "n_valid": len(per_path),
                "n_break": n_break,
                "n_pos": n_pos,
                "mean_wr_imp": float(wr_imps.mean()),
                "median_wr_imp": float(np.median(wr_imps)),
                "p25_wr_imp": float(np.percentile(wr_imps, 25)),
                "min_wr_imp": float(wr_imps.min()),
                "max_wr_imp": float(wr_imps.max()),
                "mean_kept_pct": float(kept_pcts.mean()),
            }
            print(f"    th={th}: n_break {n_break:>2d}/{stats['n_valid']:>2d}, "
                  f"mean wr↑ {stats['mean_wr_imp']:+5.2f}%, p25 {stats['p25_wr_imp']:+5.2f}%, "
                  f"kept {stats['mean_kept_pct']:.1f}%, n_pos {n_pos}/{stats['n_valid']}")

            if best_overall is None or (n_break, stats["mean_wr_imp"]) > (best_overall["n_break"], best_overall["mean_wr_imp"]):
                best_overall = stats

    # === 5. 判定 ===
    print(f"\n[5/5] 判定...")
    print()
    print("=" * 70)
    print("📊 V38b Calibration Head 結果")
    print("=" * 70)

    if best_overall is None:
        print("\n🔴 全部 model 失敗（資料對齊問題或樣本太少）")
        return

    print(f"\n最強：{best_overall['model']} threshold={best_overall['th']}")
    print(f"  n_break = {best_overall['n_break']}/{best_overall['n_valid']}")
    print(f"  mean wr↑ = {best_overall['mean_wr_imp']:+.2f}%")
    print(f"  p25 wr↑ = {best_overall['p25_wr_imp']:+.2f}%")
    print(f"  median = {best_overall['median_wr_imp']:+.2f}%")
    print(f"  min/max = {best_overall['min_wr_imp']:+.2f}% / {best_overall['max_wr_imp']:+.2f}%")
    print(f"  positive paths = {best_overall['n_pos']}/{best_overall['n_valid']}")
    print(f"  mean kept = {best_overall['mean_kept_pct']:.1f}%")

    # V38 zero-shot baseline 參考
    print(f"\n【V38 zero-shot baseline（已知）】")
    print(f"  ensemble th=0.8: n_break 11/14, mean +14.28%, p25 +5.58%, n_pos 13/14")

    # 比較
    print(f"\n【V38b vs V38 zero-shot 比較】")
    if best_overall["mean_wr_imp"] > 14.28:
        print(f"  🟢🟢 V38b 比 zero-shot 強 → fine-tune full model 大機會贏更多")
        print(f"     建議：執行 finetune_kronos.py（4-6 小時）")
    elif best_overall["mean_wr_imp"] > 8 and best_overall["n_break"] >= 8:
        print(f"  🟢 V38b 跟 zero-shot 接近，calibration head 有效")
        print(f"     建議：值得做 full fine-tune（fine-tune 通常比 calibration 強）")
    elif best_overall["mean_wr_imp"] > 5:
        print(f"  🟡 V38b 比 zero-shot 弱但有效")
        print(f"     建議：full fine-tune 效益不確定，可先用 paper trading 驗 V38")
    else:
        print(f"  🔴 V38b 失敗 → fine-tune 大機會也救不了")
        print(f"     建議：跳過 fine-tune，直接用 V38 paper trading")

    # 存
    out_path = os.path.join(USER_SE, "kronos_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(best_overall, f, indent=2, default=str)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
