"""
V38c Calibration Head 改進版（10-15 分鐘）
用法：C:\\stock-evolution> python quick_test_v38c.py

V38b 失敗原因分析：
  - 5 features 太少（只有 Kronos + market context）
  - LogReg / GB 在小樣本（133 筆）跨 regime 不泛化
  - 各 model + threshold 全部 mean wr↑ < 0

V38c 改進：
  1. Features 從 5 → 21（加 17 indicators，借 V36 的 metalabel_features）
     - 17 stock indicators (RSI/MACD/ADX/...)
     - 2 market context (market_20d_return, market_ma_ratio)
     - 2 Kronos features (pred_next_pct, pred_5d_pct)
  2. ML model 簡化 + L2 防過擬合
     - LogReg with C=0.1, 0.3, 1.0（不同 L2 強度）
     - 動態 threshold = train set proba median + offset
  3. 也試「Ensemble: Kronos rule AND ML proba > th」
     - 結合 hand rule 跟 ML
     - 兩個都過才買，過濾 false positive

判定門檻：
  🟢🟢 mean wr↑ > 14.28% (超過 V38 zero-shot)
  🟢 mean wr↑ > 8% AND n_break >= 8
  🟡 mean wr↑ > 5%
  🔴 其他 → 接受 V38 zero-shot final，不再 ML
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
from metalabel_features import extract_features_for_trades, FEATURE_NAMES

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
SANITY_CSV = os.path.join(USER_SE, "kronos_sanity_results.csv")
N_GROUPS = 6
K_TEST = 2
WARMUP = 60


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s), s.get("score", "N/A")


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
    print("V38c Calibration Head 改進版 — Quick Test")
    print("=" * 70)
    print()
    print("V38b 失敗 → V38c 加 17 indicators + L2 防過擬合 + ensemble rule")
    print()

    # === 1. 讀 sanity CSV + 89.90 trades ===
    if not os.path.exists(SANITY_CSV):
        print(f"❌ {SANITY_CSV} 不存在")
        return

    df_sanity = pd.read_csv(SANITY_CSV)
    print(f"[1/6] 讀 sanity CSV：{len(df_sanity)} 筆 Kronos predictions ✅")

    # === 2. 跑 89.90 拿完整 trades（為了抽 indicators）===
    print(f"\n[2/6] 跑 89.90 cpu_replay 拿完整 trades...")
    params, _ = fetch_gist_strategy()
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

    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    print(f"  89.90 trades: {len(completed)} 筆")

    # === 3. 抽 19 features（17 indicators + 2 market context）===
    print(f"\n[3/6] 抽 19-d features 從 metalabel_features...")
    X19, y, keep_indices = extract_features_for_trades(pre, completed)
    print(f"  X19 shape: {X19.shape}, y mean: {y.mean():.3f} (raw wr {y.mean()*100:.1f}%)")

    # 把 keep_indices 對應的 trade 跟 sanity CSV 合併（by ticker + buy_date）
    print(f"\n[4/6] 合併 trades + Kronos sanity (by ticker + buy_date)...")
    trades_kept = [completed[i] for i in keep_indices]
    df_trades = pd.DataFrame([{
        "ticker": t.get("ticker"),
        "buy_date": t.get("buy_date"),
        "actual_return": float(t.get("return", 0)),
    } for t in trades_kept])
    df_trades = df_trades.reset_index(drop=True)

    # Sanity CSV 必須包含 ticker + buy_date 才能 match
    if "ticker" not in df_sanity.columns or "buy_date" not in df_sanity.columns:
        print(f"  ⚠️ sanity CSV 缺欄位，無法合併。col: {df_sanity.columns.tolist()}")
        return

    # Merge
    df_merged = df_trades.merge(
        df_sanity[["ticker", "buy_date", "pred_next_pct", "pred_5d_pct"]],
        on=["ticker", "buy_date"], how="inner"
    )
    if len(df_merged) < 50:
        print(f"  ⚠️ 合併後只剩 {len(df_merged)} 筆，太少")
        return
    print(f"  合併後 {len(df_merged)} 筆有完整 features + Kronos predictions")

    # 重新 align X19
    # df_trades 是 keep_indices 順序，df_merged 是合併後子集
    # 用 ticker + buy_date 找回 X19 對應的 row
    trade_key = [(t.get("ticker"), t.get("buy_date")) for t in trades_kept]
    merged_keys = list(zip(df_merged["ticker"], df_merged["buy_date"]))
    aligned_idx = [trade_key.index(k) for k in merged_keys]
    X19_m = X19[aligned_idx]
    y_m = y[aligned_idx]
    rets_m = df_merged["actual_return"].values
    p_next = df_merged["pred_next_pct"].values
    p_5d = df_merged["pred_5d_pct"].values

    # 21 features = 19 indicators + 2 Kronos
    X21 = np.column_stack([X19_m, p_next, p_5d])
    feature_names_21 = FEATURE_NAMES + ["pred_next_pct", "pred_5d_pct"]
    print(f"  X21 shape: {X21.shape}, features: 17 indicators + 2 market + 2 Kronos = 21")

    # day_idx for CPCV split
    day_idx = np.array([date_to_day.get(bd, -1) for bd in df_merged["buy_date"]])
    valid_mask = day_idx >= 0
    X21 = X21[valid_mask]
    y_m = y_m[valid_mask]
    rets_m = rets_m[valid_mask]
    day_idx = day_idx[valid_mask]
    p_next = p_next[valid_mask]
    p_5d = p_5d[valid_mask]
    print(f"  最終 {len(X21)} 筆有 day_idx")

    # === 5. CPCV LOO 訓練多個 model 配置 ===
    print(f"\n[5/6] CPCV LOO 多配置 sweep...")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    print(f"  CPCV {N_GROUPS} groups, k={K_TEST}, total {len(test_combos)} paths")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # 配置：(model name, C value, threshold mode, threshold offset)
    # threshold mode = "abs"（絕對 prob）或 "rel"（train set median + offset）
    configs = []
    # LogReg L2 不同強度
    for C in [0.1, 0.3, 1.0]:
        for th_offset in [0.0, 0.05, 0.10]:
            configs.append({
                "name": f"LogReg-C{C}-relmed+{th_offset:.2f}",
                "C": C, "th_mode": "rel", "th_offset": th_offset,
                "kronos_gate": False,
            })
    # 加 Kronos rule + ML（ensemble）
    for C in [0.3, 1.0]:
        for th_offset in [0.0, 0.05]:
            configs.append({
                "name": f"Ensemble-LR-C{C}-relmed+{th_offset:.2f}",
                "C": C, "th_mode": "rel", "th_offset": th_offset,
                "kronos_gate": True,  # next > 0 AND 5d > path median
            })
    # 絕對 threshold 對照
    for C in [0.3]:
        for th_abs in [0.55, 0.60, 0.65]:
            configs.append({
                "name": f"LogReg-C{C}-abs{th_abs}",
                "C": C, "th_mode": "abs", "th_abs": th_abs,
                "kronos_gate": False,
            })

    print(f"\n  總共 {len(configs)} 個配置")
    print()

    best_overall = None

    for cfg in configs:
        per_path = []
        for pi, gi in enumerate(test_combos):
            ranges = [groups[g] for g in gi]
            in_test = np.zeros(len(X21), dtype=bool)
            for s, e in ranges:
                in_test |= (day_idx >= s) & (day_idx < e)

            test_mask = in_test
            train_mask = ~in_test
            if train_mask.sum() < 15 or test_mask.sum() < 5:
                continue

            X_tr, X_te = X21[train_mask], X21[test_mask]
            y_tr = y_m[train_mask]
            rets_te = rets_m[test_mask]
            p_next_te = p_next[test_mask]
            p_5d_te = p_5d[test_mask]

            # Standardize
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            try:
                m = LogisticRegression(C=cfg["C"], max_iter=500, penalty="l2")
                m.fit(X_tr_s, y_tr)
                proba_tr = m.predict_proba(X_tr_s)[:, 1]
                proba_te = m.predict_proba(X_te_s)[:, 1]
            except Exception:
                continue

            # Threshold
            if cfg["th_mode"] == "rel":
                th = float(np.median(proba_tr)) + cfg["th_offset"]
            else:
                th = cfg["th_abs"]

            keep_ml = proba_te > th

            # Kronos gate (optional)
            if cfg["kronos_gate"]:
                med_5d = np.median(p_5d_te)
                keep_kr = (p_next_te > 0) & (p_5d_te > med_5d)
                keep = keep_ml & keep_kr
            else:
                keep = keep_ml

            if keep.sum() < 3:
                continue

            raw_stats = stats_of(rets_te)
            filt_stats = stats_of(rets_te[keep])
            wr_imp = filt_stats["wr"] - raw_stats["wr"]
            kept_pct = filt_stats["n"] / raw_stats["n"] * 100 if raw_stats["n"] > 0 else 0

            per_path.append({
                "wr_imp": wr_imp, "kept_pct": kept_pct,
                "raw_wr": raw_stats["wr"], "filt_wr": filt_stats["wr"],
            })

        if not per_path:
            continue

        wr_imps = np.array([p["wr_imp"] for p in per_path])
        kept_pcts = np.array([p["kept_pct"] for p in per_path])
        n_break = int((wr_imps >= 5).sum())
        n_pos = int((wr_imps > 0).sum())
        stats_d = {
            "name": cfg["name"],
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

        # Print 簡短
        flag = "🟢" if stats_d["mean_wr_imp"] >= 5 else ("🟡" if stats_d["mean_wr_imp"] >= 0 else "🔴")
        print(f"  {flag} {cfg['name']:<38s}: n_break {n_break:>2d}/{stats_d['n_valid']:>2d}, "
              f"mean {stats_d['mean_wr_imp']:+5.2f}%, p25 {stats_d['p25_wr_imp']:+5.2f}%, "
              f"kept {stats_d['mean_kept_pct']:.0f}%, n_pos {n_pos}/{stats_d['n_valid']}")

        if best_overall is None or (n_break, stats_d["mean_wr_imp"]) > (best_overall["n_break"], best_overall["mean_wr_imp"]):
            best_overall = stats_d

    # === 6. 判定 ===
    print(f"\n[6/6] 判定...")
    print()
    print("=" * 70)
    print("📊 V38c Calibration Head 改進版 結果")
    print("=" * 70)

    if best_overall is None:
        print("\n🔴 全部 model 失敗")
        return

    print(f"\n最強：{best_overall['name']}")
    print(f"  n_break = {best_overall['n_break']}/{best_overall['n_valid']}")
    print(f"  mean wr↑ = {best_overall['mean_wr_imp']:+.2f}%")
    print(f"  p25 wr↑ = {best_overall['p25_wr_imp']:+.2f}%")
    print(f"  median = {best_overall['median_wr_imp']:+.2f}%")
    print(f"  min/max = {best_overall['min_wr_imp']:+.2f}% / {best_overall['max_wr_imp']:+.2f}%")
    print(f"  positive paths = {best_overall['n_pos']}/{best_overall['n_valid']}")
    print(f"  mean kept = {best_overall['mean_kept_pct']:.1f}%")

    print(f"\n【V38 zero-shot baseline】ensemble th=0.8: n_break 11/14, mean +14.28%, n_pos 13/14")
    print(f"【V38b baseline】LogReg th=0.65: n_break 3/15, mean -3.35%（失敗）")

    print(f"\n【V38c 比較】")
    if best_overall["mean_wr_imp"] > 14.28:
        print(f"  🟢🟢🟢 V38c 比 zero-shot 強！直接上線取代 V38")
    elif best_overall["mean_wr_imp"] > 8 and best_overall["n_break"] >= 8:
        print(f"  🟢 V38c 強，可進 production")
    elif best_overall["mean_wr_imp"] > 5:
        print(f"  🟡 V38c 邊際進步")
    elif best_overall["mean_wr_imp"] > 0 and best_overall["mean_wr_imp"] > -3.35:
        print(f"  🟡 V38c 比 V38b 好但仍輸 zero-shot → 接受 V38 final")
    else:
        print(f"  🔴 V38c 仍失敗 → 證實 ML 在這 sample size 跨 regime 不可行")
        print(f"     接受 V38 zero-shot 是最佳，等 paper trading 結果")

    out_path = os.path.join(USER_SE, "v38c_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(best_overall, f, indent=2, default=str)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
