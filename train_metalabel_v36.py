"""
V36 Meta-Labeling 訓練 + CPCV 評估

Multi-model sanity 顯示 GradientBoosting AUC 0.746 最強（vs RF 0.643）
本腳本：
  1. 用 GradientBoosting + LightGBM 兩個 model 分別訓練
  2. CPCV leave-one-path-out 驗證（15 條 path）
  3. 每條 path 比較 raw（89.90 不過濾）vs filtered（V36 ML filter）
  4. 統計：多少 path wr 提升 ≥ 5%（真突破門檻 12/15）
  5. 存最終 model 為 pickle（線上 deploy 用）

真突破門檻（嚴於 V34/V35）：
  - ≥ 12/15 path wr 提升 ≥ 5%
  - mean wr 提升 ≥ 5%
  - p25 path wr 提升 ≥ 0%（最差 path 不能變更差）
  - 特別關注 path 0（89.90 最弱，wr 46.8%）
"""
import os, sys, pickle, json, warnings
import urllib.request
import numpy as np
from itertools import combinations

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path: sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base
from metalabel_features import extract_features_for_trades, FEATURE_NAMES


GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
N_GROUPS = 6
K_TEST = 2
WARMUP = 60


def fetch_gist_strategy():
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    content = d["files"]["best_strategy.json"]["content"]
    s = json.loads(content)
    return s.get("params", s), s.get("score", "N/A")


def split_into_groups(n_days: int, warmup: int, n_groups: int):
    usable = n_days - warmup
    g_size = usable // n_groups
    return [(warmup + i * g_size, warmup + (i + 1) * g_size if i < n_groups - 1 else n_days)
            for i in range(n_groups)]


def trades_in_groups(trades, group_indices, groups, dates):
    """篩出 buy_date 落在 group_indices 範圍的 trades"""
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}
    ranges = [groups[gi] for gi in group_indices]
    out = []
    for t in trades:
        bd = t.get("buy_date", "")
        if bd not in date_to_day:
            continue
        day = date_to_day[bd]
        for s, e in ranges:
            if s <= day < e:
                out.append(t)
                break
    return out


def build_model(model_type: str):
    """建一個 fresh model"""
    if model_type == "GB":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    elif model_type == "LightGBM":
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=15,
                min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1
            )
        except ImportError:
            return None
    return None


def stats_of(rets):
    if len(rets) == 0:
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0}
    rets = np.array(rets)
    return {
        "n": len(rets),
        "wr": (rets > 0).mean() * 100,
        "total": float(rets.sum()),
        "avg": float(rets.mean()),
    }


def evaluate_cpcv_loo(model_type: str, X_all, y_all, kept_indices, completed,
                     groups, dates, threshold: float = 0.50, verbose: bool = True):
    """
    CPCV leave-one-path-out 評估

    對每條 path：
      - Train data = 不在該 path test groups 範圍的 trades
      - Test data = 該 path test groups 範圍的 trades
      - Train model on train，predict on test
      - Filter test by threshold，比較 raw vs filtered metrics

    Returns:
        per_path: list of dict {path_idx, test_groups, raw_stats, filtered_stats, wr_improvement, ...}
        summary: aggregated stats
    """
    if verbose:
        print(f"\n[CPCV-LOO] Model = {model_type}, threshold = {threshold}")
        print(f"[CPCV-LOO] Total path = C({N_GROUPS},{K_TEST}) = {len(list(combinations(range(N_GROUPS), K_TEST)))}")

    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}

    # 每筆 trade 的 buy_day
    trade_buy_days = []
    for kt_i in kept_indices:
        bd = completed[kt_i].get("buy_date", "")
        trade_buy_days.append(date_to_day.get(bd, -1))
    trade_buy_days = np.array(trade_buy_days)

    per_path = []
    for path_idx, test_gi in enumerate(test_combos):
        # 該 path 的 test ranges
        test_ranges = [groups[g] for g in test_gi]
        # 哪些 trades 屬於 test path
        in_test = np.zeros(len(X_all), dtype=bool)
        for s, e in test_ranges:
            in_test |= (trade_buy_days >= s) & (trade_buy_days < e)

        # train = 補集
        in_train = ~in_test

        if in_test.sum() < 5 or in_train.sum() < 30:
            if verbose:
                print(f"[CPCV-LOO] path {path_idx} (test {list(test_gi)}): "
                     f"樣本不足 (train {in_train.sum()} test {in_test.sum()})，跳過")
            continue

        # 訓練
        model = build_model(model_type)
        if model is None:
            print(f"[CPCV-LOO] {model_type} 無法載入，跳過")
            return None, None
        try:
            model.fit(X_all[in_train], y_all[in_train])
        except Exception as e:
            print(f"[CPCV-LOO] path {path_idx} 訓練失敗：{e}")
            continue

        # Test 預測
        y_pred_proba = model.predict_proba(X_all[in_test])[:, 1]

        # Test trades returns
        test_trade_indices = np.where(in_test)[0]
        test_rets = np.array([float(completed[kept_indices[i]].get("return", 0)) for i in test_trade_indices])

        # Raw stats（不過濾）
        raw_stats = stats_of(test_rets)

        # Filtered stats（用 threshold 過濾）
        keep_mask = y_pred_proba >= threshold
        filtered_rets = test_rets[keep_mask]
        filtered_stats = stats_of(filtered_rets)

        # AUC（只在 test 上）
        try:
            test_auc = roc_auc_score(y_all[in_test], y_pred_proba)
        except Exception:
            test_auc = 0.5

        wr_imp = filtered_stats["wr"] - raw_stats["wr"] if filtered_stats["n"] > 0 else None

        per_path.append({
            "path_idx": path_idx,
            "test_groups": list(test_gi),
            "raw": raw_stats,
            "filtered": filtered_stats,
            "wr_improvement": wr_imp,
            "test_auc": test_auc,
            "kept_pct": filtered_stats["n"] / raw_stats["n"] * 100 if raw_stats["n"] > 0 else 0,
        })

        if verbose:
            wr_imp_str = f"{wr_imp:+.1f}%" if wr_imp is not None else "N/A"
            print(f"[CPCV-LOO] path {path_idx:2d} (test {list(test_gi)}): "
                 f"raw n={raw_stats['n']:3d} wr={raw_stats['wr']:5.1f}% | "
                 f"filtered n={filtered_stats['n']:3d} wr={filtered_stats['wr']:5.1f}% "
                 f"wr↑={wr_imp_str} | AUC={test_auc:.3f}")

    # 統計
    valid = [p for p in per_path if p["wr_improvement"] is not None and p["filtered"]["n"] >= 5]
    n_valid = len(valid)
    n_breakthrough = sum(1 for p in valid if p["wr_improvement"] >= 5)

    wr_imps = np.array([p["wr_improvement"] for p in valid]) if valid else np.array([0])
    raw_wrs = np.array([p["raw"]["wr"] for p in per_path])
    filtered_wrs = np.array([p["filtered"]["wr"] for p in valid]) if valid else np.array([0])
    aucs = np.array([p["test_auc"] for p in per_path])
    kept_pcts = np.array([p["kept_pct"] for p in valid]) if valid else np.array([0])

    summary = {
        "model_type": model_type,
        "threshold": threshold,
        "n_paths_total": len(per_path),
        "n_paths_valid": n_valid,
        "n_breakthrough": n_breakthrough,                     # wr_imp >= 5%
        "breakthrough_rate": n_breakthrough / n_valid if n_valid > 0 else 0,
        "mean_wr_improvement": float(wr_imps.mean()) if len(wr_imps) > 0 else 0,
        "median_wr_improvement": float(np.median(wr_imps)) if len(wr_imps) > 0 else 0,
        "p25_wr_improvement": float(np.percentile(wr_imps, 25)) if len(wr_imps) > 0 else 0,
        "min_wr_improvement": float(wr_imps.min()) if len(wr_imps) > 0 else 0,
        "max_wr_improvement": float(wr_imps.max()) if len(wr_imps) > 0 else 0,
        "mean_auc": float(aucs.mean()) if len(aucs) > 0 else 0.5,
        "mean_kept_pct": float(kept_pcts.mean()) if len(kept_pcts) > 0 else 0,
    }

    return per_path, summary


def print_summary(summary, per_path):
    """印 CPCV 總結"""
    print()
    print("=" * 70)
    print(f"📊 V36 {summary['model_type']} CPCV 評估總結（threshold={summary['threshold']}）")
    print("=" * 70)
    print(f"有效 path: {summary['n_paths_valid']}/{summary['n_paths_total']}")
    print(f"AUC 平均（each path test）: {summary['mean_auc']:.4f}")
    print(f"")
    print(f"WR 改善分布:")
    print(f"  Mean   = {summary['mean_wr_improvement']:+6.2f}%")
    print(f"  Median = {summary['median_wr_improvement']:+6.2f}%")
    print(f"  P25    = {summary['p25_wr_improvement']:+6.2f}%")
    print(f"  Min    = {summary['min_wr_improvement']:+6.2f}%")
    print(f"  Max    = {summary['max_wr_improvement']:+6.2f}%")
    print(f"")
    print(f"保留比例平均: {summary['mean_kept_pct']:.1f}%")
    print(f"")
    print(f"🎯 真突破判定（≥ 5% wr 改善的 path 數）:")
    print(f"   {summary['n_breakthrough']}/{summary['n_paths_valid']} path 達標 "
          f"({summary['breakthrough_rate']*100:.1f}%)")
    print()

    # 真突破門檻
    breakthrough = (
        summary['n_breakthrough'] >= 12 and
        summary['mean_wr_improvement'] >= 5 and
        summary['p25_wr_improvement'] >= 0
    )

    if breakthrough:
        print(f"🟢 V36 通過真突破門檻！可以實作上線")
    else:
        print(f"🟡 未通過真突破門檻：")
        if summary['n_breakthrough'] < 12:
            print(f"   - 達標 path 數 {summary['n_breakthrough']} < 12（門檻）")
        if summary['mean_wr_improvement'] < 5:
            print(f"   - mean wr 改善 {summary['mean_wr_improvement']:.1f}% < 5%（門檻）")
        if summary['p25_wr_improvement'] < 0:
            print(f"   - p25 wr 改善 {summary['p25_wr_improvement']:.1f}% < 0%（門檻：最差 path 不該變更差）")

    # 特別關注 path 0
    p0 = next((p for p in per_path if p["path_idx"] == 0), None)
    if p0 and p0["wr_improvement"] is not None:
        print(f"")
        print(f"🎯 Path 0（89.90 最弱 path，2020-2021 期間）:")
        print(f"   raw wr = {p0['raw']['wr']:.1f}% (89.90 baseline 46.8%)")
        print(f"   filtered wr = {p0['filtered']['wr']:.1f}%")
        print(f"   wr 改善 = {p0['wr_improvement']:+.1f}%")


def train_final_model(model_type, X_all, y_all, save_path):
    """訓練最終 model on 全資料，存 pickle"""
    print(f"\n[Final Train] 訓練最終 {model_type} on 全期 {len(X_all)} 筆...")
    model = build_model(model_type)
    if model is None:
        print(f"[Final Train] {model_type} 不可用")
        return False
    model.fit(X_all, y_all)
    with open(save_path, "wb") as f:
        pickle.dump({
            "model_type": model_type,
            "feature_names": FEATURE_NAMES,
            "model": model,
            "trained_n": len(X_all),
            "trained_wr": float(y_all.mean()),
        }, f)
    print(f"[Final Train] 存到 {save_path}")
    return True


def main():
    print("=" * 70)
    print("V36 Meta-Labeling 訓練 + CPCV 評估")
    print("=" * 70)

    # === 載入資料 ===
    print("\n[1/5] 載入 cache + cpu_replay 拿 89.90 trades...")
    params, score = fetch_gist_strategy()
    print(f"  89.90 score = {score}")

    cache_path = os.path.join(_USER_SE, "stock_data_cache.pkl")
    raw = pickle.load(open(cache_path, "rb"))
    _lens = [len(v) for v in raw.values()]
    _n_1500 = sum(1 for l in _lens if l >= 1500)
    _n_1200 = sum(1 for l in _lens if l >= 1200)
    if _n_1500 >= 500: TARGET_DAYS = 1500
    elif _n_1200 >= 800: TARGET_DAYS = 1200
    else: TARGET_DAYS = 900
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    pre = base.precompute(data)
    print(f"  precompute: {pre['n_stocks']} 檔 × {pre['n_days']} 天")

    print("\n[2/5] cpu_replay 拿 trades + 抽 features...")
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    X_all, y_all, kept_indices = extract_features_for_trades(pre, completed)
    print(f"  完成交易 {len(completed)} 筆 → 保留 {len(X_all)} 筆")
    print(f"  label 分布: 贏 {y_all.sum()} 輸 {len(y_all)-y_all.sum()}")

    # === CPCV groups ===
    n_days = pre["n_days"]
    dates = pre["dates"]
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    print(f"\n[3/5] CPCV 切 {N_GROUPS} groups:")
    for i, (s, e) in enumerate(groups):
        print(f"  group {i}: day {s}-{e} ({str(dates[s].date())[:10]} ~ {str(dates[min(e-1, n_days-1)].date())[:10]})")

    # === 跑 GB + LightGBM 兩個 model ===
    print(f"\n[4/5] CPCV leave-one-path-out 評估...")

    results = {}
    for model_type, threshold in [("GB", 0.40), ("LightGBM", 0.40)]:
        per_path, summary = evaluate_cpcv_loo(
            model_type, X_all, y_all, kept_indices, completed,
            groups, dates, threshold=threshold, verbose=True
        )
        if summary is not None:
            print_summary(summary, per_path)
            results[model_type] = (per_path, summary)

    # === 選最強 model 訓練 final + save ===
    print(f"\n[5/5] 選最強 model 訓練 final...")
    if not results:
        print("  ❌ 所有 model 都失敗")
        return

    # 選 mean_wr_improvement 最高且 n_breakthrough 最多
    winner = max(results.keys(),
                 key=lambda k: (results[k][1]["n_breakthrough"], results[k][1]["mean_wr_improvement"]))
    print(f"  🏆 V36 採用 = {winner}")

    save_path = os.path.join(_USER_SE, "metalabel_v36_model.pkl")
    train_final_model(winner, X_all, y_all, save_path)

    # 儲存 CPCV 結果為 JSON（用於後續分析）
    json_path = os.path.join(_USER_SE, "metalabel_v36_cpcv_results.json")
    json_safe = {}
    for k, (paths, summary) in results.items():
        json_safe[k] = {
            "summary": summary,
            "per_path": [
                {kk: vv for kk, vv in p.items() if kk != "_internal"}
                for p in paths
            ],
        }
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"  CPCV JSON 存到 {json_path}")

    print(f"\n" + "=" * 70)
    print("✅ V36 訓練完成！")
    print(f"   Model: {winner}")
    print(f"   Pickle: {save_path}")
    print(f"   CPCV: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
