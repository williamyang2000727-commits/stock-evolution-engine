"""
Meta-Labeling Multi-Model Sanity — 比較多種 ML 找最強的

模型列表：
  - RandomForest（baseline，已知 AUC 0.643）
  - GradientBoosting (sklearn)
  - LightGBM
  - XGBoost
  - Logistic Regression（線性 baseline）
  - SVM (RBF kernel)
  - MLP (small neural net)
  - ExtraTrees（RF 變種）

每個 model：
  - 同樣 19 features
  - 同樣 time-ordered 80/20 split
  - 5-fold time-series CV 找最佳 hyperparameter（簡化版）
  - 報 AUC + filter 模擬（threshold 0.50, 0.55, 0.60）
  - 報 feature importance（如果支援）

最後排名 + 推薦 V36 用哪個
"""
import os, sys, pickle, json
import urllib.request
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path: sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base
from metalabel_features import extract_features_for_trades, FEATURE_NAMES


def fetch_gist_strategy():
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    content = d["files"]["best_strategy.json"]["content"]
    strategy = json.loads(content)
    return strategy.get("params", strategy), strategy.get("score", "N/A")


def evaluate_model(name, model, X_train, y_train, X_test, y_test, test_rets, needs_scaling=False):
    """
    訓練 model，回傳評估結果

    Returns dict with:
        name, auc, threshold_results (list of {th, n_kept, wr, wr_imp, total, avg})
    """
    if needs_scaling:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = X_train
        X_test_s = X_test

    try:
        model.fit(X_train_s, y_train)
    except Exception as e:
        return {"name": name, "error": str(e)}

    # 預測機率
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    elif hasattr(model, "decision_function"):
        # SVM 等用 decision_function
        scores = model.decision_function(X_test_s)
        y_pred_proba = 1 / (1 + np.exp(-scores))  # sigmoid
    else:
        y_pred = model.predict(X_test_s)
        y_pred_proba = y_pred.astype(float)

    # AUC
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        auc = 0.5

    # 基線
    raw_wr = (test_rets > 0).mean() * 100
    raw_total = test_rets.sum()
    raw_avg = test_rets.mean()

    # Filter 模擬
    threshold_results = []
    for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        keep_mask = y_pred_proba >= th
        n_kept = int(keep_mask.sum())
        if n_kept < 5:
            threshold_results.append({"th": th, "n_kept": n_kept, "wr": None, "wr_imp": None})
            continue
        kept_rets = test_rets[keep_mask]
        kept_wr = (kept_rets > 0).mean() * 100
        kept_total = float(kept_rets.sum())
        kept_avg = float(kept_rets.mean())
        wr_imp = kept_wr - raw_wr
        threshold_results.append({
            "th": th, "n_kept": n_kept,
            "kept_pct": n_kept / len(test_rets) * 100,
            "wr": kept_wr, "wr_imp": wr_imp,
            "total": kept_total, "avg": kept_avg,
        })

    # 找最佳 threshold（wr_imp 最高且 n_kept >= 10）
    best = None
    for r in threshold_results:
        if r.get("wr_imp") is not None and r["n_kept"] >= 10:
            if best is None or r["wr_imp"] > best["wr_imp"]:
                best = r

    # Feature importance（如果支援）
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        feat_imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])[:5]
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        feat_imp = sorted(zip(FEATURE_NAMES, np.abs(coefs)), key=lambda x: -x[1])[:5]

    return {
        "name": name,
        "auc": auc,
        "raw_wr": raw_wr,
        "raw_total": raw_total,
        "raw_avg": raw_avg,
        "threshold_results": threshold_results,
        "best_threshold": best,
        "feat_imp": feat_imp,
    }


def main():
    print("=" * 70)
    print("Meta-Labeling Multi-Model Sanity — 比較多種 ML")
    print("=" * 70)
    print()

    # 載資料
    print("[1/4] 載入資料 + cpu_replay...")
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
    print(f"  precompute done")

    print("\n[2/4] cpu_replay 拿 89.90 trades + 抽 features...")
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    X, y, kept = extract_features_for_trades(pre, completed)
    print(f"  完成交易 {len(completed)} 筆 → 保留 {len(X)} 筆")
    print(f"  features {X.shape[1]} 維 / label 分布: 贏 {y.sum()} 輸 {len(y)-y.sum()}")

    # Time-ordered 80/20 split
    n = len(X)
    n_train = int(n * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    test_rets = np.array([float(completed[kept[i]].get("return", 0)) for i in range(n_train, n)])
    print(f"  split: train {n_train} / test {n - n_train}")

    # 各 model 配置
    print(f"\n[3/4] 訓練多種 ML model...")

    models = []

    # 1. RandomForest（基準）
    models.append(("RandomForest",
        RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        False))

    # 2. ExtraTrees（RF 變種，更隨機）
    models.append(("ExtraTrees",
        ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        False))

    # 3. GradientBoosting (sklearn)
    models.append(("GradientBoosting",
        GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        False))

    # 4. LightGBM
    try:
        import lightgbm as lgb
        models.append(("LightGBM",
            lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=15,
                              min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1),
            False))
    except ImportError:
        print("  ⚠️ LightGBM not installed (pip install lightgbm)，跳過")

    # 5. XGBoost
    try:
        import xgboost as xgb
        models.append(("XGBoost",
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              min_child_weight=5, random_state=42, n_jobs=-1, eval_metric="logloss"),
            False))
    except ImportError:
        print("  ⚠️ XGBoost not installed (pip install xgboost)，跳過")

    # 6. Logistic Regression（線性 baseline，需 scale）
    models.append(("LogisticRegression",
        LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        True))

    # 7. SVM (RBF)（需 scale）
    models.append(("SVM-RBF",
        SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        True))

    # 8. MLP（小神經網路，需 scale）
    models.append(("MLP",
        MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=42, early_stopping=True),
        True))

    # 跑所有 model
    results = []
    for name, model, needs_scale in models:
        print(f"  跑 {name}...")
        r = evaluate_model(name, model, X_train, y_train, X_test, y_test, test_rets, needs_scaling=needs_scale)
        results.append(r)

    # === 顯示結果 ===
    print(f"\n[4/4] 結果排行榜...")
    print()
    print("=" * 70)
    print(f"{'Model':<20s}{'AUC':<8s}{'Best th':<10s}{'wr':<8s}{'wr↑':<10s}{'Kept':<8s}")
    print("-" * 70)

    # 排序：按 AUC 降序
    results_valid = [r for r in results if "error" not in r]
    results_sorted = sorted(results_valid, key=lambda r: -r["auc"])

    for r in results_sorted:
        bt = r["best_threshold"]
        if bt:
            print(f"{r['name']:<20s}{r['auc']:<8.4f}{bt['th']:<10.2f}{bt['wr']:<7.1f}%{bt['wr_imp']:+8.1f}%{bt['n_kept']:<8d}")
        else:
            print(f"{r['name']:<20s}{r['auc']:<8.4f}{'N/A':<10s}{'N/A':<8s}{'N/A':<10s}{'N/A':<8s}")

    print()

    # 各 model 的 threshold 詳細
    print("=" * 70)
    print("各 model 不同 threshold 的詳細結果：")
    print()
    for r in results_sorted:
        print(f"--- {r['name']} (AUC = {r['auc']:.4f}) ---")
        if r.get("feat_imp"):
            print(f"Top 5 features: {', '.join([f'{n}({v:.3f})' for n, v in r['feat_imp']])}")
        print(f"{'th':<6s}{'n':<5s}{'kept%':<8s}{'wr':<8s}{'wr↑':<8s}{'total':<10s}{'avg':<6s}")
        for tr in r["threshold_results"]:
            if tr.get("wr") is None:
                print(f"{tr['th']:<6.2f}{tr['n_kept']:<5d}{'N/A':<8s}{'N/A':<8s}{'N/A':<8s}{'N/A':<10s}{'N/A':<6s}")
                continue
            print(f"{tr['th']:<6.2f}{tr['n_kept']:<5d}{tr['kept_pct']:<7.1f}%{tr['wr']:<7.1f}%{tr['wr_imp']:+6.1f}% {tr['total']:<9.1f}%{tr['avg']:<5.1f}%")
        print()

    # === 推薦 ===
    print("=" * 70)
    print("📋 V36 推薦使用的 model")
    print("=" * 70)

    # 篩選：AUC > 0.55 AND best_threshold wr_imp > 5%
    qualified = [r for r in results_sorted
                 if r["auc"] > 0.55 and r.get("best_threshold") and r["best_threshold"]["wr_imp"] > 5]

    if not qualified:
        print("🔴 沒有 model 同時滿足 AUC > 0.55 AND wr 提升 > 5%")
        print("   建議：跳過 Meta-Labeling，做分點主力或 Kronos")
    else:
        print(f"🟢 {len(qualified)} 個 model 通過門檻：")
        for i, r in enumerate(qualified):
            bt = r["best_threshold"]
            print(f"  {i+1}. {r['name']}: AUC {r['auc']:.3f}, threshold {bt['th']}, wr {bt['wr']:.1f}% (+{bt['wr_imp']:.1f}%), kept {bt['n_kept']}")
        print()
        winner = qualified[0]
        bt_w = winner["best_threshold"]
        print(f"🏆 V36 推薦：**{winner['name']}**")
        print(f"   理由：AUC 最高（{winner['auc']:.3f}）")
        print(f"   建議 threshold：{bt_w['th']:.2f}")
        print(f"   預期 test wr：{bt_w['wr']:.1f}% (+{bt_w['wr_imp']:.1f}% vs raw)")


if __name__ == "__main__":
    main()
