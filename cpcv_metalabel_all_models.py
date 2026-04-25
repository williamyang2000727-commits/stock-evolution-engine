"""
全 8 model CPCV leave-one-path-out 評估

問題：sanity 的 80/20 split AUC 排名跟 CPCV 完全不同
  GB sanity 0.746 → CPCV 0.510（落差 0.236）
  LightGBM sanity 0.683 → CPCV 0.536（落差 0.147）

假設：boosting model 在小樣本下 overfit，linear/simple model 反而 CPCV 穩
驗證：跑全 8 model CPCV，看誰真的穩

如果還是沒一個過真突破門檻 → Meta-Labeling 確認失敗，跳分點主力
"""
import os, sys, pickle, json, warnings
import urllib.request
import numpy as np
from itertools import combinations

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]


def fetch_gist_strategy():
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
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0}
    rets = np.array(rets)
    return {
        "n": len(rets),
        "wr": float((rets > 0).mean() * 100),
        "total": float(rets.sum()),
        "avg": float(rets.mean()),
    }


def build_models():
    """8 個 model 全配置（部分需 scaling）"""
    models = []

    # 1. RF
    models.append(("RF",
        RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        False))

    # 2. ExtraTrees
    models.append(("ExtraTrees",
        ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        False))

    # 3. GB
    models.append(("GB",
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
        pass

    # 5. XGBoost
    try:
        import xgboost as xgb
        models.append(("XGBoost",
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              min_child_weight=5, random_state=42, n_jobs=-1, eval_metric="logloss"),
            False))
    except ImportError:
        pass

    # 6. LogReg（線性，需 scale）
    models.append(("LogReg",
        Pipeline([("sc", StandardScaler()),
                 ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42))]),
        False))  # Pipeline 內含 scaler

    # 7. SVM-RBF（需 scale）
    models.append(("SVM-RBF",
        Pipeline([("sc", StandardScaler()),
                 ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))]),
        False))

    # 8. MLP（需 scale）
    models.append(("MLP",
        Pipeline([("sc", StandardScaler()),
                 ("mlp", MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500,
                                      random_state=42, early_stopping=True))]),
        False))

    return models


def cpcv_eval_one_model(name, model_factory, X_all, y_all, kept_indices, completed,
                         groups, dates, thresholds=None):
    """
    對 one model 跑全 15 path leave-one-out
    對每個 threshold 都算 wr_imp，回傳每個 threshold 的 summary
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}

    trade_buy_days = np.array([
        date_to_day.get(completed[kt_i].get("buy_date", ""), -1)
        for kt_i in kept_indices
    ])

    # Per-path results: {threshold: list of {raw, filtered, wr_imp}}
    per_path_per_th = {th: [] for th in thresholds}
    aucs = []

    for path_idx, test_gi in enumerate(test_combos):
        in_test = np.zeros(len(X_all), dtype=bool)
        for s, e in [groups[g] for g in test_gi]:
            in_test |= (trade_buy_days >= s) & (trade_buy_days < e)
        in_train = ~in_test

        if in_test.sum() < 5 or in_train.sum() < 30:
            continue

        # Fresh model 每次 path 都重訓
        try:
            from copy import deepcopy
            model = deepcopy(model_factory)
            model.fit(X_all[in_train], y_all[in_train])
        except Exception:
            continue

        # Predict on test
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_all[in_test])[:, 1]
            else:
                # Fallback (SVM no probability)
                y_pred_proba = model.predict(X_all[in_test]).astype(float)
            test_auc = roc_auc_score(y_all[in_test], y_pred_proba)
            aucs.append(test_auc)
        except Exception:
            continue

        test_idx = np.where(in_test)[0]
        test_rets = np.array([float(completed[kept_indices[i]].get("return", 0)) for i in test_idx])
        raw_stats = stats_of(test_rets)

        # 對每個 threshold 算
        for th in thresholds:
            keep_mask = y_pred_proba >= th
            filt_rets = test_rets[keep_mask]
            filt_stats = stats_of(filt_rets)
            wr_imp = filt_stats["wr"] - raw_stats["wr"] if filt_stats["n"] >= 5 else None
            per_path_per_th[th].append({
                "path_idx": path_idx,
                "test_groups": list(test_gi),
                "raw": raw_stats,
                "filtered": filt_stats,
                "wr_imp": wr_imp,
                "kept_pct": filt_stats["n"] / raw_stats["n"] * 100 if raw_stats["n"] > 0 else 0,
            })

    # 對每個 threshold 算 summary
    summaries_per_th = {}
    for th in thresholds:
        valid = [p for p in per_path_per_th[th] if p["wr_imp"] is not None and p["filtered"]["n"] >= 5]
        if not valid:
            summaries_per_th[th] = None
            continue

        wr_imps = np.array([p["wr_imp"] for p in valid])
        kept_pcts = np.array([p["kept_pct"] for p in valid])
        n_breakthrough = int((wr_imps >= 5).sum())

        summaries_per_th[th] = {
            "n_valid": len(valid),
            "n_breakthrough": n_breakthrough,  # ≥+5% wr
            "n_positive": int((wr_imps > 0).sum()),  # > 0
            "mean_wr_imp": float(wr_imps.mean()),
            "median_wr_imp": float(np.median(wr_imps)),
            "p25_wr_imp": float(np.percentile(wr_imps, 25)),
            "min_wr_imp": float(wr_imps.min()),
            "max_wr_imp": float(wr_imps.max()),
            "mean_kept_pct": float(kept_pcts.mean()),
        }

    mean_auc = float(np.mean(aucs)) if aucs else 0.5

    return {
        "name": name,
        "mean_auc": mean_auc,
        "n_paths_tested": len(aucs),
        "summaries": summaries_per_th,
        "per_path_data": per_path_per_th,  # 完整資料給後續分析
    }


def main():
    print("=" * 75)
    print("全 8 Model CPCV Leave-One-Path-Out 評估")
    print("=" * 75)

    # 載入資料
    print("\n[1/3] 載入...")
    params, score = fetch_gist_strategy()
    print(f"  89.90 score = {score}")

    cache_path = os.path.join(_USER_SE, "stock_data_cache.pkl")
    raw = pickle.load(open(cache_path, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data)

    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    X_all, y_all, kept = extract_features_for_trades(pre, completed)
    print(f"  保留 {len(X_all)} 筆 trades，{X_all.shape[1]} features")

    n_days = pre["n_days"]
    dates = pre["dates"]
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    print(f"  CPCV groups: {N_GROUPS}, k={K_TEST}, total paths = {len(list(combinations(range(N_GROUPS), K_TEST)))}")

    # 跑全 8 model
    print(f"\n[2/3] 跑 8 個 model × 5 個 threshold × 15 path...")
    print(f"  總共 8 × 5 × 15 = 600 次模型訓練+評估")

    models = build_models()
    print(f"  實際載入 {len(models)} model")

    all_results = []
    for i, (name, model_factory, _) in enumerate(models):
        print(f"\n  [{i+1}/{len(models)}] {name}...")
        result = cpcv_eval_one_model(
            name, model_factory, X_all, y_all, kept, completed,
            groups, dates, thresholds=THRESHOLDS
        )
        all_results.append(result)

    # 結果分析
    print(f"\n[3/3] 結果分析...")
    print("=" * 75)
    print(f"\n{'Model':<14s}{'AUC':<7s}{'最佳 th':<8s}{'≥+5% paths':<12s}{'mean wr↑':<11s}{'p25 wr↑':<10s}{'kept%':<7s}")
    print("-" * 75)

    # 對每個 model，找最佳 threshold（mean_wr_imp 最高的）
    best_per_model = []
    for r in all_results:
        if not r["summaries"]:
            continue
        valid_th = [(th, s) for th, s in r["summaries"].items() if s is not None]
        if not valid_th:
            continue
        # 排序：先看 n_breakthrough，再看 mean_wr_imp
        valid_th.sort(key=lambda x: (x[1]["n_breakthrough"], x[1]["mean_wr_imp"]), reverse=True)
        best_th, best_s = valid_th[0]
        best_per_model.append({
            "name": r["name"],
            "mean_auc": r["mean_auc"],
            "best_th": best_th,
            "summary": best_s,
        })

    # 排序：先看 n_breakthrough，再看 mean_wr_imp
    best_per_model.sort(key=lambda x: (x["summary"]["n_breakthrough"], x["summary"]["mean_wr_imp"]), reverse=True)

    for b in best_per_model:
        s = b["summary"]
        print(f"{b['name']:<14s}{b['mean_auc']:<7.3f}{b['best_th']:<8.2f}"
              f"{s['n_breakthrough']:<3d}/15      "
              f"{s['mean_wr_imp']:+7.2f}%   "
              f"{s['p25_wr_imp']:+6.2f}%   "
              f"{s['mean_kept_pct']:5.1f}%")

    # 詳細：所有 model 在所有 threshold 的 mean_wr_imp 表格
    print(f"\n{'='*75}")
    print(f"完整表格：每個 model 在不同 threshold 下的 mean wr 改善")
    print(f"{'='*75}")
    print(f"{'Model':<14s}", end="")
    for th in THRESHOLDS:
        print(f"th={th:<6.2f}", end="  ")
    print()
    print("-" * 75)

    for r in all_results:
        if not r["summaries"]:
            continue
        print(f"{r['name']:<14s}", end="")
        for th in THRESHOLDS:
            s = r["summaries"].get(th)
            if s is None:
                print(f"{'N/A':<8s}", end="  ")
            else:
                print(f"{s['mean_wr_imp']:+5.2f}/{s['n_breakthrough']:>2d} ", end="  ")
        print()
    print()
    print("（格式：mean_wr_imp / n_breakthrough，例 +1.50/3 = 平均 +1.5% wr 改善，3/15 path 達 +5%）")

    # 結論
    print(f"\n{'='*75}")
    print("📋 結論")
    print(f"{'='*75}")
    if not best_per_model:
        print("🔴 所有 model 完全失敗")
        print("   → Meta-Labeling 死路，跳分點主力")
        return

    winner = best_per_model[0]
    s = winner["summary"]
    breakthrough = (s["n_breakthrough"] >= 12 and s["mean_wr_imp"] >= 5 and s["p25_wr_imp"] >= 0)
    if breakthrough:
        print(f"🟢 {winner['name']} 通過真突破門檻！")
        print(f"   threshold {winner['best_th']}, n_breakthrough {s['n_breakthrough']}/15")
        print(f"   mean wr_imp {s['mean_wr_imp']:.2f}%, p25 {s['p25_wr_imp']:.2f}%")
    else:
        print(f"🔴 沒有 model 通過真突破門檻")
        print(f"   最強：{winner['name']} threshold {winner['best_th']}")
        print(f"   - n_breakthrough {s['n_breakthrough']}/15（門檻 12）")
        print(f"   - mean wr_imp {s['mean_wr_imp']:+.2f}%（門檻 +5%）")
        print(f"   - p25 wr_imp {s['p25_wr_imp']:+.2f}%（門檻 ≥ 0%）")
        print()
        print(f"   🎯 結論：Meta-Labeling 在當前 19 features 下無 alpha")
        print(f"   🎯 下一步：跳到 plan_b 方向 4 = 分點券商主力買超")

    # 存 JSON
    json_path = os.path.join(_USER_SE, "metalabel_v36_all_models_cpcv.json")
    json_safe = []
    for r in all_results:
        json_safe.append({
            "name": r["name"],
            "mean_auc": r["mean_auc"],
            "n_paths_tested": r["n_paths_tested"],
            "summaries": {str(k): v for k, v in r["summaries"].items()},
        })
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\nJSON 結果存到 {json_path}")


if __name__ == "__main__":
    main()
