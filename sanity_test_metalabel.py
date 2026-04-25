"""
Meta-Labeling Sanity Test — 驗證「ML 能從 raw features 過濾 89.90 false positive」

跟 V35 學乖（不直接燒 GPU），先 sanity 確認有 alpha 才實作 V36

判定標準：
  🟢 GREEN：test AUC > 0.55 AND filter 後 wr 提升 ≥ 5%
  🟡 YELLOW：AUC 0.52-0.55，邊際值得試
  🔴 RED：AUC < 0.52 → ML 分不出輸贏，放棄 Meta-Labeling，跳到分點主力

設計：
  1. 取 89.90 全期 133 筆 trades
  2. 對每筆 trade 取「買入當天的 raw features」
  3. Label = 1 (return > 0) 或 0 (return ≤ 0)
  4. Time-ordered split：80% train / 20% test（防 lookahead）
  5. 訓練 RandomForest（小樣本 robust）
  6. 看 test AUC + filter 後 wr 提升

學術參考：
  López de Prado (2018) Advances in Financial Machine Learning, Ch. 3
  Hudson & Thames Meta-Labeling Triple Barrier Method
"""
import os, sys, pickle, json
import urllib.request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path: sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base


def fetch_gist_strategy():
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    content = d["files"]["best_strategy.json"]["content"]
    strategy = json.loads(content)
    return strategy.get("params", strategy), strategy.get("score", "N/A")


def extract_features(pre, trades):
    """
    對每筆 trade，取「buy_date 當天該股的 raw indicator 值」當特徵

    Returns:
        X: shape (n_trades, n_features)
        y: shape (n_trades,) binary
        feature_names: list of str
        keep_indices: 哪些 trades 被保留（有效 feature）
    """
    tickers = pre["tickers"]
    dates = pre["dates"]
    nd = pre["n_days"]
    ticker_to_si = {t: i for i, t in enumerate(tickers)}
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}

    # 17 個 raw features（直接取 GPU 算好的）
    feature_arrs = {
        "rsi": pre["rsi"],
        "bb_pos": pre["bb_pos"],
        "vol_ratio": pre["vol_ratio"],
        "macd_hist": pre["macd_hist"],
        "macd_line": pre["macd_line"],
        "k_val": pre["k_val"],
        "williams_r": pre["williams_r"],
        "near_high": pre["near_high"],
        "adx": pre["adx"],
        "bias": pre["bias"],
        "atr_pct": pre["atr_pct"],
        "up_days": pre["up_days"],
        "week52_pos": pre["week52_pos"],
        "mom_accel": pre["mom_accel"],
        "vol_up_days": pre["vol_up_days"],
        "obv_rising": pre["obv_rising"],
        "new_high_60": pre["new_high_60"],
    }

    # market context: market_close 20d return, ma20/ma60 ratio
    close = pre["close"]
    market_close = close.mean(axis=0)  # shape (nd,)
    market_ma20 = np.zeros(nd, dtype=np.float32)
    market_ma60 = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        if i >= 20: market_ma20[i] = market_close[max(0, i-19):i+1].mean()
        if i >= 60: market_ma60[i] = market_close[max(0, i-59):i+1].mean()
    market_20d_return = np.zeros(nd, dtype=np.float32)
    market_20d_return[20:] = (market_close[20:] / market_close[:-20] - 1) * 100
    market_ma_ratio = np.where(market_ma60 > 0, market_ma20 / market_ma60, 1.0)

    feature_names = list(feature_arrs.keys()) + ["market_20d_return", "market_ma_ratio"]
    n_features = len(feature_names)

    rows_X = []
    rows_y = []
    keep_indices = []

    for i, trade in enumerate(trades):
        bd = trade.get("buy_date", "")
        ticker = trade.get("ticker", "")
        if bd not in date_to_day or ticker not in ticker_to_si:
            continue
        day = date_to_day[bd]
        si = ticker_to_si[ticker]

        x = []
        for fname in feature_arrs:
            arr = feature_arrs[fname]
            x.append(float(arr[si, day]))
        x.append(float(market_20d_return[day]))
        x.append(float(market_ma_ratio[day]))

        if any(np.isnan(v) or np.isinf(v) for v in x):
            continue

        ret = float(trade.get("return", 0))
        y = 1 if ret > 0 else 0

        rows_X.append(x)
        rows_y.append(y)
        keep_indices.append(i)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    return X, y, feature_names, keep_indices


def main():
    print("=" * 60)
    print("Meta-Labeling Sanity Test — 驗證 ML 能否過濾 89.90 false positive")
    print("=" * 60)
    print()

    # === Step 1: 讀策略 ===
    print("[1/5] 讀 GPU Gist 89.90 策略...")
    params, score = fetch_gist_strategy()
    print(f"  score = {score}")
    print()

    # === Step 2: precompute ===
    print("[2/5] 載入 cache + precompute...")
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
    print(f"  precompute done: {pre['n_stocks']} 檔 × {pre['n_days']} 天")
    print()

    # === Step 3: cpu_replay 拿 89.90 trades ===
    print("[3/5] cpu_replay 拿 89.90 全期 trades...")
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    print(f"  完成交易: {len(completed)} 筆")
    rets = [float(t.get("return", 0)) for t in completed]
    print(f"  整體: total={sum(rets):.1f}% wr={sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% avg={np.mean(rets):.1f}%")
    print()

    # === Step 4: 提取特徵 + label ===
    print("[4/5] 提取 raw features + Triple Barrier label...")
    X, y, feature_names, kept = extract_features(pre, completed)
    print(f"  保留 {len(kept)}/{len(completed)} 筆（NaN/Inf 過濾後）")
    print(f"  features: {len(feature_names)} 維")
    print(f"  label 分布: 贏 {y.sum()} / 輸 {len(y)-y.sum()} ({y.sum()/len(y)*100:.1f}% positive)")
    print()

    # === Step 5: Time-ordered split + RF ===
    print("[5/5] 訓練 RandomForest + 評估...")
    n = len(X)
    n_train = int(n * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"  train: {n_train} 筆 / test: {n - n_train} 筆")
    print(f"  train wr: {y_train.mean()*100:.1f}% / test wr: {y_test.mean()*100:.1f}%")
    print()

    # 訓練 RF（n_estimators 不要太多，小樣本）
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,           # 防 overfit
        min_samples_leaf=5,    # 防 overfit
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Test 預測
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    # 評估
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"=== Test Set 結果 ===")
    print(f"AUC = {auc:.4f}")
    print(f"分類報告：")
    print(classification_report(y_test, y_pred, target_names=["輸", "贏"], zero_division=0))
    print(f"Confusion matrix：")
    cm = confusion_matrix(y_test, y_pred)
    print(f"             預測輸  預測贏")
    print(f"  實際輸     {cm[0,0]:>5d}   {cm[0,1]:>5d}")
    print(f"  實際贏     {cm[1,0]:>5d}   {cm[1,1]:>5d}")
    print()

    # Feature importance
    print(f"=== Feature Importance（top 10）===")
    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances[:10]:
        print(f"  {name:<22s} {imp:.4f}")
    print()

    # === Filter 模擬：用不同 threshold 看 wr 提升 ===
    print(f"=== Filter 模擬（用不同 P 閾值過濾）===")
    print(f"{'閾值':<6s}{'保留':<8s}{'保留比例':<10s}{'wr':<8s}{'wr 提升':<10s}{'total':<10s}{'avg':<8s}")
    print("-" * 60)
    test_rets = np.array([float(completed[kept[i]].get("return", 0)) for i in range(n_train, n)])
    raw_test_wr = (test_rets > 0).mean() * 100
    raw_test_total = test_rets.sum()
    raw_test_avg = test_rets.mean()
    print(f"{'(原)':<6s}{n - n_train:<8d}{'100.0%':<10s}{raw_test_wr:<7.1f}%{'(基線)':<10s}{raw_test_total:<9.1f}%{raw_test_avg:<7.1f}%")

    best_threshold = None
    best_wr_improvement = 0.0

    for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        keep_mask = y_pred_proba >= th
        if keep_mask.sum() < 5:
            print(f"{th:<6.2f}{int(keep_mask.sum()):<8d}{keep_mask.sum()/(n-n_train)*100:<9.1f}%{'N/A':<8s}{'樣本不足':<10s}")
            continue
        kept_rets = test_rets[keep_mask]
        kept_wr = (kept_rets > 0).mean() * 100
        kept_total = kept_rets.sum()
        kept_avg = kept_rets.mean()
        wr_imp = kept_wr - raw_test_wr
        print(f"{th:<6.2f}{int(keep_mask.sum()):<8d}{keep_mask.sum()/(n-n_train)*100:<9.1f}%{kept_wr:<7.1f}%{wr_imp:+8.1f}% {kept_total:<9.1f}%{kept_avg:<7.1f}%")
        if wr_imp > best_wr_improvement and keep_mask.sum() >= 10:
            best_wr_improvement = wr_imp
            best_threshold = th

    print()

    # === Go/no-go 裁決 ===
    print("=" * 60)
    print("📋 Meta-Labeling Sanity 裁決")
    print("=" * 60)

    if auc > 0.55 and best_wr_improvement >= 5:
        print(f"🟢 GREEN — Meta-Labeling 有真 alpha")
        print(f"   AUC = {auc:.3f} > 0.55")
        print(f"   最佳 threshold {best_threshold} → wr 提升 {best_wr_improvement:+.1f}%")
        print(f"   → 值得實作 V36 Meta-Labeling")
    elif auc > 0.52:
        print(f"🟡 YELLOW — 邊際值得試")
        print(f"   AUC = {auc:.3f}（門檻 0.55）")
        print(f"   最佳 wr 提升 {best_wr_improvement:+.1f}%（門檻 +5%）")
        print(f"   → 可選擇實作或跳到分點主力")
    else:
        print(f"🔴 RED — Meta-Labeling 無 alpha")
        print(f"   AUC = {auc:.3f} < 0.52（隨機 = 0.5）")
        print(f"   → ML 從這些 features 分不出輸贏")
        print(f"   → 跳過 Meta-Labeling，直接做分點主力（plan_b 方向 2）")
    print()


if __name__ == "__main__":
    main()
