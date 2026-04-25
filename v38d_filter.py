"""
V38d Filter — Production ML helper（paper trading Track C）

線上用法：
  from v38d_filter import V38dFilter
  v38d = V38dFilter()  # 啟動時 train 一次（用全期 89.90 + V38 過濾子集）
  decision = v38d.should_buy(ohlcv_df, market_close_history, today, kronos_pred_next, kronos_pred_5d)
  if decision["buy"]: ...

差別 vs V38（kronos_filter.py）：
  V38: only Kronos rule (next > 0.8 AND 5d > median)
  V38d: V38 rule AND ML proba > train median + 0.0
        (對於 V38 過濾後的子集，再用 ML 過濾 false positive)

訓練資料：
  - 89.90 全期 trades（cpu_replay）
  - 對每筆抽 19 features (metalabel_features.extract_features_for_trades)
  - 套 V38 rule 拿 V38-pass 子集
  - LogReg(C=1.0, L2) on V38-pass 子集

注意：
  - V38d 是 paper trading Track C，**不是上線策略**
  - 累積 ≥ 10 筆後 review，才決定要不要替換 V38
  - V38d 的 kept rate ~7%（比 V38 的 15.8% 還低），實盤交易頻率超低
"""
import os, sys, json, pickle
import urllib.request
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

KRONOS_NEXT_TH = 0.8
ML_C = 1.0
ML_TH_OFFSET = 0.0  # 動態 threshold = train proba median + offset


class V38dFilter:
    """V38d production filter（V38 rule + ML head）"""

    def __init__(self):
        self.scaler = None
        self.model = None
        self.train_proba_median = None
        self._train()

    def _train(self):
        """啟動時 train 一次（小樣本，秒級完成）"""
        import gpu_cupy_evolve as base
        from metalabel_features import extract_features_for_trades
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        print(f"[V38d] training ML head...")

        # Sanity CSV (V38 已跑過的 Kronos predictions)
        sanity_path = os.path.join(USER_SE, "kronos_sanity_results.csv")
        if not os.path.exists(sanity_path):
            raise RuntimeError(f"V38d 訓練需要 {sanity_path}，請先跑 sanity_test_kronos.py")
        df_sanity = pd.read_csv(sanity_path)

        # 跑 89.90
        GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
        r = urllib.request.urlopen(urllib.request.Request(
            f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
        d = json.loads(r.read())
        s = json.loads(d["files"]["best_strategy.json"]["content"])
        params = s.get("params", s)

        cache = pickle.load(open(os.path.join(USER_SE, "stock_data_cache.pkl"), "rb"))
        _lens = [len(v) for v in cache.values()]
        if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
        elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
        else: TARGET = 900
        data_dict = {k: v.tail(TARGET) for k, v in cache.items() if len(v) >= TARGET}
        pre = base.precompute(data_dict)

        all_trades = base.cpu_replay(pre, params)
        completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]

        # 19 features
        X19, _, keep_indices = extract_features_for_trades(pre, completed)
        trades_kept = [completed[i] for i in keep_indices]
        df_trades = pd.DataFrame([{
            "ticker": t.get("ticker"),
            "buy_date": t.get("buy_date"),
            "actual_return": float(t.get("return", 0)),
        } for t in trades_kept]).reset_index(drop=True)

        # 合併 Kronos predictions
        df_merged = df_trades.merge(
            df_sanity[["ticker", "buy_date", "pred_next_pct", "pred_5d_pct"]],
            on=["ticker", "buy_date"], how="inner"
        )
        trade_key = [(t.get("ticker"), t.get("buy_date")) for t in trades_kept]
        merged_keys = list(zip(df_merged["ticker"], df_merged["buy_date"]))
        aligned_idx = [trade_key.index(k) for k in merged_keys]
        X19_m = X19[aligned_idx]
        rets_m = df_merged["actual_return"].values
        p_next = df_merged["pred_next_pct"].values
        p_5d = df_merged["pred_5d_pct"].values

        # 套 V38 rule（path-internal 5d median 在 production 上換成全期 median）
        full_med_5d = float(np.median(p_5d))
        v38_pass = (p_next > KRONOS_NEXT_TH) & (p_5d > full_med_5d)

        if v38_pass.sum() < 8:
            raise RuntimeError(f"V38 子集只有 {v38_pass.sum()} 筆，無法訓練 V38d")

        X_train = X19_m[v38_pass]
        y_train = (rets_m[v38_pass] > 0).astype(int)
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            raise RuntimeError("V38 子集全贏或全輸，無法訓練")

        # Train
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(C=ML_C, max_iter=500, penalty="l2")
        self.model.fit(X_train_s, y_train)
        proba_tr = self.model.predict_proba(X_train_s)[:, 1]
        self.train_proba_median = float(np.median(proba_tr))

        # 也記錄 5d threshold（production 用）
        self.full_5d_median = full_med_5d

        print(f"[V38d] trained on {v38_pass.sum()} samples, "
              f"train wr {y_train.mean()*100:.1f}%, "
              f"proba median {self.train_proba_median:.3f}, "
              f"5d median proxy {full_med_5d:.2f}")

    def should_buy(self, ohlcv_df, market_close_history,
                   today, kronos_pred_next: float, kronos_pred_5d: float,
                   ticker: str = None, day_idx: int = None,
                   features_19d: np.ndarray = None) -> dict:
        """
        V38d 決策（兩層 gate）

        Args:
            kronos_pred_next, kronos_pred_5d: V38 已算過的 Kronos predictions
            features_19d: 該股當天的 19 feature vector（caller 提供）
                         如果 None，會跑 metalabel_features.extract_features_at

        Returns:
            dict {buy, reason, v38_pass, ml_proba, ...}
        """
        # Layer 1: V38 rule
        v38_pass = (kronos_pred_next > KRONOS_NEXT_TH) and (kronos_pred_5d > self.full_5d_median)
        if not v38_pass:
            return {
                "buy": False,
                "v38_pass": False,
                "ml_proba": None,
                "reason": f"V38 fail: next {kronos_pred_next:+.2f} > {KRONOS_NEXT_TH}? AND 5d {kronos_pred_5d:+.2f} > {self.full_5d_median:.2f}?",
            }

        # Layer 2: ML head
        if features_19d is None:
            return {
                "buy": False,
                "v38_pass": True,
                "ml_proba": None,
                "reason": "V38 pass but features_19d 沒提供 → 無法跑 ML",
            }

        x = self.scaler.transform(features_19d.reshape(1, -1))
        proba = float(self.model.predict_proba(x)[0, 1])
        ml_th = self.train_proba_median + ML_TH_OFFSET
        ml_pass = proba > ml_th

        buy = v38_pass and ml_pass
        if buy:
            reason = f"V38 pass AND ML proba {proba:.3f} > {ml_th:.3f}"
        else:
            reason = f"V38 pass but ML proba {proba:.3f} <= {ml_th:.3f}"

        return {
            "buy": buy,
            "v38_pass": v38_pass,
            "ml_proba": proba,
            "ml_threshold": ml_th,
            "reason": reason,
        }


# Singleton
_global_v38d = None
def get_v38d_filter():
    global _global_v38d
    if _global_v38d is None:
        _global_v38d = V38dFilter()
    return _global_v38d


if __name__ == "__main__":
    print("=" * 60)
    print("V38dFilter Smoke Test")
    print("=" * 60)
    v38d = V38dFilter()
    print(f"\nV38d trained ✅")
    print(f"  train proba median: {v38d.train_proba_median:.3f}")
    print(f"  ML threshold: {v38d.train_proba_median + ML_TH_OFFSET:.3f}")
    print(f"  5d median proxy: {v38d.full_5d_median:.2f}")
