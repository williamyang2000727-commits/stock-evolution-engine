"""
V38 Kronos Filter — Production helper
用法：
  from kronos_filter import KronosFilter

  kf = KronosFilter()  # 載入 model 一次
  decision = kf.should_buy(stock_id="2330.TW", today=pd.Timestamp("2026-04-25"))
  if decision["buy"]:
      # 下單
  else:
      # skip 該天

CPCV 驗證：ensemble_next_top5d threshold=0.8
  - pred_next_pct > 0.8（次日預測漲 > 0.8%）
  - pred_5d_pct > path 5d median（5d 在前 50%）
  - 兩個都符合才買
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
KRONOS_DIR = os.path.join(USER_SE, "Kronos")
if KRONOS_DIR not in sys.path: sys.path.insert(0, KRONOS_DIR)

# Kronos params
LOOKBACK = 60
PRED_LEN = 5
NEXT_DAY_THRESHOLD = 0.8  # CPCV 最強 threshold

# 5d median 在線上沒有 path 概念，用「最近 100 天的 5d return median」當基準
# CPCV 的 path 是 ~240 天 → 線上用 60-90 天 rolling median 作 proxy
ROLLING_MEDIAN_DAYS = 60


class KronosFilter:
    """V38 Kronos ensemble filter，CPCV th=0.8 驗證"""

    def __init__(self, model_size: str = "small", device: str = None):
        from model import Kronos, KronosTokenizer, KronosPredictor
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"[KronosFilter] 載入 Kronos-{model_size} on {device}...")
        self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        self.model = Kronos.from_pretrained(f"NeoQuasar/Kronos-{model_size}")
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=device, max_context=512)
        self.device = device
        self.model_size = model_size
        print(f"[KronosFilter] 載入完成")

    def predict(self, ohlcv_df: pd.DataFrame, today: pd.Timestamp = None):
        """
        對一檔股票預測未來 5 天

        Args:
            ohlcv_df: pd.DataFrame index=date, columns=['Open','High','Low','Close','Volume']
                     至少 LOOKBACK (60) 天歷史
            today: 預測基準日（預設最後一天）

        Returns:
            dict {
                'pred_next_pct': float,    # 次日預測漲跌 %
                'pred_5d_pct': float,      # 5 日預測漲跌 %
                'today_close': float,
                'pred_close_seq': list,    # 5 天預測 close
                'success': bool,
                'reason': str (如果 success=False)
            }
        """
        df = ohlcv_df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 確認 lowercase
        df.columns = [c.lower() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for c in required:
            if c not in df.columns:
                return {"success": False, "reason": f"missing column {c}"}

        # 找 today index
        if today is None:
            d_idx = len(df) - 1
        else:
            today = pd.Timestamp(today)
            mask = df.index <= today
            if mask.sum() < LOOKBACK:
                return {"success": False, "reason": f"history < {LOOKBACK} days"}
            d_idx = mask.sum() - 1

        if d_idx < LOOKBACK - 1:
            return {"success": False, "reason": f"history insufficient (have {d_idx+1}, need {LOOKBACK})"}

        # 抽 60 K 線
        x_df = df.iloc[d_idx - LOOKBACK + 1:d_idx + 1][required].reset_index(drop=True)
        x_ts = df.index[d_idx - LOOKBACK + 1:d_idx + 1].to_series().reset_index(drop=True)

        # Future timestamps（5 個工作日）
        last_date = df.index[d_idx]
        y_ts = pd.Series(pd.date_range(last_date + pd.Timedelta(days=1), periods=PRED_LEN, freq="B"))

        try:
            pred = self.predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False,
            )
        except Exception as e:
            return {"success": False, "reason": f"predict error: {e}"}

        today_close = float(x_df["close"].iloc[-1])
        pred_next = float(pred["close"].iloc[0])
        pred_5d = float(pred["close"].iloc[-1])

        return {
            "success": True,
            "pred_next_pct": (pred_next / today_close - 1) * 100,
            "pred_5d_pct": (pred_5d / today_close - 1) * 100,
            "today_close": today_close,
            "pred_close_seq": pred["close"].tolist(),
        }

    def get_5d_median_proxy(self, market_close_history: np.ndarray, n_days: int = ROLLING_MEDIAN_DAYS):
        """
        線上用「最近 N 天大盤 5d return median」當 5d threshold 的 proxy
        因為 CPCV 用 path-internal median，線上沒有 path 概念

        Args:
            market_close_history: np.ndarray 大盤最近 N+5 天 close
            n_days: 取最近幾天當 baseline

        Returns:
            float: 大盤 5d return median %
        """
        if market_close_history is None or len(market_close_history) < n_days + 5:
            return 0.0
        arr = np.asarray(market_close_history, dtype=np.float64)
        recent = arr[-(n_days + 5):].copy()
        # 處理 NaN / 0 / Inf：forward fill
        if np.any(np.isnan(recent)) or np.any(recent <= 0):
            valid_mask = ~np.isnan(recent) & (recent > 0)
            if valid_mask.sum() < n_days // 2:
                return 0.0
            last_valid = float(recent[valid_mask][0])
            for i in range(len(recent)):
                if np.isnan(recent[i]) or recent[i] <= 0:
                    recent[i] = last_valid
                else:
                    last_valid = float(recent[i])
        returns_5d = (recent[5:] / recent[:-5] - 1) * 100
        returns_5d = returns_5d[np.isfinite(returns_5d)]
        if len(returns_5d) == 0:
            return 0.0
        return float(np.median(returns_5d))

    def should_buy(self, ohlcv_df: pd.DataFrame, market_close_history: np.ndarray = None,
                  today: pd.Timestamp = None, threshold_next: float = NEXT_DAY_THRESHOLD,
                  threshold_5d: float = None) -> dict:
        """
        V38 ensemble decision

        Args:
            ohlcv_df: 該股 OHLCV history
            market_close_history: 大盤 close array (用來算 5d median proxy)
            today: 決策日
            threshold_next: next-day pred 門檻（CPCV 最強 0.8）
            threshold_5d: 5d pred 門檻（None = 用 market_close 算 proxy）

        Returns:
            dict {
                'buy': bool,
                'pred_next_pct': float,
                'pred_5d_pct': float,
                'threshold_next': float,
                'threshold_5d': float,
                'reason': str,
            }
        """
        pred = self.predict(ohlcv_df, today)
        if not pred["success"]:
            return {"buy": False, "reason": f"predict failed: {pred.get('reason')}", **pred}

        # 5d threshold
        if threshold_5d is None:
            if market_close_history is not None:
                threshold_5d = self.get_5d_median_proxy(market_close_history)
            else:
                threshold_5d = 0.0  # 保守 fallback

        # Ensemble check
        cond_next = pred["pred_next_pct"] > threshold_next
        cond_5d = pred["pred_5d_pct"] > threshold_5d

        buy = cond_next and cond_5d

        if buy:
            reason = f"ensemble pass: next {pred['pred_next_pct']:+.2f}% > {threshold_next} AND 5d {pred['pred_5d_pct']:+.2f}% > {threshold_5d:.2f}"
        elif not cond_next:
            reason = f"next-day fail: pred {pred['pred_next_pct']:+.2f}% <= {threshold_next}"
        else:
            reason = f"5d fail: pred {pred['pred_5d_pct']:+.2f}% <= {threshold_5d:.2f}"

        return {
            "buy": buy,
            "pred_next_pct": pred["pred_next_pct"],
            "pred_5d_pct": pred["pred_5d_pct"],
            "threshold_next": threshold_next,
            "threshold_5d": threshold_5d,
            "today_close": pred["today_close"],
            "reason": reason,
        }


# Convenience function 線上一行用
_global_filter = None
def get_filter(model_size: str = "small"):
    """Singleton instance（避免重複載 model）"""
    global _global_filter
    if _global_filter is None:
        _global_filter = KronosFilter(model_size=model_size)
    return _global_filter


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("KronosFilter Smoke Test")
    print("=" * 60)

    cache_path = os.path.join(USER_SE, "stock_data_cache.pkl")
    cache = pickle.load(open(cache_path, "rb"))

    # 測 1101.TW
    ticker = "1101.TW"
    if ticker not in cache:
        print(f"❌ {ticker} 不在 cache")
        sys.exit(1)

    df = cache[ticker].tail(80)
    print(f"\n用 {ticker} 最近 80 天測試（最後一天 {df.index[-1].date()}）")

    # 用所有股票平均當大盤 close（簡化）
    all_closes = []
    for k, v in cache.items():
        if len(v) >= 100:
            all_closes.append(v["Close"].tail(100).values)
    if all_closes:
        all_closes = np.array(all_closes)
        market_close = np.nanmean(all_closes, axis=0)  # 處理 NaN
        print(f"  大盤 proxy: {len(market_close)} 天 (NaN: {int(np.isnan(market_close).sum())})")
    else:
        market_close = None

    kf = KronosFilter()
    decision = kf.should_buy(df, market_close_history=market_close)

    print(f"\n=== 決策 ===")
    print(f"  buy = {decision['buy']}")
    print(f"  pred next-day = {decision['pred_next_pct']:+.2f}% (threshold {decision['threshold_next']})")
    print(f"  pred 5-day    = {decision['pred_5d_pct']:+.2f}% (threshold {decision['threshold_5d']:.2f})")
    print(f"  reason: {decision['reason']}")
