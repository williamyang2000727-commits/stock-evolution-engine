"""
Meta-Labeling V36 共用 feature extraction
sanity_test_metalabel.py + train_metalabel_v36.py + 線上 predict 都用這個

19 維特徵：
  17 個 stock 指標（從 base.precompute 取 raw）
  2 個 market context（market_20d_return, market_ma_ratio）
"""
import numpy as np


FEATURE_NAMES = [
    # 17 個 stock 指標
    "rsi",
    "bb_pos",
    "vol_ratio",
    "macd_hist",
    "macd_line",
    "k_val",
    "williams_r",
    "near_high",
    "adx",
    "bias",
    "atr_pct",
    "up_days",
    "week52_pos",
    "mom_accel",
    "vol_up_days",
    "obv_rising",
    "new_high_60",
    # 2 個 market context（V36 ML 抓到 market_20d_return 是 #1 重要）
    "market_20d_return",
    "market_ma_ratio",
]
N_FEATURES = len(FEATURE_NAMES)


def compute_market_context(close: np.ndarray):
    """
    從 close (n_stocks, n_days) 算市場 context
    Returns:
        market_20d_return: shape (n_days,)，大盤 20 日報酬 %
        market_ma_ratio: shape (n_days,)，大盤 MA20/MA60 比例
    """
    n_days = close.shape[1]
    market_close = close.mean(axis=0)
    market_ma20 = np.zeros(n_days, dtype=np.float32)
    market_ma60 = np.zeros(n_days, dtype=np.float32)
    for i in range(n_days):
        if i >= 20:
            market_ma20[i] = market_close[max(0, i - 19):i + 1].mean()
        if i >= 60:
            market_ma60[i] = market_close[max(0, i - 59):i + 1].mean()
    market_20d_return = np.zeros(n_days, dtype=np.float32)
    market_20d_return[20:] = (market_close[20:] / np.where(market_close[:-20] > 0, market_close[:-20], 1.0) - 1) * 100
    market_ma_ratio = np.where(market_ma60 > 0, market_ma20 / market_ma60, 1.0).astype(np.float32)
    return market_20d_return, market_ma_ratio


def extract_features_at(pre, ticker: str, day: int):
    """
    抽單筆「ticker 在 day 那天」的 19 維 feature
    用於線上 predict（單一候選股）

    Returns:
        feature vector shape (19,) 或 None（若無效）
    """
    tickers = pre["tickers"]
    if ticker not in tickers:
        return None
    si = list(tickers).index(ticker) if not isinstance(tickers, list) else tickers.index(ticker)
    nd = pre["n_days"]
    if day < 0 or day >= nd:
        return None

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

    market_20d_return, market_ma_ratio = compute_market_context(pre["close"])

    x = []
    for fname in FEATURE_NAMES[:-2]:
        x.append(float(feature_arrs[fname][si, day]))
    x.append(float(market_20d_return[day]))
    x.append(float(market_ma_ratio[day]))

    if any(np.isnan(v) or np.isinf(v) for v in x):
        return None
    return np.array(x, dtype=np.float32)


def extract_features_for_trades(pre, trades):
    """
    對 trades list 中每筆抽 19 維 feature
    用於訓練 + CPCV 評估

    Returns:
        X: shape (n_kept, 19) feature matrix
        y: shape (n_kept,) binary label (1=贏 0=輸)
        keep_indices: 哪些 trades 被保留（X[i] 對應 trades[keep_indices[i]]）
    """
    tickers = pre["tickers"]
    dates = pre["dates"]
    nd = pre["n_days"]
    ticker_to_si = {t: i for i, t in enumerate(tickers)}
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}

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

    market_20d_return, market_ma_ratio = compute_market_context(pre["close"])

    rows_X, rows_y, keep_indices = [], [], []
    for i, trade in enumerate(trades):
        bd = trade.get("buy_date", "")
        ticker = trade.get("ticker", "")
        if bd not in date_to_day or ticker not in ticker_to_si:
            continue
        day = date_to_day[bd]
        si = ticker_to_si[ticker]

        x = []
        for fname in FEATURE_NAMES[:-2]:
            x.append(float(feature_arrs[fname][si, day]))
        x.append(float(market_20d_return[day]))
        x.append(float(market_ma_ratio[day]))

        if any(np.isnan(v) or np.isinf(v) for v in x):
            continue

        ret = float(trade.get("return", 0))
        rows_X.append(x)
        rows_y.append(1 if ret > 0 else 0)
        keep_indices.append(i)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    return X, y, keep_indices
