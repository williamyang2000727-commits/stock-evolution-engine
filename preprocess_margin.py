"""
V34 preprocess — margin 資料 → margin_tensor.npy
讀取 margin_data_full.pkl + stock_data_cache.pkl，對齊日期後計算 5 個指標：
  1. margin_heat:    融資今日餘額 / 融資限額         (0-1, 散戶槓桿使用率)
  2. margin_accel:   融資 5 日 % 變化               (-200~200, 加碼加速)
  3. short_ratio:    融券餘額 / 融資餘額 * 100      (0-500, 空方對多方比)
  4. offset_rate:    資券互抵 / 融資買進 * 100      (0-200, 當沖活躍度)
  5. margin_diverge: margin_5d_pct - price_5d_pct  (-200~200, 融資-價格發散)

輸出：margin_tensor.npy shape (days × stocks × 5)
     margin_tensor_meta.pkl: tickers list, indicator names

用法：C:\\stock-evolution> python preprocess_margin.py
"""
import os, pickle
import numpy as np
import pandas as pd

CACHE_PATH  = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
MARGIN_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_data_full.pkl")
OUT_TENSOR  = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_tensor.npy")
OUT_META    = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_tensor_meta.pkl")
TARGET_DAYS = 1500

NUMERIC_COLS = [
    "MarginPurchaseBuy", "MarginPurchaseSell",
    "MarginPurchaseTodayBalance", "MarginPurchaseYesterdayBalance",
    "MarginPurchaseLimit", "MarginPurchaseCashRepayment",
    "ShortSaleBuy", "ShortSaleSell",
    "ShortSaleTodayBalance", "ShortSaleYesterdayBalance",
    "ShortSaleLimit", "ShortSaleCashRepayment",
    "OffsetLoanAndShort",
]


def compute_indicators(margin_df: pd.DataFrame, cache_df: pd.DataFrame, reference_dates=None) -> np.ndarray:
    """
    margin_df: raw FinMind DataFrame
    cache_df:  OHLCV DataFrame (tz-aware DatetimeIndex)
    reference_dates: 可選，全局統一的日期軸（避免每檔 tail 對齊不同日期造成 tensor 日期錯位）
    returns: np.ndarray shape (TARGET_DAYS, 5)
    """
    if reference_dates is not None:
        # BUG #12 修正（2026-04-25）：用全局 reference_dates 對齊，每檔 tensor 的日期一致
        cache_dates = reference_dates
        # 把該檔的 cache_df reindex 到 reference_dates 對齊（用於算 price5d 發散）
        _cache_idx = cache_df.index
        if _cache_idx.tz is not None:
            _cache_naive = cache_df.copy()
            _cache_naive.index = _cache_idx.tz_convert("Asia/Taipei").tz_localize(None).normalize()
        else:
            _cache_naive = cache_df.copy()
            _cache_naive.index = _cache_idx.normalize()
        cache_tail = _cache_naive.reindex(reference_dates, method="ffill")
    else:
        # Legacy fallback：取該檔最後 1500 天
        cache_tail = cache_df.tail(TARGET_DAYS).copy()
        cache_idx = cache_tail.index
        if cache_idx.tz is not None:
            cache_dates = cache_idx.tz_convert("Asia/Taipei").tz_localize(None).normalize()
        else:
            cache_dates = cache_idx.normalize()

    # margin df: date string → datetime, set as index
    m = margin_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date").sort_index()
    for c in NUMERIC_COLS:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)
        else:
            m[c] = 0

    # Reindex to cache dates, ffill 讓停牌/無 margin 活動日沿用前值
    aligned = m[NUMERIC_COLS].reindex(cache_dates, method="ffill").fillna(0)

    # ── 指標 1: margin_heat = balance / limit ──
    limit = aligned["MarginPurchaseLimit"].replace(0, np.nan)
    heat = (aligned["MarginPurchaseTodayBalance"] / limit).fillna(0).clip(0, 1)

    # ── 指標 2: margin_accel = 5d % change ──
    bal = aligned["MarginPurchaseTodayBalance"]
    bal5 = bal.shift(5).replace(0, np.nan)
    accel = ((bal - bal5) / bal5 * 100).fillna(0).clip(-200, 200)

    # ── 指標 3: short_ratio = short / margin * 100 ──
    margin_bal = aligned["MarginPurchaseTodayBalance"].replace(0, np.nan)
    short_ratio = (aligned["ShortSaleTodayBalance"] / margin_bal * 100).fillna(0).clip(0, 500)

    # ── 指標 4: offset_rate = offset / margin_buy * 100 ──
    mbuy = aligned["MarginPurchaseBuy"].replace(0, np.nan)
    offset = (aligned["OffsetLoanAndShort"] / mbuy * 100).fillna(0).clip(0, 200)

    # ── 指標 5: margin_diverge = margin_5d_pct - price_5d_pct ──
    close = cache_tail["Close"].values
    price5 = np.concatenate([np.zeros(5), close[5:] / np.where(close[:-5] > 0, close[:-5], np.nan) - 1]) * 100
    price5 = np.nan_to_num(price5, nan=0.0)
    diverge = (accel.values - price5).clip(-200, 200)

    out = np.stack([
        heat.values.astype(np.float32),
        accel.values.astype(np.float32),
        short_ratio.values.astype(np.float32),
        offset.values.astype(np.float32),
        diverge.astype(np.float32),
    ], axis=1)
    return out  # (TARGET_DAYS, 5)


def main():
    print("=== V34 preprocess: margin → tensor ===")
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(CACHE_PATH)
    if not os.path.exists(MARGIN_PATH):
        raise FileNotFoundError(f"{MARGIN_PATH}\n→ 先跑 fetch_margin_history.py")

    cache = pickle.load(open(CACHE_PATH, "rb"))
    margin = pickle.load(open(MARGIN_PATH, "rb"))
    print(f"cache:  {len(cache)} stocks")
    print(f"margin: {len(margin)} stocks")

    # 合格股票 = cache 中有 ≥1500 天 且 margin 中也有資料
    qualified = []
    skipped_short = 0
    skipped_no_margin = 0
    for key, df in cache.items():
        if len(df) < TARGET_DAYS:
            skipped_short += 1
            continue
        tk = key.split(".")[0]
        if tk not in margin or len(margin[tk]) == 0:
            skipped_no_margin += 1
            continue
        qualified.append(key)
    print(f"合格 (>=1500 天 & 有 margin): {len(qualified)} 檔")
    print(f"  跳過 (<1500 天): {skipped_short}")
    print(f"  跳過 (無 margin 資料): {skipped_no_margin}")

    # BUG #12 修正（2026-04-25）：建立全局 reference_dates，讓所有 stock 的 tensor 對齊同一日期軸
    # 找合格股票裡最長的 cache tail 當基準（覆蓋所有股票的日期）
    _ref_df = cache[qualified[0]].tail(TARGET_DAYS)
    _ref_idx = _ref_df.index
    if _ref_idx.tz is not None:
        reference_dates = _ref_idx.tz_convert("Asia/Taipei").tz_localize(None).normalize()
    else:
        reference_dates = _ref_idx.normalize()
    # 找所有合格股票的 tail dates union → 取最完整的作為 ref
    for key in qualified[:50]:  # 取前 50 檔嘗試找最完整的
        _df = cache[key].tail(TARGET_DAYS)
        if len(_df) == TARGET_DAYS:
            _idx = _df.index
            _d = _idx.tz_convert("Asia/Taipei").tz_localize(None).normalize() if _idx.tz else _idx.normalize()
            if _d[-1] >= reference_dates[-1]:
                reference_dates = _d
    print(f"reference_dates: {reference_dates[0].date()} ~ {reference_dates[-1].date()} ({len(reference_dates)} 天)")

    # Build tensor
    tensor = np.zeros((TARGET_DAYS, len(qualified), 5), dtype=np.float32)
    fail = []
    for i, key in enumerate(qualified):
        tk = key.split(".")[0]
        try:
            # 用 reference_dates 對齊（每檔 tensor 日期軸完全一致）
            tensor[:, i, :] = compute_indicators(margin[tk], cache[key], reference_dates=reference_dates)
        except Exception as e:
            fail.append((key, str(e)))
            # 失敗就 0 填充，不中斷
        if (i + 1) % 200 == 0:
            print(f"  處理 {i+1}/{len(qualified)}")

    if fail:
        print(f"! {len(fail)} 檔失敗（留 0）：{fail[:5]}")

    # Save
    np.save(OUT_TENSOR, tensor)
    # BUG #3 修正：meta 加 dates 欄位，讓 V34 precompute 能驗證時序方向和日期對齊
    _dates_iso = [d.date().isoformat() for d in reference_dates]
    pickle.dump({
        "shape": tensor.shape,
        "tickers": qualified,
        "indicators": ["margin_heat", "margin_accel", "short_ratio", "offset_rate", "margin_diverge"],
        "target_days": TARGET_DAYS,
        "dates": _dates_iso,  # ISO 格式 "2020-02-11" 等，長度 = TARGET_DAYS
    }, open(OUT_META, "wb"))

    # Sanity stats
    print(f"\n=== 輸出 ===")
    print(f"{OUT_TENSOR}  shape={tensor.shape}  大小={tensor.nbytes/1024/1024:.1f} MB")
    print(f"{OUT_META}")
    print(f"\n=== 指標統計 ===")
    names = ["margin_heat", "margin_accel", "short_ratio", "offset_rate", "margin_diverge"]
    for i, name in enumerate(names):
        s = tensor[:, :, i]
        nz = s[s != 0]
        nz_pct = len(nz) / s.size * 100 if s.size else 0
        print(f"  {name:<17s} mean={s.mean():7.2f} std={s.std():7.2f} "
              f"min={s.min():7.2f} max={s.max():7.2f}  non-zero={nz_pct:.1f}%")


if __name__ == "__main__":
    main()
