"""
V34 sanity test — margin 5 指標到底有沒有 alpha？
用法：C:\\stock-evolution> python sanity_test_margin.py

檢查每個 margin 指標跟 forward 5/10/30 日報酬的 Spearman/Pearson 相關係數。
|mean_spearman| > 0.03 (across all stocks) → 有 alpha，值得寫 V34 GPU kernel
|mean_spearman| < 0.01 → 沒訊號，省下一天工程時間，換方向（ex: TWT72U 借券）

這是「V31 法人籌碼 24h 白忙」教訓的補救 — 寫 CUDA kernel 前先驗證。
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
TENSOR_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_tensor.npy")
META_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_tensor_meta.pkl")

HORIZONS = [5, 10, 30]        # forward-return window 天數
MIN_VALID = 100               # 每檔股票要夠樣本才算進統計
ALPHA_THRESHOLD = 0.03        # |mean spearman| >= 這個 → 有 alpha
DEAD_THRESHOLD = 0.01         # 全部 < 這個 → 沒訊號，放棄


def forward_returns(close: np.ndarray, h: int) -> np.ndarray:
    """close: shape (T,), returns shape (T,) with forward h-day % return, last h days = NaN"""
    out = np.full_like(close, np.nan, dtype=np.float32)
    if len(close) > h:
        out[:-h] = (close[h:] / np.where(close[:-h] > 0, close[:-h], np.nan) - 1) * 100
    return out


def main():
    print("=== V34 margin sanity test ===")
    cache = pickle.load(open(CACHE_PATH, "rb"))
    tensor = np.load(TENSOR_PATH)
    meta = pickle.load(open(META_PATH, "rb"))
    tickers = meta["tickers"]
    names = meta["indicators"]
    print(f"Tensor shape: {tensor.shape}  (days x stocks x {len(names)} indicators)")
    print(f"Horizons: {HORIZONS} 天 forward return\n")

    # 對每個 (indicator, horizon) 收集所有股票的 Spearman 相關係數
    # stats: names × horizons × list of per-stock correlations
    per_stock_corr = {(n, h): [] for n in names for h in HORIZONS}
    per_stock_n = 0

    for si, key in enumerate(tickers):
        df = cache[key]
        close = df["Close"].tail(tensor.shape[0]).values.astype(np.float32)
        # 如果股票 cache 比 tensor 短（不該發生，但保險），對齊
        if len(close) < tensor.shape[0]:
            close = np.concatenate([np.zeros(tensor.shape[0] - len(close)), close])

        for h in HORIZONS:
            fr = forward_returns(close, h)
            mask = ~np.isnan(fr)
            if mask.sum() < MIN_VALID:
                continue
            for ii, name in enumerate(names):
                x = tensor[:, si, ii]
                # 避免整欄都一樣（std=0 → Spearman undefined）
                if x.std() < 1e-6:
                    continue
                m2 = mask & (~np.isnan(x))
                if m2.sum() < MIN_VALID:
                    continue
                rho, _ = stats.spearmanr(x[m2], fr[m2])
                if not np.isnan(rho):
                    per_stock_corr[(name, h)].append(rho)
        per_stock_n += 1
        if (si + 1) % 500 == 0:
            print(f"  處理 {si+1}/{len(tickers)}")

    # 彙整結果
    print(f"\n=== Spearman 相關係數（{per_stock_n} 檔平均）===")
    header = f"{'indicator':<18s}" + "".join(f"{'h='+str(h):>10s}" for h in HORIZONS)
    print(header)
    print("-" * len(header))
    any_alpha = False
    max_abs_mean = 0.0
    for name in names:
        line = f"{name:<18s}"
        for h in HORIZONS:
            arr = np.array(per_stock_corr[(name, h)])
            if len(arr) == 0:
                line += f"{'N/A':>10s}"
                continue
            mean = arr.mean()
            line += f"{mean:>+10.4f}"
            if abs(mean) >= ALPHA_THRESHOLD:
                any_alpha = True
            max_abs_mean = max(max_abs_mean, abs(mean))
        print(line)
    print()

    # Go/no-go 裁決
    if any_alpha:
        print(f"🟢 有 alpha 訊號 (max |mean|={max_abs_mean:.4f} >= {ALPHA_THRESHOLD})")
        print(f"   值得寫 V34 GPU kernel 投入 24h 搜尋")
    elif max_abs_mean < DEAD_THRESHOLD:
        print(f"🔴 無訊號 (max |mean|={max_abs_mean:.4f} < {DEAD_THRESHOLD})")
        print(f"   margin 指標可能跟 OHLCV 高度相關，GPU 搜尋勝算低")
        print(f"   建議 pivot：試 TWSE TWT72U 借券資料（法人放空訊號）")
    else:
        print(f"🟡 訊號微弱 (max |mean|={max_abs_mean:.4f})")
        print(f"   邊際值得嘗試；可先寫 V34 kernel 但降低期望")

    # 額外：每個 indicator 的分布診斷（正負都集中才有 alpha）
    print(f"\n=== 分布診斷（看各股 correlation 是否一致）===")
    for name in names:
        for h in HORIZONS:
            arr = np.array(per_stock_corr[(name, h)])
            if len(arr) == 0:
                continue
            pos_pct = (arr > 0.05).mean() * 100
            neg_pct = (arr < -0.05).mean() * 100
            print(f"  {name:<18s} h={h:>3d}: >+0.05 {pos_pct:5.1f}%  <-0.05 {neg_pct:5.1f}%  std={arr.std():.3f}")


if __name__ == "__main__":
    main()
