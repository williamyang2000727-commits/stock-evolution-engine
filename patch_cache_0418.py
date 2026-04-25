r"""
補 4/18 那天 cache 資料

問題：~/stock-evolution/stock_data_cache.pkl 缺 2026-04-18
原因：那天 update_cache 可能 fail，或 yfinance 那天沒回資料
影響：cpu_replay 在 04-17 訊號日 vs 真實 04-18 訊號日選不同股 → realistic 跟 Tab 3 分歧

修法：用 yfinance 重抓 4/18 那一天，merge 回 cache pkl
"""
import os, pickle, sys, time
import pandas as pd

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
TARGET_DATE = pd.Timestamp("2026-04-18").normalize()

print(f"Cache: {CACHE_PATH}")
raw = pickle.load(open(CACHE_PATH, "rb"))
print(f"  共 {len(raw)} 檔")

# 1. 先檢查到底缺哪些股
print(f"\n1) 檢查 {TARGET_DATE.date()} 在每個 ticker 是否有")
n_with, n_without = 0, 0
sample_with, sample_without = [], []
for tk, df in raw.items():
    df_dates_norm = df.index.normalize() if df.index.tz is None else df.index.tz_localize(None).normalize()
    if (df_dates_norm == TARGET_DATE).any():
        n_with += 1
        if len(sample_with) < 3:
            sample_with.append(tk)
    else:
        # 確認附近的日期 (4/17 / 4/21) 有，才算「缺中間」
        d417 = (df_dates_norm == pd.Timestamp("2026-04-17")).any()
        d421 = (df_dates_norm == pd.Timestamp("2026-04-21")).any()
        if d417 and d421:
            n_without += 1
            if len(sample_without) < 5:
                sample_without.append(tk)

print(f"  有 4/18 資料：{n_with} 檔")
print(f"  缺 4/18 但 4/17 + 4/21 都有：{n_without} 檔")
print(f"  範例（有）：{sample_with}")
print(f"  範例（缺）：{sample_without}")

if n_without == 0:
    print("\n  ✅ 沒人缺 4/18，無需修補")
    sys.exit()

# 2. 用 yfinance 抓 4/18 那一天
print(f"\n2) 用 yfinance 抓 4/18 數據 ...")
try:
    import yfinance as yf
except ImportError:
    print("  ❌ 需要 yfinance：pip install yfinance")
    sys.exit(1)

# yfinance period 至少 5d 比較穩
tickers_to_fix = [tk for tk, df in raw.items()
                  if not (df.index.tz_localize(None).normalize() if df.index.tz else df.index.normalize()).isin([TARGET_DATE]).any()]
print(f"  需要修的 ticker 數：{len(tickers_to_fix)}")

batch_size = 50
fixed = 0
failed = 0
for i in range(0, len(tickers_to_fix), batch_size):
    batch = tickers_to_fix[i:i+batch_size]
    print(f"  Batch {i//batch_size + 1}/{(len(tickers_to_fix)-1)//batch_size + 1} ({len(batch)} 檔) ...")
    for tk in batch:
        try:
            h = yf.Ticker(tk).history(start="2026-04-17", end="2026-04-22", auto_adjust=False)
            if len(h) == 0:
                failed += 1
                continue
            # 找 4/18 那行
            h_norm = h.index.tz_localize(None).normalize() if h.index.tz else h.index.normalize()
            mask = h_norm == TARGET_DATE
            if not mask.any():
                failed += 1
                continue
            row_418 = h[mask].iloc[0]
            # 將 row_418 加入原 cache
            df_orig = raw[tk]
            # 對齊原 cache 的 tz
            new_idx = TARGET_DATE.tz_localize(df_orig.index.tz) if df_orig.index.tz else TARGET_DATE
            new_row = pd.DataFrame({
                "Open": [row_418["Open"]],
                "High": [row_418["High"]],
                "Low": [row_418["Low"]],
                "Close": [row_418["Close"]],
                "Volume": [row_418["Volume"]],
            }, index=[new_idx])
            # 補上 df_orig 其他欄位（Dividends / Stock Splits 等）
            for col in df_orig.columns:
                if col not in new_row.columns:
                    new_row[col] = 0
            new_row = new_row[df_orig.columns]
            df_new = pd.concat([df_orig, new_row]).sort_index()
            df_new = df_new[~df_new.index.duplicated(keep="first")]
            raw[tk] = df_new
            fixed += 1
        except Exception as e:
            failed += 1
    time.sleep(1)  # rate limit 保險

print(f"\n  修補完成：成功 {fixed} / 失敗 {failed}")

# 3. 寫回 cache
if fixed > 0:
    backup_path = CACHE_PATH.replace(".pkl", f".bak_before_0418fix.pkl")
    print(f"\n3) 備份原 cache → {backup_path}")
    import shutil
    shutil.copy(CACHE_PATH, backup_path)
    print(f"4) 寫回 {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(raw, f)
    print(f"  ✅ 已寫回，請重跑 realistic_test.py 驗證")
else:
    print(f"\n  ❌ 沒修到任何一檔，cache 不寫回")
