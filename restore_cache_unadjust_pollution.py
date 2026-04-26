"""一次性救援：cache 最後一天 close 被 unadjusted 污染，重抓 yfinance adjusted 覆蓋。

執行方法（Windows）：
  cd C:\stock-evolution
  python restore_cache_unadjust_pollution.py
"""
import sys, types
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

import os, pickle, time
import pandas as pd
import yfinance as yf
from gpu_cupy_evolve import CACHE_PATH

print(f"Cache: {CACHE_PATH}")
print()

with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)
print(f"Loaded {len(cache)} tickers")

# 找 cache 末日
last_dates = {t: df.index[-1].normalize() for t, df in cache.items() if df is not None and len(df) > 0}
from collections import Counter
dist = Counter(d.date() for d in last_dates.values())
print(f"Last day dist: {dict(sorted(dist.items())[-5:])}")

most_common_date = dist.most_common(1)[0][0]  # date object
print(f"Target date to refresh: {most_common_date}")
print()

# 抓所有 ticker，最後一天用 yfinance auto_adjust=True 重抓覆蓋
target_pd = pd.Timestamp(most_common_date).tz_localize(None)
target_str = target_pd.strftime("%Y-%m-%d")
end_str = (target_pd + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

tickers_at_last = [t for t, ld in last_dates.items() if ld.tz_localize(None) == target_pd]
print(f"Tickers with last_day={target_str}: {len(tickers_at_last)}")
print()

# Batch download, auto_adjust=True (跟 append_new_days 一致)
BATCH = 100
fixed_count = 0
mismatch_count = 0
for bi in range(0, len(tickers_at_last), BATCH):
    batch = tickers_at_last[bi:bi+BATCH]
    try:
        df = yf.download(batch, start=target_str, end=end_str,
                        group_by='ticker', progress=False, threads=True,
                        auto_adjust=True)
    except Exception as e:
        print(f"  batch {bi} fail: {e}")
        continue
    
    for tk in batch:
        try:
            if len(batch) == 1:
                sub = df
            else:
                if tk not in df.columns.get_level_values(0).unique(): continue
                sub = df[tk]
            sub = sub[sub['Close'].notna()]
            if len(sub) == 0: continue
            
            new_close = float(sub['Close'].iloc[-1])
            if new_close <= 0: continue
            
            # Compare with cache
            old_close = float(cache[tk]['Close'].iloc[-1])
            if abs(new_close - old_close) / max(old_close, 1) > 0.001:  # > 0.1%
                # 有差異，覆蓋
                cache[tk].iloc[-1, cache[tk].columns.get_loc('Close')] = new_close
                # 順便把 OHLV 也覆蓋（保險）
                for col, val in [('Open', sub['Open'].iloc[-1]), 
                                  ('High', sub['High'].iloc[-1]),
                                  ('Low', sub['Low'].iloc[-1]),
                                  ('Volume', sub['Volume'].iloc[-1])]:
                    if col in cache[tk].columns and pd.notna(val):
                        cache[tk].iloc[-1, cache[tk].columns.get_loc(col)] = val
                mismatch_count += 1
                if mismatch_count <= 10:
                    print(f"  {tk}: cache {old_close:.2f} → yf {new_close:.2f}")
            fixed_count += 1
        except Exception:
            continue
    
    if (bi // BATCH) % 5 == 0:
        print(f"  進度 {min(bi+BATCH, len(tickers_at_last))}/{len(tickers_at_last)}  改寫 {mismatch_count} 檔")
    time.sleep(0.5)

print()
print(f"Total: {fixed_count} 檔 yfinance 抓到, {mismatch_count} 檔被改回 yfinance adjusted")
print()

# 寫回
tmp = CACHE_PATH + ".tmp"
with open(tmp, "wb") as f:
    pickle.dump(cache, f)
os.replace(tmp, CACHE_PATH)
print(f"✅ 寫回 cache")
