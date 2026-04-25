"""
直接看每個 ticker 的最後一天，分組看 cache 端點分布
"""
import os, pickle
from collections import Counter
import pandas as pd

USER_SE = r"C:\stock-evolution" if os.name == "nt" else os.path.expanduser("~/stock-evolution")
if not os.path.isdir(USER_SE):
    USER_SE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
raw = pickle.load(open(CACHE_PATH, "rb"))
print(f"Cache: {len(raw)} 檔")

# 1. 每檔最後一天分布
end_dates = Counter()
for tk, df in raw.items():
    last = df.index[-1]
    last_norm = pd.Timestamp(last).normalize()
    if hasattr(last_norm, "tz_localize") and last_norm.tz is not None:
        last_norm = last_norm.tz_localize(None)
    end_dates[last_norm.date()] += 1

print(f"\n=== 每個 ticker 的最後日期分布 ===")
for d, n in sorted(end_dates.items()):
    print(f"  {d}: {n} 檔 ({n/len(raw)*100:.1f}%)")

# 2. 焦點檢查：3645/6213/2484/6217/3498/6530 各自最後 5 天
print(f"\n=== 焦點 ticker 最後 5 天 ===")
for tk in ["3645.TW", "6213.TW", "2484.TW", "6217.TWO", "3498.TWO", "6530.TWO"]:
    if tk not in raw:
        print(f"  {tk}: NOT IN CACHE")
        continue
    df = raw[tk]
    print(f"  {tk}:")
    for d in df.index[-5:]:
        d_norm = pd.Timestamp(d).normalize()
        if hasattr(d_norm, "tz_localize") and d_norm.tz is not None:
            d_norm = d_norm.tz_localize(None)
        row = df.loc[d]
        print(f"    {d_norm.date()}: O={row['Open']:.2f} C={row['Close']:.2f} V={int(row['Volume'])}")

# 3. cache pkl 檔案 modified time
mt = os.path.getmtime(CACHE_PATH)
print(f"\n=== Cache file mtime ===")
print(f"  modified: {pd.Timestamp(mt, unit='s', tz='Asia/Taipei')}")
size_mb = os.path.getsize(CACHE_PATH) / 1024 / 1024
print(f"  size: {size_mb:.1f} MB")
