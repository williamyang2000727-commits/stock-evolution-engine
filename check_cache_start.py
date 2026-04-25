"""看你 cache 真實的起點分布"""
import os, sys, types, pickle
from collections import Counter
import pandas as pd

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
raw = pickle.load(open(CACHE_PATH, "rb"))
print(f"Cache: {len(raw)} tickers")

# 每個 ticker 的起點
starts = Counter()
ends = Counter()
for tk, df in raw.items():
    s = pd.Timestamp(df.index[0]).normalize()
    e = pd.Timestamp(df.index[-1]).normalize()
    if hasattr(s, "tz_localize") and s.tz is not None:
        s = s.tz_localize(None)
    if hasattr(e, "tz_localize") and e.tz is not None:
        e = e.tz_localize(None)
    starts[s.date()] += 1
    ends[e.date()] += 1

print("\n=== 起點分布（前 10 個最常見起點）===")
for d, n in sorted(starts.items())[:10]:
    print(f"  {d}: {n} 檔")
print(f"  ... (共 {len(starts)} 種不同起點)")

print("\n=== 末日分布 ===")
for d, n in sorted(ends.items()):
    print(f"  {d}: {n} 檔")

# 找最早最晚
print(f"\n最早起點: {min(starts.keys())}")
print(f"最晚起點: {max(starts.keys())}")
print(f"最早末日: {min(ends.keys())}")
print(f"最晚末日: {max(ends.keys())}")

# 多數派起點（超過 50% ticker 共有的起點）
total = len(raw)
threshold = total * 0.5
common_starts = sorted([(d, n) for d, n in starts.items() if n >= threshold])
print(f"\n=== 多數派起點（>= 50% ticker 有的起點）===")
for d, n in common_starts[:5]:
    print(f"  {d}: {n} 檔 ({n/total*100:.1f}%)")

# 建議的固定起點 = 「N 檔以上 ticker 都有資料」的最早日期
# 跑 cpu_replay 至少要 500 檔 → 找這個起點
target_n = 500
print(f"\n=== 至少 {target_n} 檔 ticker 共同覆蓋的最早起點 ===")
# 對每個日期算「有多少 ticker 在這天前已經開始」
all_dates = sorted(starts.keys())
candidate = None
for d in all_dates:
    n_active = sum(c for date, c in starts.items() if date <= d)
    if n_active >= target_n:
        candidate = d
        print(f"  最早 {d} 起，有 {n_active} 檔 ticker 已開始")
        break
print(f"\n  → 建議固定起點：{candidate}")
