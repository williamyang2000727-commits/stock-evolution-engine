"""手動 append cache 到今天（舊資料完全不動）。

用法：python update_cache.py

會印出更新前後「最後一天」分布對照，確認舊資料沒被動到。
中斷也安全（原子寫入，不會損壞 cache）。
"""
import sys, types
# mock cupy 避免 module load 時的 GPU 初始化（我們只要 append 函式）
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

import os, pickle
from collections import Counter
from gpu_cupy_evolve import append_new_days, CACHE_PATH

def dump_summary(label, cache):
    if not cache:
        print(f"  {label}: (無 cache)")
        return
    c = Counter(df.index[-1].date() for df in cache.values() if len(df))
    print(f"  {label}: 共 {len(cache)} 檔")
    for dt, n in sorted(c.items())[-6:]:
        print(f"    {dt}: {n} 檔")

print(f"Cache path: {CACHE_PATH}")
print()

if not os.path.exists(CACHE_PATH):
    print("cache 不存在，無法 append")
    sys.exit(1)

# 讀更新前狀態
with open(CACHE_PATH, "rb") as f:
    before = pickle.load(f)
print("=== 更新前 ===")
dump_summary("最後一天分布", before)

# 抓幾檔樣本存起來比對（確認舊資料沒變）
sample_tickers = list(before.keys())[:3]
sample_before = {t: before[t].copy() for t in sample_tickers}

print()
print("=== 開始 append（只抓新天，舊不動）===")
cache, n = append_new_days(CACHE_PATH)
print(f"  {n} 檔有新資料被 append")

if cache is None:
    print("cache 讀取失敗")
    sys.exit(1)

print()
print("=== 更新後 ===")
dump_summary("最後一天分布", cache)

# 驗證舊資料未變（兩邊都剝 tz 再比對，避免 tz-aware vs naive 衝突）
print()
print("=== 舊資料未變驗證（樣本）===")
for t in sample_tickers:
    old_df = sample_before[t].copy()
    new_df = cache[t].copy()
    # 統一 naive
    if old_df.index.tz is not None:
        old_df.index = old_df.index.tz_localize(None)
    if new_df.index.tz is not None:
        new_df.index = new_df.index.tz_localize(None)
    # 取新 cache 裡對應舊長度的前 N 列（舊資料應該原封不動在最前面）
    new_head = new_df.iloc[:len(old_df)]
    # 比對 Close 值（最敏感指標）
    match = (
        len(old_df) == len(new_head)
        and (old_df.index.values == new_head.index.values).all()
        and (old_df["Close"].values == new_head["Close"].values).all()
    )
    added = len(new_df) - len(old_df)
    print(f"  {t}: 舊 {len(old_df)} 天 {'✅ 完全一致' if match else '❌ 被動到了!'}, 新增 {added} 天")
