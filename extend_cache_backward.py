"""將 Windows stock_data_cache.pkl 往前擴展歷史資料
目標：補 2020-01-01 ~ cache 最早日期，讓 cache 涵蓋 6 年（含 2020 covid 暴跌 + 2020-2021 大 bull 市）

用 yfinance 批次下載，舊 cache byte-level 不動，只在前面 prepend 新的舊資料。

用法：python extend_cache_backward.py
"""
import os, sys, pickle, time
import pandas as pd

# Mock cupy（避免 GPU 初始化）
class _MockCupy:
    def __getattr__(self, name): return lambda *a, **k: None
sys.modules.setdefault("cupy", _MockCupy())

try:
    import yfinance as yf
except ImportError:
    print("請先 pip install yfinance")
    sys.exit(1)

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
TARGET_START = pd.Timestamp("2020-01-01")  # 想要補到的最早日期
BATCH = 100


def main():
    if not os.path.exists(CACHE_PATH):
        print(f"❌ cache 不存在：{CACHE_PATH}")
        sys.exit(1)
    print(f"[讀] {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"[資訊] 共 {len(cache)} 檔")

    # 掃每檔最早日期 + 剝 tz
    earliest = {}
    for t, df in cache.items():
        if df is None or len(df) == 0: continue
        idx = df.index
        if getattr(idx, 'tz', None) is not None:
            df.index = idx.tz_localize(None)
            cache[t] = df
        earliest[t] = df.index[0]

    # 統計
    if not earliest:
        print("❌ 沒 ticker 有資料"); sys.exit(1)

    ed = pd.Series(earliest)
    print(f"[現況] 最早日期分佈：")
    print(f"  min = {ed.min().date()} | max = {ed.max().date()} | median = {ed.median().date()}")
    need_extend = [t for t, d in earliest.items() if d > TARGET_START]
    print(f"[目標] 補到 {TARGET_START.date()}")
    print(f"       {len(need_extend)} 檔需要往前補，{len(cache) - len(need_extend)} 檔已涵蓋")

    if not need_extend:
        print("✅ 全部 cache 已涵蓋目標範圍，無需動作")
        return

    # 按最早日期分組（通常所有 ticker 同一天，只有 1 組）
    groups = {}
    for t in need_extend:
        groups.setdefault(earliest[t], []).append(t)

    total_added = 0
    fail_count = 0
    for old_start, tickers in groups.items():
        # 下載 TARGET_START ~ old_start (不含 old_start，避免重複)
        end_date = old_start - pd.Timedelta(days=1)
        if end_date < TARGET_START:
            continue
        print(f"\n[{TARGET_START.date()} → {old_start.date()}] {len(tickers)} 檔需要補 (yfinance 批次下載)")
        for i in range(0, len(tickers), BATCH):
            batch = tickers[i:i+BATCH]
            try:
                raw = yf.download(batch, start=TARGET_START, end=old_start,
                                  group_by='ticker', threads=True, progress=False, auto_adjust=False)
            except Exception as e:
                print(f"  批次 {i} 下載失敗：{e}")
                fail_count += len(batch)
                continue
            for t in batch:
                try:
                    if len(batch) == 1:
                        new = raw.copy()
                    else:
                        if t not in raw.columns.levels[0]: continue
                        new = raw[t].copy()
                    if new is None or len(new) == 0: continue
                    # 剝 tz
                    if getattr(new.index, 'tz', None) is not None:
                        new.index = new.index.tz_localize(None)
                    # 丟 Close NaN
                    if "Close" in new.columns:
                        new = new[new["Close"].notna()]
                    if len(new) == 0: continue
                    # 只保留跟現有 cache 相同欄位
                    df = cache[t]
                    common_cols = [c for c in df.columns if c in new.columns]
                    new = new[common_cols]
                    # 硬過濾：只接受 < df 最早日期的 row（不覆蓋現有資料）
                    new = new[new.index < df.index[0]]
                    if len(new) == 0: continue
                    # concat（new 在前 + df 在後）
                    merged = pd.concat([new, df]).sort_index()
                    # dedup 保留舊（第一個出現的）
                    if merged.index.duplicated().any():
                        merged = merged[~merged.index.duplicated(keep='last')]  # 保留原 cache 的
                    cache[t] = merged
                    total_added += len(new)
                except Exception as e:
                    fail_count += 1
                    if fail_count <= 3:
                        print(f"  {t}: {e}")
            done = min(i+BATCH, len(tickers))
            print(f"  進度 {done}/{len(tickers)} | 累計 append {total_added} 筆 舊資料")

    # 原子寫入
    print(f"\n[寫入] {CACHE_PATH}")
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, CACHE_PATH)

    # 驗證
    print(f"\n=== 擴展後 ===")
    earliest_after = {t: df.index[0] for t, df in cache.items() if df is not None and len(df) > 0}
    ed2 = pd.Series(earliest_after)
    print(f"  min = {ed2.min().date()} | max = {ed2.max().date()} | median = {ed2.median().date()}")
    lens = pd.Series([len(df) for df in cache.values()])
    print(f"  天數：min={lens.min()} max={lens.max()} median={int(lens.median())}")
    print(f"  共 append {total_added} 筆舊資料 | 失敗 {fail_count} 筆")

    # 採樣驗證（舊資料 byte-level 一致）
    print(f"\n=== 舊資料未動驗證（3 個樣本）===")
    import random
    samples = random.sample(list(cache.keys()), min(3, len(cache)))
    for t in samples:
        df = cache[t]
        after_2022_07 = df[df.index >= pd.Timestamp("2022-08-01")]
        print(f"  {t}: 新總長 {len(df)} 天, 2022-08 之後 {len(after_2022_07)} 天（舊資料區）")

    print(f"\n✅ 完成！{len(cache)} 檔 cache 已擴展")
    print(f"💡 GPU 要用新資料需要把 TARGET_DAYS 從 900 改成 1500")


if __name__ == "__main__":
    main()
