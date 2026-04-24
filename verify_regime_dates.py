"""
V35 日期對齊驗證 — 確認 regime 日期 vs cache 日期一致
用法：C:\\stock-evolution> python verify_regime_dates.py
"""
import os, sys, pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path: sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(_USER_SE, "stock_data_cache.pkl")


def main():
    print("=== V35 regime 日期對齊驗證 ===\n")

    # Step 1: 載入 cache + 過濾（跟 base.main() 一致）
    print("[1/4] 載入 cache...")
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    _n_1500 = sum(1 for l in _lens if l >= 1500)
    _n_1200 = sum(1 for l in _lens if l >= 1200)
    if _n_1500 >= 500: TARGET_DAYS = 1500
    elif _n_1200 >= 800: TARGET_DAYS = 1200
    else: TARGET_DAYS = 900
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    print(f"  {len(raw)} → {len(data)} 檔 × {TARGET_DAYS} 天")

    # Step 2: 檢查所有 ticker 的日期分布
    print(f"\n[2/4] 各 ticker 最後一天分布:")
    last_dates = {}
    for k, v in data.items():
        last = str(v.index[-1].date() if hasattr(v.index[-1], 'date') else v.index[-1])[:10]
        last_dates[last] = last_dates.get(last, 0) + 1
    for d in sorted(last_dates.keys(), reverse=True)[:10]:
        print(f"  {d}: {last_dates[d]} 檔")

    print(f"\n[2/4] 各 ticker 第一天分布:")
    first_dates = {}
    for k, v in data.items():
        first = str(v.index[0].date() if hasattr(v.index[0], 'date') else v.index[0])[:10]
        first_dates[first] = first_dates.get(first, 0) + 1
    for d in sorted(first_dates.keys())[:10]:
        print(f"  {d}: {first_dates[d]} 檔")

    # Step 3: base.precompute 跑出來的 dates 對比
    print(f"\n[3/4] base.precompute...")
    pre = base.precompute(data)
    print(f"  pre['dates'] 第一天:  {str(pre['dates'][0].date())[:10]}")
    print(f"  pre['dates'] 最後一天: {str(pre['dates'][-1].date())[:10]}")
    print(f"  pre['dates'] 長度:    {len(pre['dates'])}")
    print(f"  pre['n_days']:        {pre['n_days']}")

    # Step 4: 逐天驗證 tickers[0] 跟 market_close 的「日期」是不是對得上
    print(f"\n[4/4] 跨 ticker 日期一致性檢查...")
    # 取 3 個代表 ticker 的 index 跟 pre['dates'] 比對
    sample_tickers = list(data.keys())[:5]
    mismatch_total = 0
    for t in sample_tickers:
        df = data[t]
        df_dates = [str(d.date() if hasattr(d, 'date') else d)[:10] for d in df.index]
        pre_dates = [str(d.date() if hasattr(d, 'date') else d)[:10] for d in pre['dates']]
        n_common = min(len(df_dates), len(pre_dates))
        df_tail = df_dates[-n_common:]
        pre_tail = pre_dates[-n_common:]
        mismatch = sum(1 for a, b in zip(df_tail, pre_tail) if a != b)
        mismatch_total += mismatch
        _status = "✅" if mismatch == 0 else f"⚠️ {mismatch} 天錯位"
        print(f"  {t}: {df_tail[0]} ~ {df_tail[-1]} vs pre {pre_tail[0]} ~ {pre_tail[-1]} {_status}")

    print(f"\n總錯位: {mismatch_total} 天（5 檔 × {pre['n_days']} 天）")

    # Step 5: regime 關鍵日期標記
    from claude_v35_regime_gpu import _compute_regime_array
    market_close = pre["close"].mean(axis=0)
    regime = _compute_regime_array(market_close, 20, 60, pre['n_days'])
    names = ["BULL", "BEAR", "CHOP"]

    print(f"\n=== regime 時序樣本（每 100 天一抽）===")
    for i in range(60, pre['n_days'], 100):
        d = str(pre['dates'][i].date())[:10]
        print(f"  day {i:4d} ({d}): {names[int(regime[i])]}")
    # 印最後一天
    d = str(pre['dates'][-1].date())[:10]
    print(f"  day {pre['n_days']-1:4d} ({d}): {names[int(regime[-1])]} ← 最後一天")

    # 最關鍵：2022 熊市是不是被標 BEAR
    print(f"\n=== 2022 熊市驗證（應該大多是 BEAR）===")
    for i in range(pre['n_days']):
        d = str(pre['dates'][i].date())[:10]
        if d.startswith("2022-04") or d.startswith("2022-09") or d.startswith("2022-10"):
            print(f"  day {i:4d} ({d}): {names[int(regime[i])]}")
            if i > 0 and d.endswith("01"):
                pass  # 印每月月初 sample 就夠


if __name__ == "__main__":
    main()
