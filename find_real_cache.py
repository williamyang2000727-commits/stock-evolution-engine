"""
找出 Windows 上所有 stock_data_cache.pkl 並印端點

問題：check_cache_endpoints (用 C:\stock-evolution) 看到 4/02
     realistic_test (用 download_data() 來自 ~/stock-evolution = C:\Users\william\stock-evolution) 看到 4/24
     兩個不同 pkl
"""
import os, pickle, time
import pandas as pd

candidates = [
    r"C:\stock-evolution\stock_data_cache.pkl",
    os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl"),
    r"C:\Users\william\stock-evolution\stock_data_cache.pkl",
]

for path in candidates:
    print(f"\n{'='*70}")
    print(f"檢查：{path}")
    print(f"{'='*70}")
    if not os.path.exists(path):
        print("  ❌ 不存在")
        continue
    size_mb = os.path.getsize(path) / 1024 / 1024
    mt = os.path.getmtime(path)
    print(f"  大小：{size_mb:.1f} MB")
    print(f"  修改時間：{pd.Timestamp(mt, unit='s', tz='Asia/Taipei')}")
    try:
        raw = pickle.load(open(path, "rb"))
        print(f"  ticker 數：{len(raw)}")
        # 撈幾個焦點 ticker 的最後一天
        for tk in ["3645.TW", "6213.TW"]:
            if tk in raw:
                df = raw[tk]
                last = df.index[-1]
                last_norm = pd.Timestamp(last).normalize()
                if hasattr(last_norm, "tz_localize") and last_norm.tz is not None:
                    last_norm = last_norm.tz_localize(None)
                print(f"  {tk} 最後一天：{last_norm.date()} close={df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  ❌ 讀檔失敗：{e}")
