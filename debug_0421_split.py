"""
Debug 4/21 那天 realistic vs Tab 3 為何選不同股
Tab 3 選達邁(3645.TW)，realistic 選聯茂(6213)
"""
import json, os, sys, types, urllib.request
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
import numpy as np
from gpu_cupy_evolve import precompute, cpu_replay, download_data


def fetch_gist_strategy():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30
    )
    d = json.loads(r.read())
    return json.loads(d["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_gist_strategy()
p = strategy.get("params", strategy)
print(f"策略 {strategy.get('score',0):.3f} 分")

_lens = [len(v) for v in data.values()]
TARGET = 1500 if sum(1 for l in _lens if l >= 1500) >= 500 else (1200 if sum(1 for l in _lens if l >= 1200) >= 800 else 900)
data = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
print(f"TARGET = {TARGET}")

# 確認達邁 / 聯茂在 cache 裡
print(f"\n達邁(3645.TW) in cache: {'3645.TW' in data}")
print(f"聯茂(6213.TW) in cache: {'6213.TW' in data}")
if "3645.TW" in data:
    df = data["3645.TW"]
    print(f"  達邁 期間: {df.index[0]} ~ {df.index[-1]}")
    print(f"  達邁 4/21 close: {df[df.index.normalize() == '2026-04-20'].iloc[-1]['Close'] if len(df[df.index.normalize() == '2026-04-20']) else 'N/A'}")
    # 看 4/20 (D 訊號日) 是否在
    target_dates = ['2026-04-17', '2026-04-18', '2026-04-21', '2026-04-22', '2026-04-23', '2026-04-24']
    for td in target_dates:
        mask = df.index.normalize() == td
        if mask.any():
            row = df[mask].iloc[0]
            print(f"  達邁 {td}: O={row['Open']:.2f} C={row['Close']:.2f} V={int(row['Volume'])}")
        else:
            print(f"  達邁 {td}: 不在")

# 跑 precompute + cpu_replay
pre = precompute(data)
trades = cpu_replay(pre, p)

# 看 4/20 / 4/21 那天的 buy_date trade
print(f"\n=== buy_date 在 4/20-4/24 區間的 trade ===")
for t in sorted(trades, key=lambda x: x.get("buy_date", "")):
    bd = t.get("buy_date", "")
    if "2026-04-20" <= bd <= "2026-04-24":
        print(f"  buy_date={bd} ticker={t.get('ticker')} name={t.get('name')} buy={t.get('buy_price')} reason={t.get('reason')}")

# 看達邁、聯茂、希華有沒有出現在 trades
print(f"\n=== 達邁/聯茂/希華 在 trades 裡的所有紀錄 ===")
for tk in ["3645.TW", "6213.TW", "2484.TW"]:
    rels = [t for t in trades if t.get("ticker") == tk]
    print(f"  {tk}: {len(rels)} 筆")
    for r in rels[-3:]:
        print(f"    buy_date={r.get('buy_date')} sell_date={r.get('sell_date','持有中')} buy={r.get('buy_price')} reason={r.get('reason')}")

# 印 universe (4/20 那天的 top100)
print(f"\n=== 4/20 達邁 vs 聯茂 在 top100 嗎？(用 vol_prev 排序) ===")
dates = pre["dates"]
date_norm = [d.normalize() if hasattr(d, 'normalize') else d for d in dates]
target_d = None
for i, d in enumerate(dates):
    d_str = str(d.date()) if hasattr(d, 'date') else str(d)[:10]
    if d_str == "2026-04-20":
        target_d = i; break
if target_d is not None:
    tickers = pre["tickers"]
    vol_prev = pre.get("vol_prev")
    if vol_prev is not None:
        vols_at_d = vol_prev[:, target_d]
        order = np.argsort(-vols_at_d)
        top100 = order[:100]
        for tk_search in ["3645.TW", "6213.TW", "2484.TW"]:
            try:
                idx = tickers.index(tk_search)
                rank = list(order).index(idx) + 1
                in_top = rank <= 100
                print(f"  {tk_search}: vol_prev rank #{rank} {'(in top100)' if in_top else '(NOT in top100)'}")
            except (ValueError, IndexError):
                print(f"  {tk_search}: 不在 tickers")
