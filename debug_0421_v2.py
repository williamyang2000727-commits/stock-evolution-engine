"""
Debug 4/21 分歧 v2（用對的 cache）

之前 v1 的 cache 端點對齊看不出來（用同一份 cache）
v2 改：用 download_data() 載對的 cache，重點看
  1. 達邁 / 聯茂 / 希華 在 4/16-4/22 的 OHLCV
  2. cpu_replay 在 4/16-4/22 每天選哪檔（best_si）
  3. score 排名（不是只看 vol_prev top100）
  4. 對比 Tab 3 (scan_results.json) 當天紀錄
"""
import os, sys, json, types, urllib.request
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from gpu_cupy_evolve import precompute, cpu_replay, download_data


def fetch_gist(gist_id):
    r = urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/gists/{gist_id}"), timeout=30
    )
    return json.loads(r.read())


# 載 cache（gpu_cupy_evolve 用 ~/stock-evolution，正確的 4/24 版）
data = download_data()
print(f"Cache loaded: {len(data)} tickers")

# 抓 89.905 + scan_results
strategy = json.loads(fetch_gist("c1bef892d33589baef2142ce250d18c2")["files"]["best_strategy.json"]["content"])
p = strategy.get("params", strategy)
print(f"Strategy: {strategy.get('score',0):.3f}")

# Tab 3 / scan_results 看當天到底寫了什麼
data_gist = fetch_gist("e1159b02a87d3c6ee9f33fb9ef61bb80")
scan = json.loads(data_gist["files"]["scan_results.json"]["content"])
print(f"\nscan_results.json date: {scan.get('date','?')}")
print(f"  pending_buy: {scan.get('pending_buy')}")
print(f"  pending_sells: {scan.get('pending_sells', [])}")
top10 = scan.get("top10", [])[:10]
print(f"  top10:")
for r in top10:
    print(f"    rank{r.get('rank',0)}: {r.get('name','')}({r.get('ticker','')}) score={r.get('score',0):.2f}")

# 關鍵 ticker 4/15-4/24 K 線
print(f"\n=== 關鍵 ticker K 線 ===")
target_dates = ["2026-04-15", "2026-04-16", "2026-04-17", "2026-04-18",
                "2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"]
for tk in ["3645.TW", "6213.TW", "2484.TW"]:
    if tk not in data:
        print(f"\n  {tk}: NOT IN CACHE")
        continue
    df = data[tk]
    print(f"\n  {tk} ({tk}):")
    for td in target_dates:
        td_ts = pd.Timestamp(td).normalize()
        df_dates_norm = df.index.normalize() if df.index.tz is None else df.index.tz_localize(None).normalize()
        mask = df_dates_norm == td_ts
        if mask.any():
            row = df[mask].iloc[0]
            print(f"    {td}: O={row['Open']:.2f} H={row['High']:.2f} L={row['Low']:.2f} C={row['Close']:.2f} V={int(row['Volume'])}")
        else:
            print(f"    {td}: 無資料")

# precompute + cpu_replay
TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data_t)
trades = cpu_replay(pre, p)

# 看 cpu_replay 在 4/15 - 4/24 的 trades
print(f"\n=== cpu_replay 結果（buy_date 在 4/15 ~ 4/24）===")
for t in sorted(trades, key=lambda x: x.get("buy_date","")):
    bd = t.get("buy_date","")
    if "2026-04-15" <= bd <= "2026-04-24":
        sd = t.get("sell_date","持有中")
        print(f"  buy_date={bd} ticker={t.get('ticker')} name={t.get('name')} buy_price={t.get('buy_price')} sell_date={sd} reason={t.get('reason')}")

# 看 cpu_replay 賣出（sell_date 在 4/15-4/24）
print(f"\n=== cpu_replay 結果（sell_date 在 4/15 ~ 4/24）===")
for t in sorted(trades, key=lambda x: x.get("sell_date","")):
    sd = t.get("sell_date","")
    if "2026-04-15" <= sd <= "2026-04-24":
        print(f"  sell_date={sd} ticker={t.get('ticker')} name={t.get('name')} buy_date={t.get('buy_date')} return={t.get('return',0):+.2f}% reason={t.get('reason')}")

# 達邁 / 聯茂 在 cpu_replay 結果裡的所有 trade
print(f"\n=== 達邁/聯茂/希華 全部 trade 紀錄 ===")
for tk in ["3645.TW", "6213.TW", "2484.TW"]:
    rels = [t for t in trades if t.get("ticker") == tk]
    print(f"  {tk}: {len(rels)} 筆")
    for r in rels:
        print(f"    {r.get('buy_date')} > {r.get('sell_date','持有中'):10s} | buy={r.get('buy_price'):>8} | reason={r.get('reason','')}")
