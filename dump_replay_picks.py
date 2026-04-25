"""
跑 cpu_replay，記錄它每天 best_si 的選擇 + 算出的 score
直接 instrument cpu_replay 看它真的選誰，不靠手寫 cpu_score
"""
import os, sys, json, types, urllib.request
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd

# Monkey-patch cpu_replay 加 instrumentation
import gpu_cupy_evolve as base
from gpu_cupy_evolve import precompute, download_data


def fetch_gist_strategy():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_gist_strategy()
p = strategy.get("params", strategy)

TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data_t)
trades = base.cpu_replay(pre, p)

# 排序
trades.sort(key=lambda t: t.get("buy_date", ""))

# 印 4/15 之後所有 trades
print(f"\n=== 4/15 之後 cpu_replay 所有 trades ===")
for t in trades:
    bd = t.get("buy_date", "")
    if bd >= "2026-04-15":
        sd = t.get("sell_date", "持有中")
        print(f"  buy={bd} > sell={sd:11s} | {t.get('name',''):>10}({t.get('ticker','')}) buy_p={t.get('buy_price',0)} sell_p={t.get('sell_price',0)} reason={t.get('reason','')}")

# 也印 sell_date 在這區間的（前期持有的賣出）
print(f"\n=== sell_date 在 4/15 之後的 trades（前期持有賣掉的）===")
for t in trades:
    sd = t.get("sell_date", "")
    bd = t.get("buy_date", "")
    if sd and "2026-04-15" <= sd <= "2026-04-22" and bd < "2026-04-15":
        print(f"  buy={bd} > sell={sd} | {t.get('name',''):>10}({t.get('ticker','')}) buy_p={t.get('buy_price',0)} sell_p={t.get('sell_price',0)} reason={t.get('reason','')}")

# 找跟 cpu_replay 「真正進場日 4/21 訊號日 4/17」的兩檔
print(f"\n=== cpu_replay 持有中的 2 檔 ===")
for t in trades:
    if t.get("reason") == "持有中":
        print(f"  {t.get('ticker')} ({t.get('name')}): buy_date={t.get('buy_date')} buy_price={t.get('buy_price')}")

# 對 04-17 訊號日，列 cpu_replay 應該選的 universe top 5（用 best_si 邏輯但不真模擬）
# 04-17 = day index ?
dates = pre["dates"]
day_0417 = None
for i, d in enumerate(dates):
    if pd.Timestamp(d).strftime("%Y-%m-%d") == "2026-04-17":
        day_0417 = i; break

if day_0417 is not None:
    # 跟 cpu_replay 完全一樣的 score 邏輯：line 1442-1542 _score_stock
    # 但我之前 cpu_score 是手寫，可能漏掉 sector_hot 等
    # 直接讀 _score_stock 的 source 看怎麼算
    import inspect
    print(f"\n=== gpu_cupy_evolve.cpu_replay._score_stock source（前 60 行）===")
    src_lines = inspect.getsource(base.cpu_replay).split("\n")
    in_score = False
    line_count = 0
    for i, line in enumerate(src_lines):
        if "_score_stock" in line and "def" in line:
            in_score = True
        if in_score:
            print(f"  {i}: {line}")
            line_count += 1
            if line_count > 100:
                break
        if in_score and line.strip().startswith("return ") and line_count > 5:
            in_score = False
            break
