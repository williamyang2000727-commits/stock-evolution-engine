"""
動態股票池回測 v3 — 直接用 precompute() 指標，跟 GPU 完全一致
"""
import numpy as np
import json, os, sys, types
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name

data = download_data()
with open("best_strategy.json", encoding="utf-8") as f:
    strategy = json.load(f)
p = strategy["params"]
print(f"v{strategy.get('version','?')} | {strategy.get('score',0):.2f}分")

data = {k:v for k,v in data.items() if len(v) >= 400}
pre = precompute(data)
ns, nd = pre["n_stocks"], pre["n_days"]
dates = pre["dates"]
print(f"股票：{ns} 檔 | 天數：{nd}")
print(f"期間：{dates[0].date()} ~ {dates[-1].date()}")

# 直接用 cpu_replay（跟 GPU kernel 完全一致的 Python 版）
trades = cpu_replay(pre, p)
trades.sort(key=lambda x: x["buy_date"])
n = len(trades)
if n == 0:
    print("無交易"); sys.exit()

total = sum(t["return"] for t in trades)
avg = total / n
wins = sum(1 for t in trades if t["return"] > 0)

print(f"\n{'='*60}")
print(f" 動態回測 v3（D+1開盤賣 + D+1收盤買 + 扣交易成本）")
print(f" v{strategy.get('version','?')} | {n}筆 | 平均{avg:.2f}% | 總{total:.2f}% | 勝率{wins/n*100:.1f}%")
print(f"{'='*60}")

years = {}
for t in trades:
    y = t["buy_date"][:4]
    if y not in years: years[y] = []
    years[y].append(t)

for year in sorted(years.keys()):
    ts = years[year]
    yn = len(ts)
    yt = sum(t["return"] for t in ts)
    ya = yt / yn
    yw = sum(1 for t in ts if t["return"] > 0)
    print(f"\n--- {year}年 | {yn}筆 | 平均{ya:.2f}% | 總{yt:.2f}% | 勝率{yw/yn*100:.1f}% ---")
    for i, t in enumerate(ts):
        e = "V" if t["return"] > 0 else "X"
        tk = t["ticker"].replace(".TW", "").replace(".TWO", "")
        print(f" {i+1:2d}. {e} {t['name']}({tk}) | {t['buy_date'][5:]}>{t['sell_date'][5:]} | {t['buy_price']}>{t['sell_price']} | {t['return']:+.2f}% | {t['days']}天 | {t['reason']}")

input("\n按 Enter 結束...")
