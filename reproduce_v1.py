"""
重現 v1 107.96 分策略的原始回測結果（MA20 + 無均價過濾）
用法: python reproduce_v1.py
"""
import numpy as np
import json, os, sys, types
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name
import math

# 載入 v1 參數（從 gist 或本地）
if os.path.exists("best_strategy_v1_backup.json"):
    with open("best_strategy_v1_backup.json", encoding="utf-8") as f:
        strategy = json.load(f)
elif os.path.exists("best_strategy.json"):
    with open("best_strategy.json", encoding="utf-8") as f:
        strategy = json.load(f)
else:
    print("找不到策略檔案"); sys.exit()

p = strategy["params"]
print(f"策略: {strategy.get('score',0):.2f}分")

# v1 原始條件：>= 900 天，無均價過濾
data = download_data()
data = {k:v for k,v in data.items() if len(v) >= 900}
print(f"股票池: {len(data)} 檔（v1 原始：無均價過濾）")

pre = precompute(data)

# 還原 MA20 大盤過濾（覆蓋 precompute 的 MA60）
close = pre["close"]
ml = pre["n_days"]
market_avg = np.mean(close, axis=0)
market_ma20 = np.zeros(ml, dtype=np.float32)
for i in range(20, ml):
    market_ma20[i] = np.mean(market_avg[i-20:i])
market_bull = np.zeros(ml, dtype=np.float32)
market_bull[20:] = (market_avg[20:] > market_ma20[20:]).astype(np.float32)
market_bull[:20] = 1.0
pre["market_bull"] = market_bull
print(f"大盤過濾: MA20（v1 原始）| {np.sum(market_bull > 0.5)}/{ml} 天多頭")

ns, nd = pre["n_stocks"], pre["n_days"]
dates = pre["dates"]
print(f"期間: {dates[0].date()} ~ {dates[-1].date()}")

trades = cpu_replay(pre, p)
trades = [t for t in trades if not math.isnan(t.get("return", 0))]
trades.sort(key=lambda x: x["buy_date"])
n = len(trades)
if n == 0:
    print("無交易"); sys.exit()

total = sum(t["return"] for t in trades)
avg = total / n
wins = sum(1 for t in trades if t["return"] > 0)

print(f"\n{'='*60}")
print(f" v1 原始回測（MA20 + 無均價過濾）")
print(f" {n}筆 | 平均{avg:.2f}% | 總{total:.2f}% | 勝率{wins/n*100:.1f}%")
print(f"{'='*60}")

years = {}
for t in trades:
    y = t["buy_date"][:4]
    if y not in years: years[y] = []
    years[y].append(t)

for year in sorted(years.keys()):
    ts = years[year]
    yn = len(ts); yt = sum(t["return"] for t in ts); ya = yt / yn
    yw = sum(1 for t in ts if t["return"] > 0)
    print(f"\n--- {year}年 | {yn}筆 | 平均{ya:.2f}% | 總{yt:.2f}% | 勝率{yw/yn*100:.1f}% ---")
    for i, t in enumerate(ts):
        e = "V" if t["return"] > 0 else "X"
        tk = t["ticker"].replace(".TW", "").replace(".TWO", "")
        print(f" {i+1:2d}. {e} {t['name']}({tk}) | {t['buy_date'][5:]}>{t['sell_date'][5:]} | {t['buy_price']}>{t['sell_price']} | {t['return']:+.2f}% | {t['days']}天 | {t['reason']}")

input("\n按 Enter 結束...")
