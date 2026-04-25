"""
動態股票池回測 v5 — 對齊 Web Tab 3（自動全 cache + Gist live 89.905）

v3 → v4 改動：TARGET_DAYS 自動偵測 1500/1200/900（跟 daily_scan 一致）
v4 → v5 改動（2026-04-25 William 抓到本地 best_strategy.json 是 189 分舊策略）：
  - 改從 Gist c1bef892 即時抓 89.905（跟 Web App / GitHub Actions / verify_* 一致）
  - 不再讀本地 best_strategy.json（會 stale，misleading）

注意：cpu_replay 內部固定 day 60 warmup，所以前 60 天不交易（不是 reverse WF）
     precompute 印「反向 WF」是評估 gate 用的標籤，不影響 cpu_replay 全期模擬
"""
import numpy as np
import json, os, sys, types
import urllib.request
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name


def fetch_gist_strategy():
    """從 GPU Gist 即時抓 live 策略（89.905），不用本地 stale 檔"""
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30
    )
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s


data = download_data()
strategy = fetch_gist_strategy()
p = strategy.get("params", strategy)
print(f"v{strategy.get('version','?')} | {strategy.get('score',0):.3f}分（從 Gist live 抓）")

# v4: 自動偵測 TARGET_DAYS，全 cache 跑，對齊 Tab 3 的 daily_scan
_lens = [len(v) for v in data.values()]
if sum(1 for l in _lens if l >= 1500) >= 500:
    TARGET_DAYS = 1500
elif sum(1 for l in _lens if l >= 1200) >= 800:
    TARGET_DAYS = 1200
else:
    TARGET_DAYS = 900
print(f"自動選 TARGET_DAYS = {TARGET_DAYS}（跟 daily_scan / Tab 3 一致）")
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
pre = precompute(data)
ns, nd = pre["n_stocks"], pre["n_days"]
dates = pre["dates"]
print(f"股票：{ns} 檔 | 天數：{nd}")
print(f"期間：{dates[0].date()} ~ {dates[-1].date()}")

# 直接用 cpu_replay（跟 GPU kernel 完全一致的 Python 版）
trades = cpu_replay(pre, p)
trades.sort(key=lambda x: x["buy_date"])
completed = [t for t in trades if t.get("reason") != "持有中"]
holding = [t for t in trades if t.get("reason") == "持有中"]
trades = completed
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

if holding:
    print(f"\n=== 持有中（{len(holding)} 檔）===")
    for t in holding:
        tk = t["ticker"].replace(".TW","").replace(".TWO","")
        print(f"  {t['name']}({tk}) | {t['buy_date'][5:]} 買 {t['buy_price']} | 現 {t['sell_price']} | {t['return']:+.2f}%")
else:
    print(f"\n⚠️ 沒有持有中！推 Web 前要手動補。")

# === 對比 Tab 3 ===
print(f"\n{'='*60}")
print(f" 對比 Web Tab 3（backtest_results.json）")
print(f"{'='*60}")
print(f" Tab 3 (4/24)  ：136 筆 / +2217.5% / 勝率 69.9% / 持倉 達邁+聯茂")
print(f" 本次 realistic：{n} 筆 / {total:+.1f}% / 勝率 {wins/n*100:.1f}% / 持倉 {len(holding)} 檔")
print(f"")
print(f" 偏差說明：")
print(f"   - Tab 3 是 daily_scan 每天累積的「真實上線軌跡」（凍結）")
print(f"   - realistic 是用「今天 cache」從頭重模擬的「假設情境」")
print(f"   - 兩者數字可能差 5-15%，越接近今天偏差越大（cache drift）")
print(f"   - Tab 3 才是實盤戰況真相")

input("\n按 Enter 結束...")
