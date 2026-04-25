r"""
重建 Tab 3 backtest_results.json — 用 cpu_replay 真實結果蓋掉失真版

問題：
  Tab 3 backtest_results.json 是 daily_scan 80 天 cold start 算的失真版
  跟 cpu_replay (1500 天完整) 結果不一致

修法：
  1. 用 download_data + 1500 天 cache 跑 cpu_replay
  2. 重建 trades + stats
  3. 推 Data Gist 蓋掉 backtest_results.json

讓 Tab 3 顯示真實正確軌跡（之後 daily_scan 再從這基礎累積）
"""
import os, sys, json, types, urllib.request, base64, time
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "stock-evolution"))
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from gpu_cupy_evolve import precompute, cpu_replay, download_data

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
if not GH_TOKEN:
    print("❌ 請先設環境變數：$env:GH_TOKEN = 'ghp_xxx...'")
    sys.exit(1)
DATA_GIST = "e1159b02a87d3c6ee9f33fb9ef61bb80"
GPU_GIST = "c1bef892d33589baef2142ce250d18c2"


def fetch_gist(gist_id, fname):
    r = urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/gists/{gist_id}",
                                headers={"Authorization": f"token {GH_TOKEN}"}),
        timeout=30,
    )
    d = json.loads(r.read())
    return json.loads(d["files"][fname]["content"])


def write_gist(gist_id, fname, content_obj):
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        method="PATCH",
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "files": {fname: {"content": json.dumps(content_obj, ensure_ascii=False)}}
        }).encode(),
    )
    r = urllib.request.urlopen(req, timeout=120)
    return r.status


# 1. 載 cache + strategy
print("[1/5] 載 cache + 89.905 策略 ...")
data = download_data()
strategy = fetch_gist(GPU_GIST, "best_strategy.json")
p = strategy.get("params", strategy)
print(f"  Cache: {len(data)} tickers / Strategy: {strategy.get('score', 0):.3f}")

# 2. precompute + cpu_replay 跑全期
TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
print(f"\n[2/5] precompute + cpu_replay (TARGET={TARGET}, {len(data_t)} stocks)...")
pre = precompute(data_t)
trades = cpu_replay(pre, p)
trades.sort(key=lambda t: t.get("buy_date", ""))
print(f"  共 {len(trades)} trades")

# 3. 轉成 Tab 3 期望的格式
# Tab 3 用：return_pct (不是 return), hold_days (不是 days)
print(f"\n[3/5] 轉換為 Tab 3 格式 ...")
tab3_trades = []
for t in trades:
    item = {
        "ticker": t.get("ticker", ""),
        "name": t.get("name", ""),
        "buy_date": t.get("buy_date", ""),
        "sell_date": t.get("sell_date", ""),
        "buy_price": float(t.get("buy_price", 0)),
        "sell_price": float(t.get("sell_price", 0)),
        "return_pct": float(t.get("return", 0)),
        "hold_days": int(t.get("days", 0)),
        "reason": t.get("reason", ""),
    }
    if "peak_price" in t:
        item["peak_price"] = float(t["peak_price"])
    tab3_trades.append(item)

# 4. 算 stats（mirror daily_scan line 469-477）
completed = [t for t in tab3_trades if t.get("reason") != "持有中"]
holding = [t for t in tab3_trades if t.get("reason") == "持有中"]
rets = [t["return_pct"] for t in completed]
wins = [r for r in rets if r > 0]
losses = [r for r in rets if r <= 0]

# 起始/結束日
all_buy_dates = [t["buy_date"] for t in tab3_trades if t.get("buy_date")]
all_dates_in_pre = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in pre["dates"]]
start_date = all_dates_in_pre[60] if len(all_dates_in_pre) > 60 else all_dates_in_pre[0]  # warmup 後
end_date = all_dates_in_pre[-1]

stats = {
    "total_trades": len(completed),
    "total_return_pct": round(sum(rets), 1) if rets else 0,
    "win_rate": round(len(wins) / len(rets) * 100, 1) if rets else 0,
    "avg_return": round(sum(rets) / len(rets), 1) if rets else 0,
    "avg_win": round(sum(wins) / len(wins), 1) if wins else 0,
    "avg_loss": round(sum(losses) / len(losses), 1) if losses else 0,
    "max_win": round(max(rets), 1) if rets else 0,
    "max_loss": round(min(rets), 1) if rets else 0,
    "avg_hold_days": round(sum(t.get("hold_days", 0) for t in completed) / len(completed), 1) if completed else 0,
    "start_date": start_date,
    "end_date": end_date,
    "total_days": int(pre["n_days"]),
    "strategy_version": "auto",
    "strategy_score": float(strategy.get("score", 89.905)),
}

print(f"  完成 {stats['total_trades']} 筆 / 持有 {len(holding)} 檔")
print(f"  總報酬 {stats['total_return_pct']}% / 勝率 {stats['win_rate']}%")
print(f"  期間 {stats['start_date']} ~ {stats['end_date']}")

print(f"\n  持有中：")
for h in holding:
    print(f"    {h['name']} ({h['ticker']}) buy {h['buy_date']} @ {h['buy_price']}")

# 5. 推 Data Gist 蓋掉 backtest_results.json
print(f"\n[4/5] 比對舊版 Tab 3 數據 ...")
try:
    old_bt = fetch_gist(DATA_GIST, "backtest_results.json")
    old_stats = old_bt.get("stats", {})
    print(f"  舊版：{old_stats.get('total_trades')} 筆 / {old_stats.get('total_return_pct')}% / wr {old_stats.get('win_rate')}%")
    print(f"  新版：{stats['total_trades']} 筆 / {stats['total_return_pct']}% / wr {stats['win_rate']}%")
    old_hold = [t for t in old_bt.get("trades", []) if t.get("reason") == "持有中"]
    print(f"  舊版持有：{[(h.get('name'), h.get('ticker')) for h in old_hold]}")
    print(f"  新版持有：{[(h['name'], h['ticker']) for h in holding]}")
except Exception as e:
    print(f"  讀舊版失敗: {e}")

new_bt = {"stats": stats, "trades": tab3_trades}

print(f"\n[5/5] 推 Data Gist 蓋掉 backtest_results.json ...")
try:
    status = write_gist(DATA_GIST, "backtest_results.json", new_bt)
    print(f"  ✅ Status {status} - Tab 3 已更新為 cpu_replay 真實結果")
    print(f"  Web App 重新整理後會看到新數據")
except Exception as e:
    print(f"  ❌ Push fail: {e}")
    sys.exit(1)
