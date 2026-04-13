"""
Run realistic backtest on GPU machine → push results to Web App Gist.
Usage:
  1. (Get-Item $HOME\stock-evolution\stock_data_cache.pkl).LastWriteTime = Get-Date
  2. python backtest_to_web.py
"""
import numpy as np
import json, os, sys, types

# Mock CuPy (not needed for CPU backtest)
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name

# === Load strategy ===
with open("best_strategy.json", encoding="utf-8") as f:
    strategy = json.load(f)
p = strategy["params"]
print(f"Strategy: v{strategy.get('version','?')} | score={strategy.get('score',0):.2f}")

# === Load data ===
data = download_data()
data = {k: v for k, v in data.items() if len(v) >= 900}
print(f"Stocks with 900+ days: {len(data)}")

# === Precompute + Run ===
pre = precompute(data)
ns, nd = pre["n_stocks"], pre["n_days"]
dates = pre["dates"]
print(f"Period: {dates[0].date()} ~ {dates[-1].date()} ({nd} days)")
print(f"Running cpu_replay...")

trades = cpu_replay(pre, p)
trades.sort(key=lambda x: x["buy_date"])
n = len(trades)
if n == 0:
    print("No trades!"); sys.exit(1)

total = sum(t["return"] for t in trades)
avg = total / n
wins = sum(1 for t in trades if t["return"] > 0)
print(f"\nTrades: {n} | Avg: {avg:.2f}% | Total: {total:.2f}% | Win Rate: {wins/n*100:.1f}%")

# === Format for Web ===
web_trades = []
for t in trades:
    web_trades.append({
        "ticker": t.get("ticker", ""),
        "name": t.get("name", ""),
        "buy_price": round(t.get("buy_price", 0), 2),
        "sell_price": round(t.get("sell_price", 0), 2),
        "hold_days": t.get("days", 0),
        "return_pct": round(t.get("return", 0), 1),
        "reason": t.get("reason", ""),
        "buy_date": t.get("buy_date", ""),
        "sell_date": t.get("sell_date", ""),
    })

rets = [t["return_pct"] for t in web_trades]
wins_r = [r for r in rets if r > 0]
losses_r = [r for r in rets if r <= 0]

stats = {
    "total_trades": len(rets),
    "total_return_pct": round(sum(rets), 1),  # Simple sum (matching GPU)
    "win_rate": round(len(wins_r) / len(rets) * 100, 1) if rets else 0,
    "avg_return": round(np.mean(rets), 1) if rets else 0,
    "avg_win": round(np.mean(wins_r), 1) if wins_r else 0,
    "avg_loss": round(np.mean(losses_r), 1) if losses_r else 0,
    "max_win": round(max(rets), 1) if rets else 0,
    "max_loss": round(min(rets), 1) if rets else 0,
    "avg_hold_days": round(np.mean([t["hold_days"] for t in web_trades]), 1),
    "start_date": str(dates[0].date()),
    "end_date": str(dates[-1].date()),
    "total_days": nd,
    "strategy_version": strategy.get("version", "?"),
    "strategy_score": strategy.get("score", "?"),
}

print(f"\nCompounded Return: {stats['total_return_pct']:,.1f}%")
print(f"Win Rate: {stats['win_rate']:.1f}% | Avg: {stats['avg_return']:+.1f}%")

# === Push to Gist ===
content = json.dumps({"stats": stats, "trades": web_trades}, ensure_ascii=False, indent=2)
print(f"\nPushing to Gist ({len(content)//1024} KB)...")

GIST_ID = "e1159b02a87d3c6ee9f33fb9ef61bb80"
import urllib.request

token = None
try:
    import subprocess
    token = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
except:
    pass
if not token:
    token = input("Enter GitHub token: ").strip()

try:
    req = urllib.request.Request(
        f"https://api.github.com/gists/{GIST_ID}",
        data=json.dumps({"files": {"backtest_results.json": {"content": content}}}).encode(),
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
        method="PATCH",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"Push: OK!")
    print(f"Open web app to see results.")
except Exception as e:
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Push failed: {e}")
    print(f"Saved to backtest_results.json")

input("\nPress Enter to exit...")
