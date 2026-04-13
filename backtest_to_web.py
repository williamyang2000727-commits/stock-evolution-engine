"""
Run realistic backtest on GPU machine → push results to Web App Gist.
Usage: python backtest_to_web.py
"""
import json, os, sys, numpy as np

# === Load strategy and data ===
with open("best_strategy.json", "r") as f:
    strategy = json.load(f)
p = strategy["params"]

print(f"Strategy: v{strategy.get('version','?')} score={strategy.get('score','?')}")
print(f"Loading data...")

from gpu_cupy_evolve import cpu_replay
import pickle

cache_path = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
if not os.path.exists(cache_path):
    cache_path = "stock_data_cache.pkl"

with open(cache_path, "rb") as f:
    all_data = pickle.load(f)

# Filter >= 900 days
data = {k: v for k, v in all_data.items() if len(v) >= 900}
print(f"Stocks with 900+ days: {len(data)}")

if not data:
    data = {k: v for k, v in all_data.items() if len(v) >= 500}
    print(f"Fallback: stocks with 500+ days: {len(data)}")

# === Run cpu_replay ===
print(f"Running realistic backtest...")
result = cpu_replay(p, data)

if result is None:
    print("cpu_replay returned None!")
    sys.exit(1)

# Parse result
stats = result.get("stats", {})
trade_list = result.get("trades", [])

# If cpu_replay returns different format, adapt
if not trade_list and "yearly" in result:
    # Try to extract from result directly
    trade_list = result.get("all_trades", [])

print(f"\n=== Results ===")
print(f"Total Return: {stats.get('total_return', 'N/A')}")
print(f"Win Rate: {stats.get('win_rate', 'N/A')}")
print(f"Trades: {stats.get('total_trades', len(trade_list))}")

# Format for web
web_trades = []
for t in trade_list:
    web_trades.append({
        "ticker": t.get("ticker", ""),
        "name": t.get("name", t.get("ticker", "")),
        "buy_price": round(t.get("buy_price", 0), 2),
        "sell_price": round(t.get("sell_price", 0), 2),
        "hold_days": t.get("hold_days", 0),
        "return_pct": round(t.get("return_pct", t.get("ret", 0)), 1),
        "reason": t.get("reason", t.get("exit_reason", "")),
        "buy_date": str(t.get("buy_date", "")),
        "sell_date": str(t.get("sell_date", "")),
    })

# Compute stats if not provided
if not stats.get("total_return_pct"):
    rets = [t["return_pct"] for t in web_trades if t["reason"] != "持有中"]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    tr = 1.0
    for r in rets:
        tr *= (1 + r / 100)
    stats = {
        "total_trades": len(rets),
        "total_return_pct": round((tr - 1) * 100, 1),
        "win_rate": round(len(wins) / len(rets) * 100, 1) if rets else 0,
        "avg_return": round(np.mean(rets), 1) if rets else 0,
        "avg_win": round(np.mean(wins), 1) if wins else 0,
        "avg_loss": round(np.mean(losses), 1) if losses else 0,
        "max_win": round(max(rets), 1) if rets else 0,
        "max_loss": round(min(rets), 1) if rets else 0,
        "avg_hold_days": round(np.mean([t["hold_days"] for t in web_trades if t["reason"] != "持有中"]), 1) if web_trades else 0,
        "strategy_version": strategy.get("version", "?"),
        "strategy_score": strategy.get("score", "?"),
    }
    if web_trades:
        dates = [t["buy_date"] for t in web_trades if t["buy_date"]]
        if dates:
            stats["start_date"] = min(dates)
            stats["end_date"] = max(t.get("sell_date", "") for t in web_trades)
            stats["total_days"] = len(set(dates))

web_result = {"stats": stats, "trades": web_trades}
content = json.dumps(web_result, ensure_ascii=False, indent=2)

print(f"\nFormatted: {len(web_trades)} trades, {len(content)//1024} KB")
print(f"Total Return: {stats.get('total_return_pct', 'N/A')}%")
print(f"Win Rate: {stats.get('win_rate', 'N/A')}%")

# === Push to Gist ===
print(f"\nPushing to Web App Gist...")
import urllib.request

GIST_ID = "e1159b02a87d3c6ee9f33fb9ef61bb80"
# Read token from gist (the strategy gist has the same auth)
try:
    gist_url = f"https://api.github.com/gists/{GIST_ID}"
    # Use gh CLI token if available
    token = None
    try:
        import subprocess
        token = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
    except:
        pass

    if not token:
        token = input("Enter GitHub token (ghp_... or gho_...): ").strip()

    req = urllib.request.Request(
        gist_url,
        data=json.dumps({
            "files": {"backtest_results.json": {"content": content}}
        }).encode(),
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
        },
        method="PATCH",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"Push: {'OK' if resp.status == 200 else 'FAIL'}")
    print(f"\nDone! Open web app to see results.")
except Exception as e:
    # Save locally if push fails
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Push failed: {e}")
    print(f"Saved to backtest_results.json locally.")
    print(f"Manual push: copy content to Gist e1159b02a87d3c6ee9f33fb9ef61bb80")
