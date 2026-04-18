"""把 pending_push.json（或 .pushed）的原始戰績直接推 Data Gist — 不重跑 cpu_replay。

為什麼不重跑：Windows cache 每 append 一天，yfinance adjusted price 就可能微變，
cpu_replay 結果也會微變。pending 裡的 trade_details 是 GPU 當下找到的戰績（84 筆
1475% 69%），應該當「策略真相」推 Web，而不是用當下 cache 重算的變形版。

用法：python pending_to_web.py
"""
import json, os, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for fname in ["pending_push.json", "pending_push.json.pushed"]:
    path = os.path.join(HERE, fname)
    if os.path.exists(path):
        PENDING = path
        break
else:
    print("❌ 找不到 pending_push.json 或 .pushed")
    sys.exit(1)

print(f"讀 pending：{os.path.basename(PENDING)}")
d = json.load(open(PENDING, encoding='utf-8'))
trades = d.get("trade_details", [])
if not trades:
    print("❌ pending 裡沒 trade_details")
    sys.exit(1)

# 格式化為 Web 期望的格式（跟 backtest_to_web.py 一樣）
web_trades = []
for t in trades:
    wt = {
        "ticker": t.get("ticker", ""),
        "name": t.get("name", ""),
        "buy_price": round(t.get("buy_price", 0), 2),
        "sell_price": round(t.get("sell_price", 0), 2),
        "hold_days": t.get("days", 0),
        "return_pct": round(t.get("return", 0), 1),
        "reason": t.get("reason", ""),
        "buy_date": t.get("buy_date", ""),
        "sell_date": t.get("sell_date", ""),
    }
    if t.get("reason") == "持有中":
        wt["peak_price"] = round(t.get("peak_price", t.get("buy_price", 0)), 2)
    web_trades.append(wt)

completed = [t for t in web_trades if t.get("reason") != "持有中"]
holding = [t for t in web_trades if t.get("reason") == "持有中"]
rets = [t["return_pct"] for t in completed]
wins_r = [r for r in rets if r > 0]
losses_r = [r for r in rets if r <= 0]

# end_date = 資料最後一天（GPU cache 的 last date），不是最後賣出日
# 推法：從持倉 buy_date + 持有天數（交易日）推導
# 若沒持倉，用最後完成交易的賣出日 + 持倉空出後的 safe margin
import pandas as pd
candidates = []
for h in holding:
    try:
        bd = pd.Timestamp(h["buy_date"])
        # 持有 days 是交易日數；簡化用 days × 1.4 當日曆日推算（保守）
        approx_last = bd + pd.Timedelta(days=int(h["hold_days"] * 1.4 + 1))
        candidates.append(str(approx_last.date()))
    except Exception:
        pass
for c in completed:
    if c.get("sell_date"):
        candidates.append(c["sell_date"])
end_date = max(candidates) if candidates else ""
# 不能超過今天
_today_str = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
if end_date > _today_str:
    end_date = _today_str
start_date = min((t.get("buy_date", "") for t in web_trades), default="")

stats = {
    "total_trades": len(completed),
    "total_return_pct": round(sum(rets), 1),
    "win_rate": round(len(wins_r) / len(rets) * 100, 1) if rets else 0,
    "avg_return": round(np.mean(rets), 1) if rets else 0,
    "avg_win": round(np.mean(wins_r), 1) if wins_r else 0,
    "avg_loss": round(np.mean(losses_r), 1) if losses_r else 0,
    "max_win": round(max(rets), 1) if rets else 0,
    "max_loss": round(min(rets), 1) if rets else 0,
    "avg_hold_days": round(np.mean([t["hold_days"] for t in completed]), 1) if completed else 0,
    "start_date": start_date,
    "end_date": end_date,
    "total_days": 900,
    "strategy_version": d.get("mode", d.get("source", "?")),
    "strategy_score": d.get("score", 0),
}

print()
print(f"完成交易：{stats['total_trades']} 筆")
print(f"總報酬：  {stats['total_return_pct']}%")
print(f"勝率：    {stats['win_rate']}%")
print(f"平均：    {stats['avg_return']}%")
print(f"持倉中：  {len(holding)} 檔")
for h in holding:
    print(f"  {h['ticker']} {h.get('name','')} 買 {h['buy_date']} @ {h['buy_price']}")
print(f"end_date：{end_date}")

content = json.dumps({"stats": stats, "trades": web_trades}, ensure_ascii=False, indent=2)
print(f"\nPushing to Data Gist ({len(content)//1024} KB)...")

GIST_ID = "e1159b02a87d3c6ee9f33fb9ef61bb80"

# 取 token：gh auth → 環境變數 → 手動輸入
token = None
try:
    import subprocess
    token = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
except:
    pass
if not token:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN_GIST")
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
    print(f"✅ Push OK ({resp.status})")
    print(f"Web Tab 3 重整應顯示：總報酬 {stats['total_return_pct']}% / 勝率 {stats['win_rate']}% / 持倉 {len(holding)} 檔")
except Exception as e:
    print(f"❌ Push failed: {e}")
    backup_path = os.path.join(HERE, "backtest_results_pending.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已存本地：{backup_path}")
    sys.exit(1)

# 自動觸發 GitHub Actions daily-scan 延續到今天（不用人工去網頁按 Run workflow）
print()
print("觸發 daily-scan workflow 延續到今天...")
try:
    import subprocess
    result = subprocess.run(
        ["gh", "workflow", "run", "daily-scan.yml",
         "--repo", "williamyang2000727-commits/stock-web-app"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        print("✅ daily-scan 已自動觸發")
        print("   5-10 分鐘後 Web Tab 3 會自動更新 end_date 到最新交易日")
        print("   監看：https://github.com/williamyang2000727-commits/stock-web-app/actions")
    else:
        raise RuntimeError(result.stderr or "gh command failed")
except Exception as _e:
    print(f"⚠️ 自動觸發失敗（{_e}）")
    print("手動觸發（二擇一）：")
    print("  A. 網頁：https://github.com/williamyang2000727-commits/stock-web-app/actions/workflows/daily-scan.yml")
    print("  B. 指令：gh workflow run daily-scan.yml --repo williamyang2000727-commits/stock-web-app")
