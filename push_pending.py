"""審核後手動推 pending 策略到 GPU Gist。
用法：python push_pending.py

讀 pending_push.json，顯示摘要，按 y 確認後推到 Gist（Web/daily_scan 會讀到）。
"""
import os, json, sys, urllib.request

PENDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_push.json")
GIST_ID = os.environ.get("GIST_ID", "c1bef892d33589baef2142ce250d18c2")
GH_TOKEN = os.environ.get("GH_TOKEN", "")

if not os.path.exists(PENDING):
    print("❌ 沒有 pending_push.json，GPU 還沒找到突破")
    sys.exit(1)
if not GH_TOKEN:
    print("❌ 環境變數 GH_TOKEN 沒設")
    sys.exit(1)

d = json.load(open(PENDING, encoding="utf-8"))
bt = d.get("backtest", {})
p = d.get("params", {})
td = d.get("trade_details", [])

print("=" * 50)
print(f"  📋 待推策略摘要")
print("=" * 50)
print(f"  Score       : {d.get('score')}")
print(f"  Source      : {d.get('source')}")
print(f"  Updated     : {d.get('updated_at')}")
print()
print(f"  全期筆數    : {bt.get('total_trades')}")
print(f"  總報酬      : {bt.get('total_return')}%")
print(f"  平均        : {bt.get('avg_return')}%")
print(f"  勝率        : {bt.get('win_rate')}%")
print()
print(f"  關鍵參數：")
for k in ["stop_loss", "take_profit", "trailing_stop", "hold_days",
          "use_breakeven", "breakeven_trigger", "buy_threshold",
          "above_ma60", "w_new_high", "max_positions"]:
    if k in p:
        print(f"    {k:25} = {p[k]}")
print()

# 檢查含持有中幾筆
_holding = [t for t in td if t.get("reason") == "持有中"]
print(f"  持有中：{len(_holding)} 筆")
for h in _holding:
    print(f"    {h.get('ticker')} {h.get('name','')} 買{h.get('buy_date')} @{h.get('buy_price')}")
print()
print("⚠️ 推 Gist 會直接影響 Web 和 daily_scan（實盤交易）")
print("=" * 50)

ans = input("確定推到 Gist？[y/N]: ").strip().lower()
if ans != "y":
    print("取消")
    sys.exit(0)

payload = json.dumps({"files": {"best_strategy.json": {"content": open(PENDING, encoding="utf-8").read()}}}).encode()
req = urllib.request.Request(
    f"https://api.github.com/gists/{GIST_ID}",
    data=payload, method="PATCH",
    headers={"Authorization": f"token {GH_TOKEN}", "Content-Type": "application/json"}
)
r = urllib.request.urlopen(req, timeout=30)
print(f"✅ 推送成功：{r.status}")
dst = PENDING + ".pushed"
# Windows os.rename 不覆蓋既有檔案 → 用 os.replace（跨平台支援覆蓋）
os.replace(PENDING, dst)
print(f"pending_push.json → pending_push.json.pushed")
