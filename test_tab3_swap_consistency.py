r"""驗證 Tab 3 換股提醒一致性

問題：rebuild_tab3 蓋掉 backtest_results.json 後，daily_scan 寫的 pending_sells
是針對 daily_scan 認為的持倉，可能跟 Tab 3 顯示的持倉不一致

Test：
  1. backtest_results.json 的「持有中」（rebuild_tab3 寫的）
  2. scan_results.json 的 pending_sells / pending_buy（daily_scan 寫的）
  3. 兩邊 ticker 是否對得起來
"""
import os, json, urllib.request

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
DATA_GIST = "e1159b02a87d3c6ee9f33fb9ef61bb80"


def fetch(fname):
    req = urllib.request.Request(
        f"https://api.github.com/gists/{DATA_GIST}",
        headers={"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
    )
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    return json.loads(d["files"][fname]["content"])


bt = fetch("backtest_results.json")
scan = fetch("scan_results.json")

# 1. backtest_results 的持有中
holdings_bt = [t for t in bt.get("trades", []) if t.get("reason") == "持有中"]
print("=== backtest_results.json (rebuild_tab3 寫的) ===")
print(f"持有中 {len(holdings_bt)} 檔:")
for h in holdings_bt:
    print(f"  {h.get('name', '')} ({h.get('ticker', '')}) buy {h.get('buy_date', '')} @ {h.get('buy_price', 0)}")

bt_tickers = {h.get("ticker") for h in holdings_bt}

# 2. scan_results 的 pending
print(f"\n=== scan_results.json (daily_scan 寫的) ===")
print(f"date: {scan.get('date', '?')}")
pending_sells = scan.get("pending_sells", []) or []
pending_buy = scan.get("pending_buy")
print(f"pending_sells {len(pending_sells)} 檔:")
for ps in pending_sells:
    print(f"  賣 {ps.get('name', '')} ({ps.get('ticker', '')}) reason={ps.get('reason', '')}")
print(f"pending_buy: {pending_buy}")

ps_tickers = {ps.get("ticker") for ps in pending_sells}

# 3. 一致性檢查
print(f"\n=== 一致性檢查 ===")
inconsistent = ps_tickers - bt_tickers
if inconsistent:
    print(f"❌ pending_sells 裡有 ticker 不在 backtest 持倉內:")
    for tk in inconsistent:
        print(f"   {tk}")
    print("→ Tab 3 會顯示「持倉聯茂+希華」但 pending sell 卻是達邁 → 用戶混亂")
else:
    print(f"✅ pending_sells 全在 backtest 持倉內")

# 還要檢查 holdings_status
hs = scan.get("holdings_status", [])
print(f"\nholdings_status (Tab 2 用): {len(hs)} 檔")
for h in hs:
    print(f"  {h}")
