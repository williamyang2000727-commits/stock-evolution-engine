r"""
每日健康報告 — 推 Telegram 給 William 看

驗證 Web 所有資料都對：
  1. backtest_results.json 持有中 + 統計
  2. scan_results.json pending + buy_signals top 3
  3. history_cache.json 末日
  4. state Gist updated 是今日
  5. 跟昨天比對：trade 數變化、持倉變化、pending 是否合理

異常會額外推紅色警報。
"""
import os, sys, json, urllib.request, urllib.parse, ssl
from datetime import datetime, timezone, timedelta

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "5785839733")
if not TELEGRAM_BOT:
    print("⚠️ TELEGRAM_BOT_TOKEN 未設，跳過 Telegram 推播")
TW_TZ = timezone(timedelta(hours=8))

DATA_GIST = "e1159b02a87d3c6ee9f33fb9ef61bb80"
HISTORY_GIST = "572d4ca53b0bfbd37dd5485becdcce49"
STATE_GIST = "18a7270d897c8821b291cfd61796bd80"


def fetch(gist_id, fname):
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
    )
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=30).read())["files"][fname]["content"])


def telegram(msg, chat_id=None):
    if not TELEGRAM_BOT:
        return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id or TELEGRAM_CHAT,
        "text": msg[:4000],
        "parse_mode": "Markdown",
    }).encode()
    urllib.request.urlopen(urllib.request.Request(url, data=data), context=ctx, timeout=15)


# ─── 拿資料 ───
now = datetime.now(TW_TZ)
today = now.strftime("%Y-%m-%d")
errors = []
warnings = []
info = []

# 1. backtest
try:
    bt = fetch(DATA_GIST, "backtest_results.json")
    stats = bt.get("stats", {})
    holdings = [t for t in bt.get("trades", []) if t.get("reason") == "持有中"]
    pu = stats.get("pipeline_updated", "")
    if not pu.startswith(today):
        errors.append(f"❌ Tab 3 backtest 不是今日（最後 {pu[:10] or '從未'}）")
    else:
        info.append(f"✅ Tab 3: {stats.get('total_trades')} 筆 / {stats.get('total_return_pct')}% / wr {stats.get('win_rate')}%")
        for h in holdings:
            ret = (h.get("sell_price", 0) / h.get("buy_price", 1) - 1) * 100 if h.get("buy_price", 0) > 0 else 0
            info.append(f"   • {h.get('name','')} ({h.get('ticker','')}) buy {h.get('buy_date','')} @{h.get('buy_price',0)} → 現{h.get('sell_price',0)} ({ret:+.1f}%)")
except Exception as e:
    errors.append(f"❌ 讀 backtest 失敗: {e}")
    holdings = []

# 2. scan_results
try:
    scan = fetch(DATA_GIST, "scan_results.json")
    sd = scan.get("date", "")
    pending_sells = scan.get("pending_sells", []) or []
    pending_buy = scan.get("pending_buy")
    buy_signals = scan.get("buy_signals", []) or []
    if sd != today:
        warnings.append(f"⚠️ scan_results.date={sd}（不是今日 {today}）")
    if pending_sells:
        info.append(f"📤 Pending sells:")
        for ps in pending_sells:
            info.append(f"   • {ps.get('name','')} ({ps.get('ticker','')}) — {ps.get('reason','')}")
    if pending_buy:
        info.append(f"🎯 Pending buy: {pending_buy.get('name','')} ({pending_buy.get('ticker','')}) score={pending_buy.get('score',0):.0f}")
    if not pending_sells and not pending_buy:
        info.append("✋ 明日無動作（無賣出、滿倉無新買入）")
    info.append(f"📊 Top 3 達標股:")
    for s in buy_signals[:3]:
        info.append(f"   #{s.get('rank',0)} {s.get('name','')} ({s.get('ticker','')}) score={s.get('score',0):.0f}")
except Exception as e:
    errors.append(f"❌ 讀 scan_results 失敗: {e}")

# 3. history_cache
try:
    hc = fetch(HISTORY_GIST, "history_cache.json")
    hu = hc.get("updated", "")
    n_stocks = len(hc.get("stocks", {}))
    if hu != today:
        warnings.append(f"⚠️ history_cache.updated={hu}（不是今日）")
    if n_stocks < 1500:
        errors.append(f"❌ history_cache 只有 {n_stocks} stocks (< 1500)")
    info.append(f"✅ history_cache: {n_stocks} stocks, updated={hu}")
except Exception as e:
    errors.append(f"❌ 讀 history_cache 失敗: {e}")

# 4. state Gist
try:
    st = fetch(STATE_GIST, "indicator_state.json")
    su = st.get("updated", "")
    n_state = len(st.get("states", {}))
    if n_state < 1500:
        errors.append(f"❌ state Gist 只有 {n_state} ticker (< 1500)")
    info.append(f"✅ state Gist: {n_state} ticker, updated={su}")
except Exception as e:
    errors.append(f"❌ 讀 state Gist 失敗: {e}")

# 5. 異常偵測：D 跟 D-1（昨天）持倉變化 vs 昨天的 pending 是否符合
# D 16:35 跑時：今天的「持有中」= 昨天 D-1 的「持有中」+ 執行「昨天 D-1 的 pending」
# 所以今天的變化必須對應「昨天的 pending」
last_data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_health_state.json")
try:
    last = json.load(open(last_data_file)) if os.path.exists(last_data_file) else {}
except Exception:
    last = {}

today_tks = sorted(h.get("ticker", "") for h in holdings)
last_tks = sorted(last.get("holdings", []))
last_pending_sells = last.get("pending_sells_yesterday", [])
last_pending_buy = last.get("pending_buy_yesterday")

if last_tks and today_tks != last_tks:
    # 昨天的 pending 該驅動今天的變化
    expected_sells = set(last_pending_sells)
    expected_buys = {last_pending_buy} if last_pending_buy else set()
    expected_after = (set(last_tks) - expected_sells) | expected_buys
    if set(today_tks) != expected_after:
        errors.append(f"🚨 持倉變化異常！\n   昨天: {last_tks}\n   昨天 pending sells: {list(expected_sells)}\n   昨天 pending buy: {last_pending_buy}\n   預期今天: {sorted(expected_after)}\n   實際今天: {today_tks}")
    else:
        info.append(f"📊 持倉變動正常: {last_tks} → {today_tks}（執行了昨天 pending）")

# 存今日狀態（給明天比對用）
try:
    today_pending_sells_tks = [ps.get("ticker", "") for ps in (scan.get("pending_sells", []) or [])]
    today_pending_buy_tk = pending_buy.get("ticker") if pending_buy else None
    json.dump({
        "holdings": today_tks,
        "pending_sells_yesterday": today_pending_sells_tks,
        "pending_buy_yesterday": today_pending_buy_tk,
        "date": today,
    }, open(last_data_file, "w"))
except Exception:
    pass
# 存今日持倉（給明天比對用）
try:
    json.dump({"holdings": today_tks, "date": today}, open(last_holdings_file, "w"))
except Exception:
    pass

# 計算今天該做什麼
todo_list = []
if pending_sells:
    for ps in pending_sells:
        todo_list.append(f"📤 D+1 09:00 賣 {ps.get('name','')} ({ps.get('ticker','')})")
if pending_buy:
    todo_list.append(f"🎯 D+1 13:25 前買 {pending_buy.get('name','')} ({pending_buy.get('ticker','')})")

# ─── 組訊息 ───
if errors:
    msg = f"🚨 *Pipeline 異常 {today}*\n\n"
    msg += "\n".join(errors)
    if warnings:
        msg += "\n\n⚠️ Warnings:\n" + "\n".join(warnings)
    msg += "\n\n*🛠️ 你要做*：\n手動修復 `python auto_daily_pipeline.py --force`"
elif warnings:
    msg = f"⚠️ *Pipeline 警告 {today}*\n\n"
    msg += "\n".join(warnings)
    msg += "\n\n" + "\n".join(info)
    if todo_list:
        msg += f"\n\n*🎯 D+1 你要做*：\n" + "\n".join(todo_list)
        msg += "\n\n下單後到 Tab 2 更新持倉"
else:
    msg = f"✅ *每日健康報告 {today}*\n\n"
    msg += "\n".join(info)
    if todo_list:
        msg += f"\n\n*🎯 D+1 你要做*：\n" + "\n".join(todo_list)
        msg += "\n\n下單後記得到 Tab 2 更新持倉"
    else:
        msg += f"\n\n_今天無動作，可以放心睡覺_ 💤"

print(msg)
try:
    telegram(msg)
    print("\n📱 已推 Telegram (William)")
except Exception as e:
    print(f"\n❌ Telegram fail: {e}")

# ─── 訂閱者真實持倉警報（多用戶）───
# scan_results.user_pending_sells 由 daily_scan step 7b 寫入
# 每個 user 真實持倉觸發 5 條 sell_rules 任一 → 推給該 user（有 chat_id）或併入 William 總機訊息
try:
    user_pending = (scan or {}).get("user_pending_sells", {}) or {}
    portfolios = fetch(DATA_GIST, "portfolios.json") if user_pending else {}
    william_summary_lines = []
    for uname, signals in user_pending.items():
        if not signals:
            continue
        u_chat = (portfolios.get(uname, {}) or {}).get("telegram_chat_id", "")
        u_lines = [f"🚨 *持倉警報 {today}* — {uname}"]
        for s in signals:
            u_lines.append(
                f"📤 *{s.get('name','')}* ({s.get('ticker','')})\n"
                f"   買入 ${s.get('buy_price',0)} → 現 ${s.get('current_price',0)} ({s.get('return_pct',0):+.2f}%)\n"
                f"   持有 {s.get('days_held',0)} 天 ｜ {s.get('reason','')}"
            )
        u_lines.append("\n*🎯 D+1 09:00 開盤賣出*")
        u_msg = "\n\n".join(u_lines)
        if u_chat:
            try:
                telegram(u_msg, chat_id=u_chat)
                print(f"📱 已推 Telegram ({uname} → {u_chat})")
            except Exception as e:
                print(f"❌ Telegram fail ({uname}): {e}")
                william_summary_lines.append(f"⚠️ {uname} 推送失敗（chat_id={u_chat}）：{e}")
                william_summary_lines.append(u_msg)
        else:
            william_summary_lines.append(f"📋 {uname}（無 chat_id，請手動轉達）：\n{u_msg}")
    if william_summary_lines:
        wm = f"📊 *訂閱者警報摘要 {today}*\n\n" + "\n\n---\n\n".join(william_summary_lines)
        try:
            telegram(wm)
            print(f"📱 已推 William 訂閱者摘要（{len(william_summary_lines)} 個）")
        except Exception as e:
            print(f"❌ William 摘要推送失敗: {e}")
except Exception as e:
    print(f"❌ 訂閱者警報處理失敗: {e}")
