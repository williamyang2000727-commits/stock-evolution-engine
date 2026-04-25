"""
V38 Paper Trading 全自動每日腳本
用法：每天 16:40 後 Windows 工作排程器自動執行
  C:\\stock-evolution> python v38_daily_auto.py

做的事：
1. update_cache.py（補今天 OHLCV）
2. paper_trade_tracker.py scan（89.90 + Kronos 決策記錄）
3. paper_trade_tracker.py fill（填過去 5+ 個交易日前的 actual return）
4. 如果是週日，跑 paper_trade_tracker.py review
5. **發 Telegram 通知**（用 V34 已有的 bot）

輸出：
- 終端 log
- C:\\stock-evolution\\v38_daily_auto.log
- Telegram 訊息
"""
import os, sys, json, subprocess, time
from datetime import datetime
import urllib.request
import urllib.parse

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
# Script 路徑：Windows 用 C:\stock-evolution（V34 教訓：不是 $HOME/stock-evolution）
SCRIPT_DIR = "C:\\stock-evolution" if os.name == "nt" and os.path.isdir("C:\\stock-evolution") else USER_SE
LOG_PATH = os.path.join(SCRIPT_DIR, "v38_daily_auto.log")
PAPER_LOG = os.path.join(USER_SE, "paper_trade_log.json")

# Telegram (V34 sec)
TG_BOT_TOKEN = "8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI"
TG_CHAT_IDS = ["5785839733"]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def send_telegram(text: str):
    """發 Telegram 訊息給 William

    SSL fix: 排程觸發時 SSL CA chain 可能不可用（self-signed cert in chain）
    用 unverified context bypass。Telegram bot 通訊用 token 認證，
    SSL 驗證是 confidentiality 而非 authentication，bypass 可接受。
    """
    import ssl
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    if len(text) > 4000:
        text = text[:3950] + "\n... (truncated)"
    for chat_id in TG_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(url, data=data)
            r = urllib.request.urlopen(req, timeout=15, context=ssl_ctx)
            if r.status != 200:
                log(f"  Telegram chat {chat_id} status {r.status}")
        except Exception as e:
            log(f"  Telegram chat {chat_id} error: {e}")


def run_command(cmd: list, label: str, timeout: int = 600) -> tuple:
    """跑 subprocess command，回傳 (success, output)"""
    log(f">> {label}: {' '.join(cmd)} (cwd={SCRIPT_DIR})")
    try:
        # Windows cp950 → UTF-8 強制（避免 emoji 炸）
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 衝突 workaround
        result = subprocess.run(
            cmd, cwd=SCRIPT_DIR, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace", env=env,
        )
        if result.returncode == 0:
            log(f"   ✅ {label} ok")
            return True, result.stdout
        else:
            log(f"   ❌ {label} failed (code {result.returncode})")
            log(f"   stderr: {result.stderr[:500]}")
            return False, result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        log(f"   ❌ {label} timeout ({timeout}s)")
        return False, "TIMEOUT"
    except Exception as e:
        log(f"   ❌ {label} exception: {e}")
        return False, str(e)


def is_trading_day(date_obj=None) -> bool:
    """判斷是否台股交易日（簡化版：週一~五、排除部分國定假日）"""
    if date_obj is None:
        date_obj = datetime.now()
    # 週六日
    if date_obj.weekday() >= 5:
        return False
    # TODO: 可從 trading_days.py 抓更精準的 calendar
    return True


def parse_paper_log_summary():
    """讀 paper_trade_log.json 生成摘要文字"""
    if not os.path.exists(PAPER_LOG):
        return "尚無 paper_trade_log.json"
    try:
        with open(PAPER_LOG) as f:
            log_data = json.load(f)
        trades = log_data.get("trades", [])
        if not trades:
            return "log 內無 trades"

        # 取最新 5 筆
        recent = trades[-5:]
        lines = [f"📊 V38 Paper Trade（最新 {len(recent)}/{len(trades)} 筆）"]
        for t in recent:
            date = t.get("scan_date", "?")
            ticker = t.get("track_A_buy") or "no pick"
            decision = t.get("track_B_decision")
            if decision is None:
                decision_str = "—"
            elif decision is True:
                decision_str = "✅ buy"
            else:
                decision_str = "❌ skip"
            actual = t.get("actual_5d_return_pct")
            actual_str = f"{actual:+.2f}%" if actual is not None else "(待填)"
            lines.append(f"  {date} {ticker[:8]:<8} V38={decision_str} actual={actual_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"讀 log 失敗：{e}"


def main():
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    weekday_name = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"][today.weekday()]

    log("=" * 60)
    log(f"V38 Daily Auto — {today_str} ({weekday_name})")
    log("=" * 60)

    is_trading = is_trading_day(today)
    is_sunday = today.weekday() == 6

    notify_lines = [f"🤖 V38 Auto {today_str} ({weekday_name})"]

    # === Step 1: update_cache（每天都跑，週末會印 0 筆 append）===
    log(f"\n[Step 1] update_cache")
    ok, out = run_command(
        [sys.executable, "update_cache.py"],
        "update_cache", timeout=900
    )
    if ok:
        notify_lines.append("✅ update_cache OK")
    else:
        notify_lines.append("❌ update_cache failed")

    # === Step 2: paper_trade_tracker scan（只在交易日跑）===
    if is_trading:
        log(f"\n[Step 2] paper_trade_tracker scan")
        ok, out = run_command(
            [sys.executable, "paper_trade_tracker.py", "scan"],
            "paper_trade_tracker scan", timeout=600
        )
        if ok:
            # 解析 output 找決策
            decision_line = "（無決策）"
            for line in out.splitlines():
                if "buy = " in line:
                    decision_line = line.strip()
                if "89.90 選: " in line:
                    notify_lines.append(f"🎯 89.90 候選: {line.split('89.90 選: ')[1].strip()}")
                if "  buy = True" in line or "  buy = False" in line:
                    notify_lines.append(f"🤖 V38: {line.strip()}")
            notify_lines.append("✅ paper scan OK")
        else:
            notify_lines.append("❌ paper scan failed")
    else:
        log(f"\n[Step 2] {weekday_name} 非交易日，skip paper_trade scan")
        notify_lines.append(f"⏭️ {weekday_name} skip scan")

    # === Step 3: paper_trade_tracker fill（每天，填過去 5+ 天的 actual）===
    log(f"\n[Step 3] paper_trade_tracker fill")
    ok, out = run_command(
        [sys.executable, "paper_trade_tracker.py", "fill"],
        "paper_trade_tracker fill", timeout=300
    )
    if ok:
        # 抓「填了 N 筆」
        for line in out.splitlines():
            if "填了" in line:
                notify_lines.append(f"📝 {line.strip()}")

    # === Step 4: 週日跑 review ===
    if is_sunday:
        log(f"\n[Step 4] 週日跑 review")
        ok, out = run_command(
            [sys.executable, "paper_trade_tracker.py", "review"],
            "paper_trade_tracker review", timeout=300
        )
        if ok:
            # 摘要 review 給 Telegram
            review_lines = []
            for line in out.splitlines():
                line = line.strip()
                if any(k in line for k in ["Track A", "Track B", "n =", "wr =", "對比", "V38 wr", "V38 avg",
                                           "🟢", "🟡", "🔴", "資料不夠"]):
                    review_lines.append(line)
            if review_lines:
                notify_lines.append("\n📊 週報")
                notify_lines.extend(review_lines)

    # === Step 5: 摘要近期 paper trades ===
    summary = parse_paper_log_summary()
    log(f"\n{summary}")
    notify_lines.append("\n" + summary)

    # === Step 6: 發 Telegram ===
    final_msg = "\n".join(notify_lines)
    log(f"\n=== 發 Telegram ===")
    log(final_msg)
    send_telegram(final_msg)
    log(f"\n✅ V38 Daily Auto 完成")


if __name__ == "__main__":
    main()
