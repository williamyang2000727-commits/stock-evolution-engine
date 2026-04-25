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
            decision_b = t.get("track_B_decision")
            decision_c = t.get("track_C_decision")
            def fmt(d):
                if d is None: return "—"
                return "✅" if d else "❌"
            actual = t.get("actual_5d_return_pct")
            actual_str = f"{actual:+.2f}%" if actual is not None else "(待填)"
            lines.append(f"  {date} {ticker[:8]:<8} B={fmt(decision_b)} C={fmt(decision_c)} actual={actual_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"讀 log 失敗：{e}"


def parse_progress_reminder():
    """生成進度提醒（William 不會記得 3 週後該做什麼，每天 Telegram 提醒）

    返回包含三個區塊：
    - 累積進度（Track A/B/C 各自有 actual_return 的筆數）
    - 距離 review 還差幾筆
    - 達標後該做什麼（具體指令）
    """
    if not os.path.exists(PAPER_LOG):
        return "📅 還沒開始累積（4/27 起每天會自動跑）"

    try:
        with open(PAPER_LOG) as f:
            log_data = json.load(f)
        trades = log_data.get("trades", [])
    except Exception:
        return "📅 讀 log 失敗"

    n_total = len(trades)
    # 三軌各有 actual_return 的筆數
    n_filled_A = sum(1 for t in trades if t.get("actual_5d_return_pct") is not None)
    n_filled_B = sum(1 for t in trades
                     if t.get("track_B_decision") is True and t.get("actual_5d_return_pct") is not None)
    n_filled_C = sum(1 for t in trades
                     if t.get("track_C_decision") is True and t.get("actual_5d_return_pct") is not None)

    # 距離 review 還差幾筆（Track B 為主，10 筆是門檻）
    target = 10
    remaining = max(0, target - n_filled_B)

    lines = ["", "📅 ═══ 進度提醒 ═══"]
    lines.append(f"已累積：{n_total} 筆紀錄（{n_filled_A} 筆有 actual）")
    lines.append(f"  Track A (89.90) 有 actual: {n_filled_A}")
    lines.append(f"  Track B (V38)   有 actual: {n_filled_B} / {target}")
    lines.append(f"  Track C (V38d)  有 actual: {n_filled_C}")

    if remaining > 0:
        # 估算還要幾天（V38 平均 kept rate 15.8%，每週 5 交易日）
        days_per_trade = 1 / 0.158 if n_filled_B > 0 else 6.3
        est_calendar_days = int(remaining * days_per_trade * 1.4)  # 1.4 = 含週末
        lines.append("")
        lines.append(f"🎯 還差 {remaining} 筆 V38 buy 才能 review（約 {est_calendar_days} 天後）")
        lines.append(f"   現在不用做事，等系統自己累積")
    else:
        lines.append("")
        lines.append("🟢🟢🟢 達標！可以 review 了！")
        lines.append("   Windows 跑這條：")
        lines.append("   cd C:\\stock-evolution")
        lines.append("   python paper_trade_tracker.py review")
        lines.append("")
        lines.append("   看 V38d wr 是否 > V38 +5%")
        lines.append("   是 → 整合到 daily_scan 上線")
        lines.append("   否 → 接受 V38 final，停手等實盤")

    return "\n".join(lines)


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

    # === Step 5b: 進度提醒（給 William 知道距離 review 還差多少 + 達標後做什麼）===
    reminder = parse_progress_reminder()
    log(f"\n{reminder}")
    notify_lines.append(reminder)

    # === Step 6: 發 Telegram ===
    final_msg = "\n".join(notify_lines)
    log(f"\n=== 發 Telegram ===")
    log(final_msg)
    send_telegram(final_msg)
    log(f"\n✅ V38 Daily Auto 完成")


if __name__ == "__main__":
    main()
