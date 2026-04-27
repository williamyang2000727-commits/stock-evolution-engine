r"""
全自動化日常 pipeline — Windows 工作排程器跑

每個交易日台股 13:35 收盤後跑：
  1. update_cache.py   — 抓今日 K 線 append 到 stock_data_cache.pkl
  2. init_state_gist   — 從 1500 天完整 cache 算 state，推 state Gist
  3. rebuild_tab3      — 用 cpu_replay 重建 backtest_results.json，推 Data Gist

跑完後：
  - state Gist 完整正確（1950 ticker，下次 daily_scan 用 with_state path）
  - Tab 3 backtest 對齊 cpu_replay
  - 永久同步，無 drift

異常處理：
  - 每步失敗會 retry 3 次（網路抖動）
  - 最終失敗 → log 寫 daily_pipeline_error.log + Telegram 推
  - 整體 raise → Windows 排程顯示 fail，下次重試

設環境變數：
  $env:GH_TOKEN = "ghp_..."
"""
import os, sys, time, json, traceback, urllib.request
from datetime import datetime, timezone, timedelta

# 自動偵測 USER_SE：先看 Windows C:\stock-evolution，fallback 到 ~/stock-evolution
_candidates = [
    r"C:\stock-evolution",
    os.path.join(os.path.expanduser("~"), "stock-evolution"),
    os.path.dirname(os.path.abspath(__file__)),  # script 自己所在目錄
]
USER_SE = next((p for p in _candidates if os.path.isfile(os.path.join(p, "update_cache.py"))), _candidates[0])
sys.path.insert(0, USER_SE)
print(f"USER_SE = {USER_SE}")

LOG_FILE = os.path.join(USER_SE, "daily_pipeline.log")
TW_TZ = timezone(timedelta(hours=8))
TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "5785839733")


def log(msg):
    ts = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def telegram(msg):
    try:
        import urllib.parse, ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT, "text": msg[:4000]}).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, context=ctx, timeout=15)
    except Exception:
        pass


def run_step(step_name, func, max_retry=3):
    """跑一個 step，失敗 retry，最終失敗 raise + Telegram"""
    log(f"=== {step_name} ===")
    last_err = None
    for attempt in range(max_retry):
        try:
            result = func()
            log(f"  ✅ {step_name} done")
            return result
        except Exception as e:
            last_err = e
            log(f"  ❌ Attempt {attempt + 1}/{max_retry} fail: {e}")
            if attempt < max_retry - 1:
                wait = 2 ** (attempt + 2)  # 4s, 8s, 16s
                log(f"  → retry in {wait}s")
                time.sleep(wait)
    err_msg = f"❌ Pipeline {step_name} 最終失敗：{last_err}\n{traceback.format_exc()[:1500]}"
    log(err_msg)
    telegram(f"🚨 daily pipeline {step_name} 失敗\n{str(last_err)[:300]}")
    raise RuntimeError(err_msg)


# ─────── Step 1: update_cache ───────
def _run_subprocess(py, script, timeout):
    """跑 Python script 並強制 UTF-8 stdout（避免 Windows cp950 emoji 炸）"""
    import subprocess
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run(
        [py, script],
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    # 手動 decode（避免 text=True 用系統 codec）
    out = r.stdout.decode("utf-8", errors="replace") if r.stdout else ""
    err = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
    return r.returncode, out, err


# ─────── Step 0: 自動拉新版檔案（從 GitHub）───────
def step_self_update():
    """從 GitHub 拉最新 4 個關鍵檔案，避免本地 stale"""
    import base64
    repo = "williamyang2000727-commits/stock-evolution-engine"
    files_to_sync = [
        "auto_daily_pipeline.py",  # 自己（特殊處理：寫 .new 下次生效）
        "init_state_gist.py",
        "rebuild_tab3.py",
        "update_cache.py",
        "write_web_data.py",
        "daily_health_report.py",  # 健康報告 + 異常偵測（給 William）
    ]
    n_updated = 0
    for fn in files_to_sync:
        url = f"https://api.github.com/repos/{repo}/contents/{fn}"
        try:
            req = urllib.request.Request(url)
            r = urllib.request.urlopen(req, timeout=30)
            d = json.loads(r.read())
            new_content = base64.b64decode(d["content"])
            local_path = os.path.join(USER_SE, fn)
            old_content = b""
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    old_content = f.read()
            if new_content != old_content:
                if fn == "auto_daily_pipeline.py":
                    # 自己有更新 → 寫 .new 檔，下次啟動時換
                    with open(local_path + ".new", "wb") as f:
                        f.write(new_content)
                    log(f"  ⚠️ {fn} 有新版，下次啟動時自動換（不 overwrite 自己避免崩潰）")
                    continue
                with open(local_path, "wb") as f:
                    f.write(new_content)
                log(f"  ⬇️ 更新 {fn}")
                n_updated += 1
        except Exception as e:
            log(f"  ⚠️ 拉 {fn} 失敗（不擋繼續）: {e}")
    if n_updated == 0:
        log("  所有檔案都是最新版")
    return n_updated


def _swap_self_if_pending():
    """如果有 auto_daily_pipeline.py.new，啟動時換上去再重啟"""
    here = os.path.abspath(__file__)
    new_file = here + ".new"
    if os.path.exists(new_file):
        try:
            with open(new_file, "rb") as f:
                new_content = f.read()
            backup = here + ".backup"
            if os.path.exists(here):
                with open(here, "rb") as old_f:
                    with open(backup, "wb") as bf:
                        bf.write(old_f.read())
            with open(here, "wb") as f:
                f.write(new_content)
            os.remove(new_file)
            print(f"  ⬇️ 已換新版 {os.path.basename(here)}（舊版備份 .backup），重啟 ...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"  ❌ swap self failed: {e}")


def step_update_cache():
    """抓 TWSE/TPEX/yfinance 今日 K 線 append 到 cache"""
    py = sys.executable
    script = os.path.join(USER_SE, "update_cache.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"update_cache.py 不在 {script}")
    code, out, err = _run_subprocess(py, script, 600)
    if code != 0:
        raise RuntimeError(f"update_cache.py exit {code}\nstdout:\n{out[-1500:]}\nstderr:\n{err[-1500:]}")
    log(f"  update_cache stdout (tail): {out[-500:]}")
    return True


# ─────── Step 2: init_state_gist ───────
def step_init_state():
    """從 1500 天完整 cache 算 state 推 Gist"""
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN 未設定，請 $env:GH_TOKEN = ...")
    py = sys.executable
    script = os.path.join(USER_SE, "init_state_gist.py")
    code, out, err = _run_subprocess(py, script, 600)
    if code != 0:
        raise RuntimeError(f"init_state_gist exit {code}\n{out[-1000:]}\n{err[-1000:]}")
    if "✅" not in out:
        raise RuntimeError(f"init_state 沒看到 ✅ 成功標記\n{out[-1500:]}")
    log(f"  init_state stdout (tail): {out[-500:]}")
    return True


# ─────── Step 3: rebuild_tab3 ───────
def step_rebuild_tab3():
    """用 cpu_replay 重建 Tab 3 backtest_results.json"""
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN 未設定")
    py = sys.executable
    script = os.path.join(USER_SE, "rebuild_tab3.py")
    code, out, err = _run_subprocess(py, script, 900)
    if code != 0:
        raise RuntimeError(f"rebuild_tab3 exit {code}\n{out[-1000:]}\n{err[-1000:]}")
    if "✅" not in out:
        raise RuntimeError(f"rebuild_tab3 沒看到 ✅\n{out[-1500:]}")
    log(f"  rebuild_tab3 stdout (tail): {out[-500:]}")
    return True


def step_write_web_data():
    """從 1500 天 cache 切 80 天 K + 算 pending + buy_signals 推 Gist"""
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN 未設定")
    py = sys.executable
    script = os.path.join(USER_SE, "write_web_data.py")
    if not os.path.exists(script):
        log("  write_web_data.py 還沒拉，跳過（下次 step 0 self-update 會拉）")
        return False
    code, out, err = _run_subprocess(py, script, 600)
    if code != 0:
        raise RuntimeError(f"write_web_data exit {code}\n{out[-1000:]}\n{err[-1000:]}")
    if "🎉" not in out:
        raise RuntimeError(f"write_web_data 沒看到完成標記\n{out[-1500:]}")
    log(f"  write_web_data stdout (tail): {out[-500:]}")
    return True


def step_health_report():
    """推每日健康報告 Telegram（含異常偵測）"""
    py = sys.executable
    script = os.path.join(USER_SE, "daily_health_report.py")
    if not os.path.exists(script):
        log("  daily_health_report.py 還沒拉，跳過")
        return False
    code, out, err = _run_subprocess(py, script, 60)
    if code != 0:
        log(f"  ⚠️ health_report 失敗（不擋 pipeline）: {err[-500:]}")
        return False
    log(f"  ✅ Telegram 已推每日健康報告")
    return True


# step_monitor_users 已移除 — Web App 即時算每個用戶自己的 sell_signals 已足夠
# 詳見 app.py:474 check_sell_signals(user_holdings, ...)
# 每個用戶登入後 Tab 0 即時顯示自己持倉的賣出訊號（客製化）


# ─────── 健康檢查（成功後驗證 Gist）───────
def health_check():
    """跑完後驗證 Gist 真的更新了"""
    GH_TOKEN = os.environ["GH_TOKEN"]
    today = datetime.now(TW_TZ).strftime("%Y-%m-%d")

    # 檢查 state Gist
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/18a7270d897c8821b291cfd61796bd80",
                                headers={"Authorization": f"token {GH_TOKEN}"}),
        timeout=30,
    )
    d = json.loads(r.read())
    state = json.loads(d["files"]["indicator_state.json"]["content"])
    n_ticker = len(state.get("states", {}))
    if n_ticker < 1500:
        raise RuntimeError(f"state Gist 只有 {n_ticker} ticker（< 1500），不正常")
    log(f"  ✅ state Gist: {n_ticker} ticker, updated={state.get('updated','?')}")

    # 檢查 Data Gist
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/e1159b02a87d3c6ee9f33fb9ef61bb80",
                                headers={"Authorization": f"token {GH_TOKEN}"}),
        timeout=30,
    )
    d = json.loads(r.read())
    bt = json.loads(d["files"]["backtest_results.json"]["content"])
    n_trades = bt["stats"].get("total_trades", 0)
    end_date = bt["stats"].get("end_date", "")
    if n_trades < 100:
        raise RuntimeError(f"backtest_results 只有 {n_trades} 筆")
    log(f"  ✅ Tab 3: {n_trades} trades, end={end_date}, total={bt['stats'].get('total_return_pct')}%, wr={bt['stats'].get('win_rate')}%")
    return n_trades, end_date


# ─────── Main ───────
def main():
    # 啟動時：如果有 .new 待換，先換再重啟
    _swap_self_if_pending()

    log("\n" + "=" * 70)
    log("🚀 Auto daily pipeline 啟動")
    log("=" * 70)

    # 週末跳過（除非 --force）
    now = datetime.now(TW_TZ)
    force = "--force" in sys.argv
    if now.weekday() >= 5 and not force:
        log(f"  週末（{now.strftime('%a')}），跳過。要手動測試請加 --force")
        return 0
    if force:
        log("  ⚡ --force mode：強制執行（即使週末/假日）")

    # ⭐ Idempotent check：今日已成功跑過就跳過（避免 18:00 / 19:00 retry 重複跑）
    # 用 daily_pipeline.success_marker 檔案標記今日成功
    today_str = now.strftime("%Y-%m-%d")
    marker_file = os.path.join(USER_SE, f"daily_pipeline.success.{today_str}")
    if not force and os.path.exists(marker_file):
        log(f"  ✅ 今日已成功跑過（marker {marker_file}），跳過")
        return 0

    try:
        run_step("Step 0: self-update from GitHub", step_self_update)
        run_step("Step 1: update_cache", step_update_cache)
        run_step("Step 2: init_state_gist", step_init_state)
        run_step("Step 3: rebuild_tab3", step_rebuild_tab3)
        run_step("Step 4: write_web_data (history+scan)", step_write_web_data)
        run_step("Step 5: daily_health_report (Telegram)", step_health_report)
        # 註：每個用戶持倉的 sell_signals 由 Web App 即時算（app.py:474 check_sell_signals）
        # 每個用戶登入時看自己的訊號，已經客製化，不需要 pipeline 預算

        log("\n=== Health check ===")
        n_trades, end_date = run_step("Health check", health_check)

        log("\n" + "=" * 70)
        log(f"🎉 Pipeline 全部成功！Tab 3: {n_trades} 筆 / 截止 {end_date}")
        log("=" * 70)
        telegram(f"✅ daily pipeline 完成 — Tab 3 {n_trades} 筆 / 截止 {end_date}")
        # 寫今日成功 marker（idempotent，避免 retry trigger 重跑）
        try:
            today_str = now.strftime("%Y-%m-%d")
            marker_file = os.path.join(USER_SE, f"daily_pipeline.success.{today_str}")
            # 清理舊 marker（保留最近 7 天）
            for old_f in os.listdir(USER_SE):
                if old_f.startswith("daily_pipeline.success.") and old_f != f"daily_pipeline.success.{today_str}":
                    try:
                        os.remove(os.path.join(USER_SE, old_f))
                    except Exception:
                        pass
            with open(marker_file, "w") as f:
                f.write(datetime.now(TW_TZ).isoformat())
        except Exception as e:
            log(f"  ⚠️ 寫 success marker 失敗（不影響本次）: {e}")
        return 0
    except Exception as e:
        log(f"\n❌ Pipeline 失敗：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
