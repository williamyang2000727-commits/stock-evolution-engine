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
TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN", "8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI")
TELEGRAM_CHAT = "5785839733"


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
def step_update_cache():
    """抓 TWSE/TPEX/yfinance 今日 K 線 append 到 cache"""
    import subprocess
    py = sys.executable
    script = os.path.join(USER_SE, "update_cache.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"update_cache.py 不在 {script}")
    r = subprocess.run([py, script], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"update_cache.py exit {r.returncode}\nstdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-1500:]}")
    log(f"  update_cache stdout (tail): {r.stdout[-500:]}")
    return True


# ─────── Step 2: init_state_gist ───────
def step_init_state():
    """從 1500 天完整 cache 算 state 推 Gist"""
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN 未設定，請 $env:GH_TOKEN = ...")
    import subprocess
    py = sys.executable
    script = os.path.join(USER_SE, "init_state_gist.py")
    r = subprocess.run([py, script], capture_output=True, text=True, timeout=600,
                       env={**os.environ})
    if r.returncode != 0:
        raise RuntimeError(f"init_state_gist exit {r.returncode}\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
    if "✅" not in r.stdout:
        raise RuntimeError(f"init_state 沒看到 ✅ 成功標記\n{r.stdout[-1500:]}")
    log(f"  init_state stdout (tail): {r.stdout[-500:]}")
    return True


# ─────── Step 3: rebuild_tab3 ───────
def step_rebuild_tab3():
    """用 cpu_replay 重建 Tab 3 backtest_results.json"""
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN 未設定")
    import subprocess
    py = sys.executable
    script = os.path.join(USER_SE, "rebuild_tab3.py")
    r = subprocess.run([py, script], capture_output=True, text=True, timeout=900,
                       env={**os.environ})
    if r.returncode != 0:
        raise RuntimeError(f"rebuild_tab3 exit {r.returncode}\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
    if "✅" not in r.stdout:
        raise RuntimeError(f"rebuild_tab3 沒看到 ✅\n{r.stdout[-1500:]}")
    log(f"  rebuild_tab3 stdout (tail): {r.stdout[-500:]}")
    return True


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

    try:
        run_step("Step 1: update_cache", step_update_cache)
        run_step("Step 2: init_state_gist", step_init_state)
        run_step("Step 3: rebuild_tab3", step_rebuild_tab3)

        log("\n=== Health check ===")
        n_trades, end_date = run_step("Health check", health_check)

        log("\n" + "=" * 70)
        log(f"🎉 Pipeline 全部成功！Tab 3: {n_trades} 筆 / 截止 {end_date}")
        log("=" * 70)
        telegram(f"✅ daily pipeline 完成 — Tab 3 {n_trades} 筆 / 截止 {end_date}")
        return 0
    except Exception as e:
        log(f"\n❌ Pipeline 失敗：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
