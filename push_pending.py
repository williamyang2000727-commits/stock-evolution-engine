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

# === Cache 飄風險警告 ===
_cache_path = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
if os.path.exists(_cache_path):
    import pickle
    from datetime import datetime as _dt
    try:
        with open(_cache_path, "rb") as _f:
            _cache = pickle.load(_f)
        _last_days = [df.index[-1] for df in _cache.values() if df is not None and len(df) > 0]
        _cache_last = max(_last_days) if _last_days else None
        _pending_updated = d.get("updated_at", "")
        print(f"  Cache 最後日期：{_cache_last.date() if _cache_last else '?'}")
        print(f"  Pending 生成時間：{_pending_updated}")
        if _cache_last and _pending_updated:
            import re
            _m = re.match(r'(\d{4}-\d{2}-\d{2})', _pending_updated)
            if _m:
                _pd_date = _dt.fromisoformat(_m.group(1))
                _gap = (_dt.now() - _pd_date).days
                if _gap >= 3:
                    print(f"  ⚠️ pending 已 {_gap} 天沒跑，Step 2 重跑 cpu_replay 可能會飄（筆數 ± 幾筆）")
                    print(f"     若希望數字精準跟 Telegram 一致 → 跑 pending_to_web.py 代替 Step 2")
    except Exception as _e:
        pass
print("=" * 50)

ans = input("確定推到 Gist？[y/N]: ").strip().lower()
if ans != "y":
    print("取消")
    sys.exit(0)

# === Step 0: 確保 Windows cache 最新（補到今天）===
print()
print("=" * 50)
print("  📥 Step 0: 確認 cache 最新（Windows cache → 今天）")
print("=" * 50)
import subprocess
_here = os.path.dirname(os.path.abspath(__file__))
_uc_path = os.path.join(_here, "update_cache.py")
if os.path.exists(_uc_path):
    try:
        subprocess.run([sys.executable, _uc_path], cwd=_here, timeout=600)
    except Exception as _e:
        print(f"⚠️ update_cache 失敗：{_e}，繼續但 end_date 可能不到今天")
else:
    print("⚠️ 找不到 update_cache.py，略過 cache 更新")
print()

payload = json.dumps({"files": {"best_strategy.json": {"content": open(PENDING, encoding="utf-8").read()}}}).encode()
req = urllib.request.Request(
    f"https://api.github.com/gists/{GIST_ID}",
    data=payload, method="PATCH",
    headers={"Authorization": f"token {GH_TOKEN}", "Content-Type": "application/json"}
)
print()
print("=" * 50)
print("  📤 Step 1: 推 GPU Gist（策略 params）")
print("=" * 50)
r = urllib.request.urlopen(req, timeout=30)
print(f"✅ GPU Gist 推送成功：{r.status}")
dst = PENDING + ".pushed"
# Windows os.rename 不覆蓋既有檔案 → 用 os.replace（跨平台支援覆蓋）
os.replace(PENDING, dst)
print(f"pending_push.json → pending_push.json.pushed")

# === 自動跑 backtest_to_web.py 補完整歷史 + 觸發 daily-scan workflow ===
print()
print("=" * 50)
print("  🚀 Step 2: 執行 backtest_to_web.py 補完整歷史到今天")
print("=" * 50)
_bt_path = os.path.join(_here, "backtest_to_web.py")
if not os.path.exists(_bt_path):
    print(f"⚠️ 找不到 backtest_to_web.py，略過。請手動執行。")
else:
    try:
        # 用 stdin 傳空字串給 input("Press Enter to exit...")，讓它不卡住
        result = subprocess.run(
            [sys.executable, _bt_path],
            input="\n", text=True,
            cwd=_here, timeout=600  # 10 分鐘
        )
        if result.returncode == 0:
            print(f"\n✅ backtest_to_web.py 執行完成（Data Gist 已更新到今天）")
        else:
            print(f"\n⚠️ backtest_to_web.py 失敗 (returncode={result.returncode})，請手動跑一次")
    except subprocess.TimeoutExpired:
        print(f"\n⚠️ backtest_to_web.py 超時，請手動確認")
    except Exception as e:
        print(f"\n⚠️ backtest_to_web.py 執行出錯：{e}")

# === 自動觸發 daily-scan workflow（讓 Web 立刻延續到今天）===
print()
print("=" * 50)
print("  🔔 Step 3: 觸發 GitHub Actions daily-scan workflow")
print("=" * 50)
try:
    result = subprocess.run(
        ["gh", "workflow", "run", "daily-scan.yml",
         "--repo", "williamyang2000727-commits/stock-web-app"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        print("✅ workflow 已觸發（5-10 分鐘內 Web 會延續到今天）")
    else:
        print(f"⚠️ workflow 觸發失敗：{result.stderr}")
        print("   請手動執行：gh workflow run daily-scan.yml --repo williamyang2000727-commits/stock-web-app")
except FileNotFoundError:
    print("⚠️ 找不到 gh CLI，略過自動觸發")
    print("   請手動到 https://github.com/williamyang2000727-commits/stock-web-app/actions 觸發")
except Exception as e:
    print(f"⚠️ 觸發出錯：{e}")

print()
print("=" * 50)
print("  ✅ 一條龍完成！")
print("=" * 50)
print(f"  1. GPU Gist 策略 params 已更新")
print(f"  2. Data Gist 回測結果已補到今天")
print(f"  3. daily-scan workflow 已觸發")
print(f"  4. 去 Web 按 🔄 確認 Tab 3 顯示新策略")
print(f"  5. 記得 Mac 跑：bash ~/.stock-backup/switch_strategy.sh --backup <策略名>")
