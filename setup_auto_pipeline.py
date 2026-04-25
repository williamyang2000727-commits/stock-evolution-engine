r"""
一鍵設定 Windows 工作排程器（用 Python 替代 PS1，避免中文編碼問題）

用法：
  1. 設環境變數（admin PowerShell 跑一次）：
     [Environment]::SetEnvironmentVariable("GH_TOKEN", "ghp_...", "User")
  2. cd C:\stock-evolution
  3. python setup_auto_pipeline.py

排程：週一-週五 17:00（TWSE 收盤資料 16:30 settle 後）
"""
import os, sys, subprocess, getpass

TASK_NAME = "AutoDailyStockPipeline"
WORK_DIR = r"C:\stock-evolution"
SCRIPT_PATH = os.path.join(WORK_DIR, "auto_daily_pipeline.py")

# 找 Python 執行檔（優先 anaconda）
USER = getpass.getuser()
candidates = [
    rf"C:\Users\{USER}\anaconda3\python.exe",
    rf"C:\Users\{USER}\miniconda3\python.exe",
    sys.executable,  # fallback
]
PY_EXE = next((p for p in candidates if os.path.exists(p)), sys.executable)
print(f"Python: {PY_EXE}")
print(f"Script: {SCRIPT_PATH}")

if not os.path.exists(SCRIPT_PATH):
    print(f"ERROR: 找不到 {SCRIPT_PATH}")
    print(f"  先確認你在 {WORK_DIR} 而且 auto_daily_pipeline.py 已 git pull")
    sys.exit(1)

# 刪除既有 task
print(f"\n刪除既有排程（如果有）...")
subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
               capture_output=True, text=True)

# 建立排程：週一-五 17:00
# /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 17:00
print(f"\n建立排程: {TASK_NAME}")
cmd = [
    "schtasks", "/Create",
    "/TN", TASK_NAME,
    "/TR", f'"{PY_EXE}" "{SCRIPT_PATH}"',
    "/SC", "WEEKLY",
    "/D", "MON,TUE,WED,THU,FRI",
    "/ST", "17:00",
    "/RL", "HIGHEST",
    "/F",  # 覆蓋
]
result = subprocess.run(cmd, capture_output=True, text=True)
print(f"  stdout: {result.stdout}")
if result.returncode != 0:
    print(f"  stderr: {result.stderr}")
    print("\n❌ 排程建立失敗")
    print("可能原因：需要管理員權限。請用 admin PowerShell 重跑")
    sys.exit(1)

# 驗證
print(f"\n驗證排程已建立...")
verify = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME, "/V", "/FO", "LIST"],
                        capture_output=True, text=True, encoding="cp950", errors="ignore")
print(verify.stdout[:1500])

print("\n" + "=" * 60)
print("✅ 排程已建立")
print("=" * 60)
print(f"  排程名:  {TASK_NAME}")
print(f"  時間:    週一-週五 17:00")
print(f"  Python:  {PY_EXE}")
print(f"  Script:  {SCRIPT_PATH}")
print(f"  Log:     {os.path.join(WORK_DIR, 'daily_pipeline.log')}")

print("\n⚠️ 確認 GH_TOKEN 已永久存（admin PowerShell）：")
print('   [Environment]::SetEnvironmentVariable("GH_TOKEN", "ghp_...", "User")')

print("\n手動測試（先確認環境變數已設）：")
print("   schtasks /Run /TN " + TASK_NAME)
print("   Get-Content C:\\stock-evolution\\daily_pipeline.log -Tail 30")

print("\n查看下次執行時間：")
print("   schtasks /Query /TN " + TASK_NAME + " /FO LIST /V")
