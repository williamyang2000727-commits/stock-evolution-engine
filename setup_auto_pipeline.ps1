# 一鍵設定 Windows 工作排程器：每天 14:00 自動跑 auto_daily_pipeline.py
# 用法：
#   1. 確認 $env:GH_TOKEN 已存到使用者環境變數（永久）：
#      [Environment]::SetEnvironmentVariable("GH_TOKEN", "ghp_...", "User")
#   2. 用系統管理員身份開 PowerShell
#   3. cd C:\stock-evolution
#   4. .\setup_auto_pipeline.ps1

$ErrorActionPreference = "Stop"

$TaskName = "AutoDailyStockPipeline"
$WorkDir = "C:\stock-evolution"
$ScriptPath = Join-Path $WorkDir "auto_daily_pipeline.py"
$LogPath = Join-Path $WorkDir "daily_pipeline.log"

# 找 conda python（用戶用 anaconda）
$PyExe = "C:\Users\$env:USERNAME\anaconda3\python.exe"
if (-not (Test-Path $PyExe)) {
    $PyExe = (Get-Command python).Source
    Write-Host "  找不到 anaconda，改用 system python: $PyExe"
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "找不到 $ScriptPath，請先 git pull 確認檔案在"
    exit 1
}

# 刪除既有 task（如果有）
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "  刪除既有排程: $TaskName"
}

# Action: 跑 python script，cwd = C:\stock-evolution
$Action = New-ScheduledTaskAction `
    -Execute $PyExe `
    -Argument $ScriptPath `
    -WorkingDirectory $WorkDir

# Trigger: 週一到週五 17:00（TWSE/TPEX 收盤資料 16:30 後才齊全，留 30 分鐘 buffer）
# 16:35 daily_scan GitHub Actions 跑完後再跑這個，重置 state + Tab 3 用 cpu_replay 真公式
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "17:00"

# Settings: 失敗自動重試 + 不要強制電腦睡眠
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries

# Principal: 用當前用戶身份（不要 SYSTEM，環境變數會丟失）
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:COMPUTERNAME\$env:USERNAME" `
    -LogonType InteractiveOrPassword `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "每日 14:00 自動跑：update_cache → init_state_gist → rebuild_tab3 → 健康檢查"

Write-Host ""
Write-Host "✅ 排程已建立: $TaskName"
Write-Host "   執行時間: 週一-週五 17:00（TWSE 收盤資料 16:30 齊全後）"
Write-Host "   Python:   $PyExe"
Write-Host "   Script:   $ScriptPath"
Write-Host "   Log:      $LogPath"
Write-Host ""
Write-Host "⚠️ 重要：確認 GH_TOKEN 已存使用者環境變數："
Write-Host "   [Environment]::SetEnvironmentVariable('GH_TOKEN', 'ghp_...', 'User')"
Write-Host ""
Write-Host "驗證：手動跑一次"
Write-Host "   Start-ScheduledTask -TaskName $TaskName"
Write-Host "   Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "   Get-Content $LogPath -Tail 30"
