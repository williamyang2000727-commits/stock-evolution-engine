# V38 Daily Auto — Windows 工作排程器設定
# 用法：開啟 PowerShell（管理員），cd C:\stock-evolution，執行：
#   powershell -ExecutionPolicy Bypass -File setup_v38_schedule.ps1

$taskName = "V38_Kronos_PaperTrade_Daily"
$pythonExe = (Get-Command python).Path
$workDir = "C:\stock-evolution"
$script = "$workDir\v38_daily_auto.py"

# 移除舊 task
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# 建新 task
$action = New-ScheduledTaskAction -Execute $pythonExe -Argument "$script" -WorkingDirectory $workDir
# 每天 16:40 跑（台股盤後 + daily_scan 16:35 跑完後）
$trigger = New-ScheduledTaskTrigger -Daily -At "16:40"
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "V38 Kronos Paper Trade Daily Auto"

Write-Host "✅ 排程已建立：$taskName"
Write-Host "  每天 16:40 自動執行：$pythonExe $script"
Write-Host ""
Write-Host "驗證：開啟『工作排程器』搜尋 '$taskName' 看到 enabled"
Write-Host ""
Write-Host "立刻手動測試（不等 16:40）："
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
Write-Host ""
Write-Host "查看 log："
Write-Host "  Get-Content $workDir\v38_daily_auto.log -Tail 30"
