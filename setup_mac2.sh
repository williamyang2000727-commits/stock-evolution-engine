#!/bin/bash
# === 第二台 Mac 一鍵部署進化引擎 ===
# 用法：在另一台 MacBook 打開終端機，貼上：
#   curl -sL 或直接 bash setup_mac2.sh

set -e
echo "🦞 進化引擎部署開始..."

# 1. 建立工作目錄
mkdir -p ~/stock-evolution
cd ~/stock-evolution

# 2. 下載 cloud_evolve.py（從你的 GitHub repo）
echo "📥 下載進化引擎..."
curl -sL "https://raw.githubusercontent.com/williamyang2000727-commits/stock-evolution-engine/main/cloud_evolve.py" -o cloud_evolve.py
curl -sL "https://raw.githubusercontent.com/williamyang2000727-commits/stock-evolution-engine/main/requirements.txt" -o requirements.txt

# 3. 安裝 Python 依賴
echo "📦 安裝依賴..."
pip3 install --user numpy yfinance requests 2>/dev/null || python3 -m pip install --user numpy yfinance requests

# 4. 設定環境變數
cat > .env.sh << 'ENVEOF'
export TELEGRAM_BOT_TOKEN="8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI"
export TELEGRAM_CHAT_IDS="5785839733,8236911077"
export GIST_ID="c1bef892d33589baef2142ce250d18c2"
export GH_TOKEN=""
export N_TESTS="20000"
export JOB_ID="mac2"
export SEED_OFFSET="88888888"
ENVEOF

# 5. 建立永久執行腳本
cat > run_forever.sh << 'RUNEOF'
#!/bin/bash
cd ~/stock-evolution
source .env.sh
PY=$(which python3)

echo "🦞 第二台 Mac 進化引擎啟動！"
echo "每輪 20,000 組 x 5 並行"

for i in 1 2 3 4 5; do
    while true; do
        $PY cloud_evolve.py 2>/dev/null
        sleep 1
    done &
done
wait
RUNEOF
chmod +x run_forever.sh

# 6. 建立 LaunchAgent（開機自動啟動）
PLIST=~/Library/LaunchAgents/com.lobster.evolve.plist
mkdir -p ~/Library/LaunchAgents
cat > "$PLIST" << PEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lobster.evolve</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${HOME}/stock-evolution/run_forever.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${HOME}/stock-evolution/evolve.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/stock-evolution/evolve_err.log</string>
</dict>
</plist>
PEOF

# 7. 啟動！
launchctl load "$PLIST" 2>/dev/null
launchctl start com.lobster.evolve 2>/dev/null

# 也直接背景跑一份
nohup bash ~/stock-evolution/run_forever.sh > ~/stock-evolution/evolve.log 2>&1 &

echo ""
echo "✅ 部署完成！進化引擎已啟動"
echo "📂 工作目錄：~/stock-evolution/"
echo "📊 Log 檔案：~/stock-evolution/evolve.log"
echo "🔄 開機會自動啟動"
echo ""
echo "管理指令："
echo "  查看狀態：ps aux | grep cloud_evolve"
echo "  停止：pkill -f cloud_evolve"
echo "  重啟：bash ~/stock-evolution/run_forever.sh &"
