#!/bin/bash
# === 第二台 Mac 一鍵部署進化引擎 ===
# 用法：在另一台 MacBook 打開終端機，貼上：
#   bash <(curl -sL https://raw.githubusercontent.com/williamyang2000727-commits/stock-evolution-engine/main/setup_mac2.sh)

set -e
echo "🦞 進化引擎部署開始..."

# 1. 取得 GitHub Token（寫 Gist 用）
echo ""
echo "📌 需要 GitHub Token 才能同步策略到 Gist"
GH_TOKEN_VAL=""

# 嘗試用 gh CLI 自動取得
if command -v gh &>/dev/null; then
    GH_TOKEN_VAL=$(gh auth token 2>/dev/null || true)
fi

if [ -z "$GH_TOKEN_VAL" ]; then
    echo "⚠️  找不到 gh CLI 或未登入"
    echo ""
    echo "請先安裝並登入 GitHub CLI："
    echo "  brew install gh"
    echo "  gh auth login"
    echo ""
    echo "或者直接貼上你的 Personal Access Token（需要 gist 權限）："
    read -p "GitHub Token: " GH_TOKEN_VAL
fi

if [ -z "$GH_TOKEN_VAL" ]; then
    echo "⚠️  沒有 Token，將以唯讀模式運行（突破只推 Telegram，不寫 Gist）"
fi

# 2. 建立工作目錄
mkdir -p ~/stock-evolution
cd ~/stock-evolution

# 3. 下載 cloud_evolve.py
echo "📥 下載進化引擎..."
curl -sL "https://raw.githubusercontent.com/williamyang2000727-commits/stock-evolution-engine/main/cloud_evolve.py" -o cloud_evolve.py
curl -sL "https://raw.githubusercontent.com/williamyang2000727-commits/stock-evolution-engine/main/requirements.txt" -o requirements.txt

# 4. 建立 venv 並安裝依賴
echo "📦 建立虛擬環境並安裝依賴..."
python3 -m venv ~/stock-evolution/venv
~/stock-evolution/venv/bin/pip install --upgrade pip numpy yfinance requests

# 5. 設定環境變數
cat > .env.sh << ENVEOF
export TELEGRAM_BOT_TOKEN="8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI"
export TELEGRAM_CHAT_IDS="5785839733,8236911077"
export GIST_ID="c1bef892d33589baef2142ce250d18c2"
export GH_TOKEN="${GH_TOKEN_VAL}"
export N_TESTS="30000"
export MAX_MINUTES="999999"
export JOB_ID="mac2"
export SEED_OFFSET="88888888"
ENVEOF

# 6. 建立永久執行腳本
cat > run_forever.sh << 'RUNEOF'
#!/bin/bash
cd ~/stock-evolution
source .env.sh
PY=~/stock-evolution/venv/bin/python3

echo "🦞 第二台 Mac 進化引擎啟動！"
echo "先下載資料快取，再 8 組並行..."

# 先跑一次下載資料快取
$PY cloud_evolve.py 2>&1 | tail -3
echo "快取完成，啟動 8 組並行"

for i in 1 2 3 4 5 6 7 8; do
    while true; do
        $PY cloud_evolve.py 2>/dev/null
        sleep 1
    done &
done
wait
RUNEOF
chmod +x run_forever.sh

# 7. 建立 LaunchAgent（開機自動啟動）
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

# 8. 啟動！
launchctl load "$PLIST" 2>/dev/null
launchctl start com.lobster.evolve 2>/dev/null

# 也直接背景跑一份
nohup bash ~/stock-evolution/run_forever.sh > ~/stock-evolution/evolve.log 2>&1 &

echo ""
echo "✅ 部署完成！進化引擎已啟動"
echo "📂 工作目錄：~/stock-evolution/"
echo "📊 Log 檔案：~/stock-evolution/evolve.log"
echo "🔄 開機會自動啟動"
if [ -n "$GH_TOKEN_VAL" ]; then
    echo "🔗 Gist 同步：已啟用（突破會自動寫入 Gist）"
else
    echo "⚠️  Gist 同步：未啟用（突破只推 Telegram）"
fi
echo ""
echo "管理指令："
echo "  查看狀態：ps aux | grep cloud_evolve"
echo "  停止：pkill -f cloud_evolve"
echo "  重啟：bash ~/stock-evolution/run_forever.sh &"
