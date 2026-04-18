"""對實盤持倉用當前 GPU Gist 策略從買入日逐日 replay 到今天，
印出每檔是否該賣、何時該賣、賣出理由。

用法：
  python replay_holdings.py              # 預設 username=william
  python replay_holdings.py ken          # 指定 username
  python replay_holdings.py --simulate 2330.TW 2026-04-01 500   # 假想持倉
"""
import json, sys, os, pickle, urllib.request
from datetime import date
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# 載入股票中文名
_CN_NAMES = {}
_names_file = os.path.join(HERE, "tw_stock_names.json")
if os.path.exists(_names_file):
    try:
        with open(_names_file, "r", encoding="utf-8") as f:
            _CN_NAMES = json.load(f)
    except Exception:
        pass


def get_name(ticker):
    n = _CN_NAMES.get(ticker, "")
    return n if n else ticker.replace(".TW", "").replace(".TWO", "")
TOKEN = os.environ.get("GH_TOKEN", "")
if not TOKEN:
    # Try gh CLI
    try:
        import subprocess
        TOKEN = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
    except Exception:
        pass
DATA_GIST = "e1159b02a87d3c6ee9f33fb9ef61bb80"
GPU_GIST = "c1bef892d33589baef2142ce250d18c2"
CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")


def read_gist(gist_id, filename):
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {TOKEN}"} if TOKEN else {}
    )
    data = json.load(urllib.request.urlopen(req, timeout=15))
    fdata = data["files"].get(filename) or list(data["files"].values())[0]
    if fdata.get("truncated"):
        raw = urllib.request.urlopen(fdata["raw_url"], timeout=60).read().decode()
        return json.loads(raw)
    return json.loads(fdata["content"])


def should_sell_simple(bp, cur, peak, days_held, params):
    """88.60 / 189 都只用這 5 條（其他 use_X 都關著）。Mirror sell_rules.py 核心。"""
    if bp <= 0 or cur <= 0 or days_held < 1:
        return None
    ret = (cur / bp - 1) * 100
    peak_gain = (peak / bp - 1) * 100 if bp > 0 else 0

    # 1. 停損（含保本）
    eff_stop = params.get("stop_loss", -20)
    if params.get("use_breakeven", 0) and peak_gain >= params.get("breakeven_trigger", 20):
        eff_stop = 0
    if ret <= eff_stop:
        return f"保本出場 {ret:+.1f}%（曾漲 +{peak_gain:.1f}%）" if eff_stop == 0 else f"停損 {ret:+.1f}%"

    # 2. 停利
    if params.get("use_take_profit", 1) and ret >= params.get("take_profit", 80):
        return f"停利 +{ret:.1f}%"

    # 3. 移動停利
    trailing = params.get("trailing_stop", 0)
    if trailing > 0 and peak > bp * 1.01:
        dd = (cur / peak - 1) * 100
        if dd <= -trailing:
            return f"移動停利 {dd:.1f}%（peak {peak:.2f}）"

    # 4. 到期
    if days_held >= int(params.get("hold_days", 30)):
        return f"到期 {days_held} 交易日 {ret:+.1f}%"

    return None


def replay_holding(ticker, buy_date, buy_price, cache, params):
    """對單一持倉從買入日逐日 replay 到 cache 最後一天。"""
    if ticker not in cache:
        return {"ticker": ticker, "error": f"cache 無 {ticker} 資料"}

    df = cache[ticker].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    buy_ts = pd.Timestamp(buy_date)
    # 從買入日（含）之後
    mask = df.index >= buy_ts
    df_replay = df[mask]
    if len(df_replay) == 0:
        return {"ticker": ticker, "error": f"買入日 {buy_date} 在 cache 最後一天之後"}

    peak = buy_price
    triggered = None
    for idx, (dt, row) in enumerate(df_replay.iterrows()):
        cur = float(row["Close"])
        peak = max(peak, cur)
        days_held = idx  # idx=0 為買入日當天；idx=1 第一個持有交易日
        if days_held < 1:
            continue
        reason = should_sell_simple(buy_price, cur, peak, days_held, params)
        if reason:
            triggered = {
                "date": str(dt.date()),
                "days": days_held,
                "cur": cur,
                "peak": peak,
                "reason": reason,
                "return_pct": (cur / buy_price - 1) * 100,
            }
            break

    # 沒觸發：印當前狀態
    if not triggered:
        last_cur = float(df_replay.iloc[-1]["Close"])
        last_peak = peak
        last_days = len(df_replay) - 1
        return {
            "ticker": ticker,
            "buy_date": buy_date,
            "buy_price": buy_price,
            "status": "safe",
            "cur": last_cur,
            "peak": last_peak,
            "days": last_days,
            "return_pct": (last_cur / buy_price - 1) * 100,
        }

    # 觸發：算若抱到今天的報酬供對照
    last_cur = float(df_replay.iloc[-1]["Close"])
    return {
        "ticker": ticker,
        "buy_date": buy_date,
        "buy_price": buy_price,
        "status": "would_sell",
        **triggered,
        "held_to_today_cur": last_cur,
        "held_to_today_pct": (last_cur / buy_price - 1) * 100,
    }


def main():
    args = sys.argv[1:]
    simulate = None
    if args and args[0] == "--simulate":
        if len(args) < 4:
            print("用法: python replay_holdings.py --simulate <ticker> <buy_date YYYY-MM-DD> <buy_price>")
            sys.exit(1)
        simulate = {"ticker": args[1], "buy_date": args[2], "buy_price": float(args[3])}

    username = args[0] if args and args[0] != "--simulate" else "william"

    # Load params
    print(f"[讀] GPU Gist 策略...")
    strategy = read_gist(GPU_GIST, "best_strategy.json")
    params = strategy["params"]
    print(f"  策略 score: {strategy.get('score')}")
    print(f"  stop_loss={params.get('stop_loss')}  take_profit={params.get('take_profit')}")
    print(f"  trailing_stop={params.get('trailing_stop')}  breakeven_trigger={params.get('breakeven_trigger')}")
    print(f"  hold_days={params.get('hold_days')}")

    # Load cache
    print(f"[讀] Windows cache...")
    if not os.path.exists(CACHE_PATH):
        print(f"❌ cache 不存在：{CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  cache 共 {len(cache)} 檔")

    # Load holdings
    holdings = []
    if simulate:
        holdings = [simulate]
        print(f"[模擬持倉] {simulate['ticker']} 買 {simulate['buy_date']} @ {simulate['buy_price']}")
    else:
        print(f"[讀] Data Gist portfolios...")
        try:
            portfolios = read_gist(DATA_GIST, "portfolios.json")
            holdings = portfolios.get(username, {}).get("holdings", [])
        except Exception as e:
            print(f"❌ 讀 portfolios 失敗: {e}")
            sys.exit(1)
        print(f"  {username} 共 {len(holdings)} 檔實盤持倉")

    if not holdings:
        print(f"\n⚠️ {username} 無實盤持倉。想測試功能可用 --simulate。")
        sys.exit(0)

    # Replay each
    print()
    print("=" * 80)
    print(f"  持倉 replay（當前策略 score {strategy.get('score')}）")
    print("=" * 80)

    for h in holdings:
        res = replay_holding(h["ticker"], h["buy_date"], float(h["buy_price"]), cache, params)
        name = get_name(res.get("ticker", h.get("ticker", "")))
        print()
        if "error" in res:
            print(f"⚠️ {res['ticker']} {name}: {res['error']}")
            continue
        if res["status"] == "safe":
            print(f"✅ {res['ticker']} {name}（買 {res['buy_date']} @ {res['buy_price']}）")
            print(f"   至今 {res['days']} 交易日，現 {res['cur']:.2f}，peak {res['peak']:.2f}")
            print(f"   報酬 {res['return_pct']:+.1f}% — 當前策略下仍可繼續持有")
        else:
            print(f"❗ {res['ticker']} {name}（買 {res['buy_date']} @ {res['buy_price']}）")
            print(f"   在 {res['date']}（第 {res['days']} 交易日）觸發「{res['reason']}」")
            print(f"   出場價 {res['cur']:.2f}（peak {res['peak']:.2f}）")
            print(f"   該出場報酬：{res['return_pct']:+.1f}%")
            print(f"   若抱到今天：{res['held_to_today_cur']:.2f} 報酬 {res['held_to_today_pct']:+.1f}%")

    print()
    print("=" * 80)
    print("  建議：標 ❗ 的持倉，在當前策略下應已出場。")
    print("        實盤處理：照 replay 的賣出時點回看，判斷是否手動平倉。")
    print("        若 ✅ 則繼續持有，daily_scan 每天會從今天前瞻檢查。")


if __name__ == "__main__":
    main()
