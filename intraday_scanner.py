"""
V44 盤中即時掃描 — 89.90 規則套盤中報價
用法：C:\\stock-evolution> python intraday_scanner.py
        loop 模式：python intraday_scanner.py --loop

核心想法（不 filter，不換策略，只是讓 89.90 訊號早出現）：
  daily_scan 16:35 跑 → 你 D+1 13:25 才能買 → 中間如果股票漲了你少賺
  intraday_scanner 09:30 / 10:00 / ... 每 30 分鐘掃 → 訊號出立刻 Telegram → 你即時下單

對比：
  daily_scan: 用收盤價 → 隔日 D+1 開盤買（中間 17 小時 + 隔夜風險）
  intraday:   用盤中價 → 當日下單買（節省 17 小時）

預期效益：
  進場價平均改善 0.5-1.5%（避免「隔夜跳空高」）
  每年 22 筆 × 1% = +22% 總報酬改善（不換策略）
  完全跟 89.90 相容，不需要找新 alpha

實作：
  1. 載 Windows cache (1500 天歷史)
  2. 用 mis.twse.com.tw 即時 API 抓盤中 close（pipe 分隔最多 100 檔）
  3. 把盤中 close 接到 cache 末尾當「今天的 close」
  4. 跑 base.precompute → cpu_replay
  5. 看 trades 最後一筆 buy_date == 今天 → 訊號！
  6. Telegram 推

不接 Shioaji 自動下單（先看訊號頻率/品質決定）
"""
import os, sys, json, pickle, time, argparse
import numpy as np
import pandas as pd
import requests
import urllib3
import warnings
from datetime import datetime, timedelta, timezone

urllib3.disable_warnings()
warnings.filterwarnings("ignore")

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
TW_TZ = timezone(timedelta(hours=8))

# Telegram (從 stock_secrets.md / gpu_cupy_evolve.py)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = [os.environ.get("TELEGRAM_CHAT_ID", "5785839733")]


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def send_telegram(msg):
    """送 Telegram 訊息"""
    import ssl, urllib.request, urllib.parse
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": chat_id, "text": msg[:4000], "parse_mode": "Markdown"
            }).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, context=ctx, timeout=15)
        except Exception as e:
            print(f"  Telegram fail: {e}")


def fetch_intraday_quotes(tickers, batch_size=100):
    """
    用 mis.twse.com.tw 抓盤中即時報價
    一次最多 100 檔 (pipe-separated)
    回傳 dict: {ticker: {"open", "high", "low", "close", "vol"}}
    """
    result = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        # ex_ch 格式: tse_2330.tw 或 otc_3264.tw
        ex_ch_parts = []
        for t in batch:
            code = t.replace(".TW", "").replace(".TWO", "")
            if t.endswith(".TWO"):
                ex_ch_parts.append(f"otc_{code}.tw")
            else:
                ex_ch_parts.append(f"tse_{code}.tw")
        ex_ch = "|".join(ex_ch_parts)
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={ex_ch}"
        try:
            r = requests.get(url, timeout=15, verify=False,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://mis.twse.com.tw/"})
            data = r.json()
            for row in data.get("msgArray", []):
                code = row.get("c", "").strip()
                ex = row.get("ex", "tse")
                tk = f"{code}.TW" if ex == "tse" else f"{code}.TWO"
                # mis api fields:
                # "z" = 最後成交價, "o" = 開盤, "h" = 最高, "l" = 最低
                # "v" = 累積成交量(張), "y" = 昨收
                try:
                    close = float(row["z"]) if row.get("z") and row["z"] != "-" else 0
                    o = float(row["o"]) if row.get("o") and row["o"] != "-" else close
                    h = float(row["h"]) if row.get("h") and row["h"] != "-" else close
                    lo = float(row["l"]) if row.get("l") and row["l"] != "-" else close
                    vol = int(row["v"]) if row.get("v") and row["v"].isdigit() else 0
                    # 沒成交價就跳過（盤前/暫停）
                    if close <= 0:
                        continue
                    result[tk] = {"open": o, "high": h, "low": lo,
                                  "close": close, "vol": vol * 1000}  # 張 → 股
                except (ValueError, KeyError):
                    continue
        except Exception as e:
            print(f"  fetch_intraday batch {i}: {e}")
        time.sleep(2)  # 保險：3 req / 5 sec rate limit
    return result


def merge_intraday_into_cache(cache, quotes, today_str):
    """
    把盤中 quotes 接到 cache 末尾當「今天」的 row
    回傳新 cache（不修改原物件，避免污染）
    """
    new_cache = {}
    today_ts = pd.Timestamp(today_str)
    for ticker, df in cache.items():
        if df.index[-1].date() == today_ts.date():
            # 已有今天，直接拷貝
            new_cache[ticker] = df.copy()
            continue
        if ticker not in quotes:
            # 盤中沒這檔（沒成交 / 不在 universe）→ 拷貝原 cache
            new_cache[ticker] = df.copy()
            continue
        q = quotes[ticker]
        # 加一行
        new_row = pd.DataFrame({
            "Open": [q["open"]], "High": [q["high"]],
            "Low": [q["low"]], "Close": [q["close"]],
            "Volume": [q["vol"]],
        }, index=[today_ts])
        # 對齊欄位
        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = df[col].iloc[-1]  # 用昨日值填
        new_row = new_row[df.columns]
        new_cache[ticker] = pd.concat([df, new_row])
    return new_cache


def run_scan_once(send_tg=True):
    """跑一次盤中掃描"""
    now = datetime.now(TW_TZ)
    today_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H:%M:%S")
    print(f"\n{'='*70}")
    print(f"盤中掃描 @ {today_str} {timestamp}")
    print(f"{'='*70}")

    # 1. 載 cache + 89.90 params
    print(f"\n[1/4] 載 cache + 89.90 params...")
    if not os.path.exists(CACHE_PATH):
        print(f"❌ {CACHE_PATH} 不存在")
        return None
    raw = pickle.load(open(CACHE_PATH, "rb"))
    params = fetch_gist_strategy()

    # 2. 抓今天盤中報價
    tickers = list(raw.keys())
    print(f"\n[2/4] 抓盤中報價 ({len(tickers)} 檔)...")
    t_start = time.time()
    quotes = fetch_intraday_quotes(tickers)
    print(f"  抓到 {len(quotes)}/{len(tickers)} 檔 ({time.time()-t_start:.1f}s)")

    if len(quotes) < 100:
        print(f"  ⚠️ 盤中報價少於 100，可能盤前/盤後/節日")
        return None

    # 3. 把盤中接到 cache + precompute
    print(f"\n[3/4] 接 cache + precompute...")
    cache_with_today = merge_intraday_into_cache(raw, quotes, today_str)
    _lens = [len(v) for v in cache_with_today.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in cache_with_today.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    print(f"  precompute done ({len(data_dict)} stocks × {pre['n_days']} days)")

    # 4. cpu_replay 找今天的訊號
    print(f"\n[4/4] cpu_replay 找今日訊號...")
    trades = base.cpu_replay(pre, params)
    today_trades = [t for t in trades if t.get("buy_date") == today_str]

    if not today_trades:
        print(f"  ❌ 今日無買入訊號")
        return None

    # 整理訊號
    today_trades.sort(key=lambda t: -float(t.get("score", 0)))
    print(f"\n  🟢 找到 {len(today_trades)} 個今日訊號:")
    for i, t in enumerate(today_trades, 1):
        ticker = t.get("ticker", "?")
        score = t.get("score", 0)
        buy_p = t.get("buy_price", 0)
        print(f"    {i}. {ticker} score={score:.1f} buy@{buy_p:.2f}")

    # Telegram 推
    if send_tg:
        msg = f"📡 *盤中訊號* @ {timestamp}\n89.90 找到 {len(today_trades)} 檔:\n\n"
        for t in today_trades[:5]:
            ticker = t.get("ticker", "?")
            buy_p = t.get("buy_price", 0)
            score = t.get("score", 0)
            msg += f"• `{ticker}` score={score:.1f} @ {buy_p:.2f}\n"
        msg += f"\n⚠️ 立即去券商下單，不要等 D+1 16:35"
        send_telegram(msg)
        print(f"\n  ✅ Telegram 推送完成")

    return today_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="盤中持續每 30 分鐘跑一次")
    parser.add_argument("--no-tg", action="store_true", help="不推 Telegram (debug)")
    args = parser.parse_args()

    if not args.loop:
        # 單次跑
        run_scan_once(send_tg=not args.no_tg)
        return

    # Loop 模式：盤中（09:30 - 13:30）每 30 分鐘跑
    print("=" * 70)
    print("盤中即時掃描 (loop mode) — 每 30 分鐘跑一次")
    print("=" * 70)
    last_signals = {}  # ticker → 已推時間，避免重複推
    while True:
        now = datetime.now(TW_TZ)
        # 只在盤中跑（09:00 - 13:35）
        if now.weekday() >= 5:  # 週末
            print(f"\n{now.strftime('%H:%M')} 週末休市，sleep 1 hour")
            time.sleep(3600)
            continue
        if now.hour < 9 or (now.hour == 13 and now.minute > 35) or now.hour >= 14:
            print(f"\n{now.strftime('%H:%M')} 非盤中時段，sleep 30 min")
            time.sleep(1800)
            continue

        try:
            sigs = run_scan_once(send_tg=not args.no_tg)
            if sigs:
                today_str = now.strftime("%Y-%m-%d")
                # 更新已推紀錄（同一檔同一天只推一次）
                for t in sigs:
                    last_signals[t.get("ticker")] = today_str
        except Exception as e:
            import traceback
            print(f"\n❌ scan error: {e}")
            traceback.print_exc()

        # 等 30 分鐘
        print(f"\n下次掃描：{(now + timedelta(minutes=30)).strftime('%H:%M')}")
        time.sleep(1800)


if __name__ == "__main__":
    main()
