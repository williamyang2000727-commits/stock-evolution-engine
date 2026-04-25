"""
V37 分點券商主力買超 — 抓 FinMind 個股分點完整歷史
用法：C:\\stock-evolution> python fetch_broker_history.py
輸出：C:\\stock-evolution\\broker_data_full.pkl
       {stock_id: DataFrame(date, BrokerID, 買量, 賣量, 買賣超, ...)}

FinMind dataset: TaiwanStockTradingDailyReport
免費 API rate limit: 600 calls/hour，script 用 6s 間隔 + 402 retry

預估：1726 stocks × 1500 trading days，但 FinMind 一個 query 可以拿一檔股票一段日期
所以是 1726 calls，~3 小時跑完（比 V34 margin 8.7h 快）
"""
import os, sys, time, pickle
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockTradingDailyReport"
CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
OUT = os.path.join(os.path.expanduser("~"), "stock-evolution", "broker_data_full.pkl")
LOG = os.path.join(os.path.expanduser("~"), "stock-evolution", "broker_fetch.log")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_cache_info():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"找不到 {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    qualified = []
    all_starts = []
    all_ends = []
    for k, v in cache.items():
        tk = k.split(".")[0]
        if not (tk and len(tk) == 4 and tk.isdigit()):
            continue
        if len(v) < 1500:
            continue
        start = v.index[-1500].date()
        end = v.index[-1].date()
        qualified.append(tk)
        all_starts.append(start)
        all_ends.append(end)
    if not qualified:
        raise RuntimeError("沒有任何股票有 ≥ 1500 天資料")
    window_start = min(all_starts)
    window_end = max(all_ends)
    return qualified, window_start, window_end


def fetch_one_stock(stock_id: str, start: str, end: str, max_retry: int = 3):
    """
    抓單一股票一段時間的所有分點資料
    Returns: DataFrame 或 None
    """
    params = {
        "dataset": DATASET,
        "data_id": stock_id,
        "start_date": start,
        "end_date": end,
    }
    for attempt in range(max_retry):
        try:
            r = requests.get(BASE, params=params, timeout=60)
            if r.status_code == 402:
                log(f"  [{stock_id}] 402 rate limit，sleep 60s")
                time.sleep(60)
                continue
            if r.status_code != 200:
                log(f"  [{stock_id}] HTTP {r.status_code}: {r.text[:100]}")
                if attempt < max_retry - 1:
                    time.sleep(5)
                    continue
                return None
            j = r.json()
            if j.get("status") != 200:
                if "limit" in str(j.get("msg", "")).lower():
                    log(f"  [{stock_id}] API limit: {j.get('msg')}, sleep 60s")
                    time.sleep(60)
                    continue
                log(f"  [{stock_id}] API msg: {j.get('msg')}")
                return None
            data = j.get("data", [])
            if not data:
                return pd.DataFrame()  # 空但不是錯
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            log(f"  [{stock_id}] exception (attempt {attempt+1}): {e}")
            if attempt < max_retry - 1:
                time.sleep(10)
    return None


def main():
    log("=" * 60)
    log("V37 分點券商主力 — 抓 FinMind 完整歷史")
    log("=" * 60)

    # === 載入現有進度（如果有）===
    existing = {}
    if os.path.exists(OUT):
        log(f"載入既有進度：{OUT}")
        with open(OUT, "rb") as f:
            existing = pickle.load(f)
        log(f"  既有 {len(existing)} 檔資料（resume mode）")

    # === 從 cache 拿股票清單 + 日期範圍 ===
    log(f"從 {CACHE_PATH} 讀取 GPU universe...")
    qualified, window_start, window_end = load_cache_info()
    log(f"合格股票（≥1500 天）: {len(qualified)} 檔")
    log(f"日期範圍: {window_start} ~ {window_end}")

    start_str = window_start.strftime("%Y-%m-%d")
    end_str = window_end.strftime("%Y-%m-%d")

    # === 逐檔抓 ===
    todo = [t for t in qualified if t not in existing]
    log(f"待抓 {len(todo)}/{len(qualified)} 檔（已抓 {len(existing)} 檔）")

    if not todo:
        log("✅ 全部已抓完")
        return

    SLEEP = 6.0  # 每個 call 間隔（FinMind 600/hr = 10/min = 6s/call）
    BATCH_SAVE = 50  # 每 50 檔 save 一次（防 crash）

    start_time = time.time()
    fetched = 0
    failed = []

    for i, ticker in enumerate(todo):
        df = fetch_one_stock(ticker, start_str, end_str)
        if df is None:
            failed.append(ticker)
            log(f"  [{i+1}/{len(todo)}] {ticker}: ❌ 抓取失敗")
        else:
            existing[ticker] = df
            fetched += 1
            elapsed = time.time() - start_time
            rate = fetched / elapsed * 60 if elapsed > 0 else 0
            eta_min = (len(todo) - i - 1) * SLEEP / 60
            n_brokers = len(df["BrokerID"].unique()) if len(df) > 0 and "BrokerID" in df.columns else 0
            log(f"  [{i+1}/{len(todo)}] {ticker}: {len(df)} rows / {n_brokers} brokers ({rate:.1f}/min, ETA {eta_min:.0f}m)")

        # 定期存檔
        if (i + 1) % BATCH_SAVE == 0:
            with open(OUT, "wb") as f:
                pickle.dump(existing, f)
            log(f"  💾 存檔（{len(existing)} 檔累計）")

        time.sleep(SLEEP)

    # 最終存檔
    with open(OUT, "wb") as f:
        pickle.dump(existing, f)
    log(f"")
    log(f"=" * 60)
    log(f"✅ 完成！共抓 {len(existing)} 檔，失敗 {len(failed)} 檔")
    if failed:
        log(f"失敗清單前 20: {failed[:20]}")
    total_min = (time.time() - start_time) / 60
    log(f"總耗時: {total_min:.1f} 分鐘")


if __name__ == "__main__":
    main()
