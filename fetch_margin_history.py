"""
V34 Margin Gambit — 抓 FinMind 個股融資融券完整歷史 (Windows 版)
用法：C:\stock-evolution> python fetch_margin_history.py
輸出：C:\stock-evolution\margin_data_full.pkl  {stock_id: DataFrame(date, 15 欄 margin)}

日期範圍自動從 stock_data_cache.pkl 讀取，和 GPU cache 1:1 對齊。
FinMind 免費 API rate limit: ~600 calls/hour，script 內建 time.sleep(0.4) 防 block。
"""
import os, sys, time, pickle
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockMarginPurchaseShortSale"
CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
OUT = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_data_full.pkl")
LOG = os.path.join(os.path.expanduser("~"), "stock-evolution", "margin_fetch.log")


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
    """從 stock_data_cache.pkl 讀：
    - qualified_tickers: 有 ≥ 1500 天的股票（GPU 實際 universe）
    - start / end: 這批股票中最早/最晚的 OHLCV 日期（含 buffer）
    """
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
        # tail(1500) 的第一個日期
        start = v.index[-1500].date() if len(v) >= 1500 else v.index[0].date()
        end = v.index[-1].date()
        qualified.append(tk)
        all_starts.append(start)
        all_ends.append(end)
    if not qualified:
        raise RuntimeError("沒有任何股票有 ≥ 1500 天資料")
    # 取全部合格股票中最早的 start / 最晚的 end，確保涵蓋所有 GPU 用的日期
    window_start = min(all_starts)
    window_end = max(all_ends)
    return sorted(qualified), str(window_start), str(window_end)


def fetch_one(stock_id, start, end, retry=3):
    for _ in range(retry):
        try:
            r = requests.get(BASE, params={
                "dataset": DATASET,
                "data_id": stock_id,
                "start_date": start,
                "end_date": end,
            }, timeout=30)
            j = r.json()
            if j.get("status") == 200:
                return j.get("data", [])
            if j.get("status") == 402:
                log(f"  Rate limit (402) on {stock_id}, sleep 60s")
                time.sleep(60)
                continue
        except Exception as e:
            log(f"  retry {stock_id}: {e}")
            time.sleep(2)
    return None


def main():
    import urllib3
    urllib3.disable_warnings()
    log("=== V34 Margin Gambit 資料抓取啟動 ===")

    tickers, start, end = load_cache_info()
    log(f"Cache 窗口: {start} -> {end}")
    log(f"合格股票（>= 1500 天）: {len(tickers)}")
    log(f"輸出: {OUT}")

    # Resume
    result = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, "rb") as f:
                result = pickle.load(f)
            log(f"Resume: 已有 {len(result)} 檔")
        except Exception:
            result = {}

    remaining = [t for t in tickers if t not in result]
    log(f"剩餘要抓: {len(remaining)}")
    if not remaining:
        log("全部已抓完，直接結束")
        return

    t_start = time.time()
    fail = []
    for i, tk in enumerate(remaining):
        data = fetch_one(tk, start, end)
        if data is None:
            fail.append(tk)
            continue
        if data:
            result[tk] = pd.DataFrame(data)
        # 每 50 檔 checkpoint
        if (i + 1) % 50 == 0:
            with open(OUT, "wb") as f:
                pickle.dump(result, f)
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (len(remaining) - i - 1) / max(rate, 0.01)
            log(f"  {i+1}/{len(remaining)}  rate={rate:.1f}/s  eta={eta/60:.1f}min  data_cnt={len(result)}")
        # Rate control：免費 tier 600 calls/hour = 1 call / 6s
        # 測試顯示快跑 4s 內 5 次 OK，但長跑要穩，取中間值 0.4s 配合 402 自動處理
        time.sleep(0.4)

    with open(OUT, "wb") as f:
        pickle.dump(result, f)
    log(f"=== 完成 ===")
    log(f"有資料 {len(result)} 檔, 失敗 {len(fail)} 檔")
    if fail:
        log(f"失敗前 20: {fail[:20]}")
    log(f"總耗時: {(time.time()-t_start)/60:.1f} 分鐘")
    log(f"檔案大小: {os.path.getsize(OUT)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
