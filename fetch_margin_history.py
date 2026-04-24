"""
V34 Margin Gambit — 抓 FinMind 個股融資融券完整歷史 (Windows 版)
用法：C:\stock-evolution> python fetch_margin_history.py
輸出：C:\stock-evolution\margin_data_full.pkl  {stock_id: DataFrame(date, 15 欄 margin)}
"""
import os, time, pickle
import requests
import pandas as pd
from datetime import datetime

BASE = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockMarginPurchaseShortSale"
START = "2020-01-01"
END = datetime.now().strftime("%Y-%m-%d")
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


def load_ticker_universe():
    """從已存在的 stock_data_cache.pkl 拿股票清單（跟 GPU 一致，不另打 TWSE/TPEx）"""
    cache_path = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"找不到 {cache_path}，請先跑過 GPU 讓 cache 建立")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    # cache key 是 "2330.TW" 或 "3264.TWO"，我們要純數字代號
    tickers = set()
    for k in cache.keys():
        tk = k.split(".")[0]
        if tk and len(tk) == 4 and tk.isdigit():
            tickers.add(tk)
    return sorted(tickers)


def fetch_one(stock_id, retry=3):
    for _ in range(retry):
        try:
            r = requests.get(BASE, params={
                "dataset": DATASET,
                "data_id": stock_id,
                "start_date": START,
                "end_date": END,
            }, timeout=30)
            j = r.json()
            if j.get("status") == 200:
                return j.get("data", [])
            # 402 = 超過 free tier 額度
            if j.get("status") == 402:
                log(f"  ⚠️ Rate limit (402) on {stock_id}, sleep 60s")
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
    log(f"期間: {START} -> {END}")
    log(f"輸出: {OUT}")

    # Resume: 已有檔案 load 進來，skip 已完成的
    result = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, "rb") as f:
                result = pickle.load(f)
            log(f"Resume: 已有 {len(result)} 檔")
        except Exception:
            result = {}

    log("從 stock_data_cache.pkl 讀股票清單...")
    tickers = load_ticker_universe()
    log(f"總股票數: {len(tickers)}")

    remaining = [t for t in tickers if t not in result]
    log(f"剩餘要抓: {len(remaining)}")
    if not remaining:
        log("全部已抓完，直接結束")
        return

    t_start = time.time()
    fail = []
    for i, tk in enumerate(remaining):
        data = fetch_one(tk)
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
        # Rate control: FinMind free tier 約 600 calls/hour，每 call 間隔 6s 即安全
        # 但測試顯示 0.5s 也能撐，保守用 0.3s
        time.sleep(0.3)

    with open(OUT, "wb") as f:
        pickle.dump(result, f)
    log(f"完成! 有資料 {len(result)} 檔, 失敗 {len(fail)} 檔")
    if fail:
        log(f"失敗前 20: {fail[:20]}")
    log(f"總耗時: {(time.time()-t_start)/60:.1f} 分鐘")
    log(f"檔案大小: {os.path.getsize(OUT)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
