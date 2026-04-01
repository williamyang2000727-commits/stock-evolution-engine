"""
下載 TWSE/TPEx 歷史日K（2022/1 ~ 2025/5）
用官方批量 API，一次抓一天全市場資料
"""
import requests
import json
import pickle
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 4, 1)

def get_trading_days(start, end):
    """生成所有可能的交易日（週一到週五）"""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # 週一到週五
            days.append(d)
        d += timedelta(days=1)
    return days

def fetch_twse_day(date_str):
    """抓 TWSE 上市股票某一天的全市場資料"""
    url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={date_str}&type=ALLBUT0999"
    try:
        r = requests.get(url, timeout=15, verify=False)
        data = r.json()
        if data.get("stat") != "OK":
            return {}

        # 新版 API 用 tables 陣列，個股在最大的那個 table
        rows = []
        if "tables" in data:
            for t in data["tables"]:
                if t.get("data") and len(t["data"]) > 100:
                    rows = t["data"]
                    break
        else:
            # 舊版 API 用 data9 等 key
            for key in ["data9", "data8", "data7", "data5"]:
                if key in data and data[key] and len(data[key]) > 100:
                    rows = data[key]
                    break

        if not rows:
            return {}

        result = {}
        for row in rows:
            try:
                code = row[0].strip()
                # 跳過非股票（ETF等）
                if len(code) != 4 or not code.isdigit():
                    continue

                ticker = code + ".TW"

                # 處理數字（移除逗號）
                def parse_num(s):
                    return float(str(s).replace(",", "").replace("--", "0").replace("X", "0").strip() or "0")

                vol = parse_num(row[2])  # 成交股數
                o = parse_num(row[5])    # 開盤
                h = parse_num(row[6])    # 最高
                l = parse_num(row[7])    # 最低
                c = parse_num(row[8])    # 收盤

                if o > 0 and c > 0 and vol > 0:
                    result[ticker] = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol}
            except:
                continue

        return result
    except Exception as e:
        return {}

def fetch_tpex_day(date_str):
    """抓 TPEx 上櫃股票某一天的全市場資料"""
    # TPEx 用民國年格式：111/01/03
    dt = datetime.strptime(date_str, "%Y%m%d")
    roc_year = dt.year - 1911
    roc_date = f"{roc_year}/{dt.month:02d}/{dt.day:02d}"

    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&o=json&d={roc_date}"
    try:
        r = requests.get(url, timeout=15, verify=False)
        data = r.json()

        # 找最大的 table（個股資料）
        rows = []
        for t in data.get("tables", []):
            tdata = t.get("data", [])
            if len(tdata) > len(rows):
                rows = tdata
        if not rows:
            return {}

        result = {}
        for row in rows:
            try:
                code = str(row[0]).strip()
                if len(code) != 4 or not code.isdigit():
                    continue

                ticker = code + ".TWO"

                def parse_num(s):
                    return float(str(s).replace(",", "").replace("--", "0").replace("---", "0").strip() or "0")

                c = parse_num(row[2])    # 收盤
                o = parse_num(row[4])    # 開盤
                h = parse_num(row[5])    # 最高
                l = parse_num(row[6])    # 最低
                vol = parse_num(row[8])  # 成交股數（row[7]是均價）

                if o > 0 and c > 0 and vol > 0:
                    result[ticker] = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol}
            except:
                continue

        return result
    except Exception as e:
        return {}

def main():
    print(f"下載歷史資料：{START_DATE.date()} ~ {END_DATE.date()}")

    # 檢查是否有已下載的進度
    progress_file = os.path.join(os.path.dirname(CACHE_PATH), "download_progress.pkl")
    all_data = {}  # {ticker: {date: {OHLCV}}}
    last_date = None

    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            saved = pickle.load(f)
            all_data = saved.get("data", {})
            last_date = saved.get("last_date")
            print(f"繼續上次進度：已下載到 {last_date}，{len(all_data)} 檔")

    trading_days = get_trading_days(START_DATE, END_DATE)
    print(f"預計交易日：{len(trading_days)} 天")

    success = 0
    skip = 0
    fail = 0

    for i, day in enumerate(trading_days):
        date_str = day.strftime("%Y%m%d")

        # 跳過已下載的
        if last_date and date_str <= last_date:
            skip += 1
            continue

        print(f"  下載 {date_str} ...", end=" ", flush=True)

        # TWSE 上市
        twse = fetch_twse_day(date_str)
        time.sleep(0.5)  # 避免被擋

        # TPEx 上櫃
        tpex = fetch_tpex_day(date_str)
        time.sleep(0.5)

        total = len(twse) + len(tpex)
        if total == 0:
            # 可能是假日
            fail += 1
            if fail > 5:
                fail = 0  # 重置連續失敗
            continue

        fail = 0
        success += 1

        # 存入 all_data
        for ticker, ohlcv in {**twse, **tpex}.items():
            if ticker not in all_data:
                all_data[ticker] = {}
            all_data[ticker][date_str] = ohlcv

        # 每 10 天存一次進度
        if success % 10 == 0:
            with open(progress_file, "wb") as f:
                pickle.dump({"data": all_data, "last_date": date_str}, f)

        elapsed_days = success
        total_days = len(trading_days) - skip
        pct = elapsed_days / max(total_days, 1) * 100
        print(f"上市{len(twse)} 上櫃{len(tpex)} | {elapsed_days}/{total_days} ({pct:.0f}%) | {len(all_data)}檔")

    print(f"\n\n下載完成！{len(all_data)} 檔股票")

    # 轉換成 DataFrame 格式（跟 yfinance 一致）
    print("轉換格式中...")
    cache = {}
    for ticker, day_data in all_data.items():
        if len(day_data) < 100:  # 至少 100 天資料
            continue

        dates_sorted = sorted(day_data.keys())
        df = pd.DataFrame([day_data[d] for d in dates_sorted],
                         index=pd.to_datetime(dates_sorted, format="%Y%m%d"),
                         columns=["Open", "High", "Low", "Close", "Volume"])
        cache[ticker] = df

    print(f"有效股票：{len(cache)} 檔")

    # 存檔
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已存到：{CACHE_PATH}")

    # 統計
    lens = [len(v) for v in cache.values()]
    print(f"天數範圍：{min(lens)} ~ {max(lens)} 天")
    print(f"中位數：{sorted(lens)[len(lens)//2]} 天")

    # 清理進度檔
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print("\n完成！可以跑 GPU 了。")

if __name__ == "__main__":
    main()
