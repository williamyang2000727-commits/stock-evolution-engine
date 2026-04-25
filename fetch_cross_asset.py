"""
V37 跨資產 Gate — 抓 3 個外部訊號
用法：C:\\stock-evolution> python fetch_cross_asset.py
輸出：C:\\stock-evolution\\cross_asset_data.pkl
       {
         "tsm_close": pd.Series,        # TSM ADR 紐約收盤（USD）
         "tsm_change_pct": pd.Series,    # TSM 日漲跌 %
         "soxx_close": pd.Series,
         "soxx_change_pct": pd.Series,
         "nvda_close": pd.Series,
         "nvda_change_pct": pd.Series,
         "usdtwd_close": pd.Series,
         "usdtwd_change_pct": pd.Series,
         # 選擇權（測 free tier）：
         "vixtwn": pd.Series,            # 若 free 可抓
         "foreign_pcr": pd.Series,        # 若 free 可抓
       }

資料源：
  1. TSM / SOXX / NVDA / USD/TWD: yfinance（30 秒，全免費）
  2. VIXTWN: FinMind TaiwanVIXFutures or TWSE openapi（測試）
  3. 外資選擇權 PCR: FinMind TaiwanOptionInstitutionalInvestors（測試 free tier）
"""
import os, sys, pickle, time, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

OUT = os.path.join(os.path.expanduser("~"), "stock-evolution", "cross_asset_data.pkl")
LOG = os.path.join(os.path.expanduser("~"), "stock-evolution", "cross_asset_fetch.log")

START = "2020-01-01"
END = "2026-04-30"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def fetch_yfinance(ticker: str, start: str, end: str):
    """用 yfinance 抓單一商品收盤"""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            return None
        # 處理 multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        log(f"  yfinance {ticker} failed: {e}")
        return None


def test_finmind_options():
    """測試 FinMind free tier 能否抓選擇權資料"""
    import requests
    BASE = "https://api.finmindtrade.com/api/v4/data"

    # 測 TaiwanOptionInstitutionalInvestors
    log(f"\n[Test] FinMind TaiwanOptionInstitutionalInvestors（外資選擇權 PCR）")
    try:
        r = requests.get(BASE, params={
            "dataset": "TaiwanOptionInstitutionalInvestors",
            "data_id": "TXO",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }, timeout=30)
        log(f"  HTTP {r.status_code}")
        j = r.json()
        if j.get("status") == 200:
            log(f"  ✅ 可抓 free")
            log(f"  範例: {j.get('data', [])[:1]}")
            return "InstitutionalInvestors", True
        else:
            log(f"  ❌ {j.get('msg', 'unknown')}")
            return "InstitutionalInvestors", False
    except Exception as e:
        log(f"  exception: {e}")
        return "InstitutionalInvestors", False


def test_finmind_vix():
    """測試 VIXTWN 來源"""
    import requests
    BASE = "https://api.finmindtrade.com/api/v4/data"

    log(f"\n[Test] FinMind TaiwanFuturesDaily（找 VIXTWN）")
    try:
        # FinMind 應該有 TaiwanVIXFutures 或類似
        for ds in ["TaiwanVIX", "TaiwanFuturesDaily"]:
            r = requests.get(BASE, params={
                "dataset": ds,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            }, timeout=30)
            log(f"  {ds}: HTTP {r.status_code}")
            try:
                j = r.json()
                if j.get("status") == 200:
                    data = j.get("data", [])
                    if data:
                        log(f"  ✅ {ds} 可抓，sample: {data[0]}")
                        return ds, True
            except Exception:
                pass
    except Exception as e:
        log(f"  exception: {e}")
    return None, False


def main():
    log("=" * 60)
    log("V37 跨資產 Gate — 抓 3 個外部訊號")
    log("=" * 60)

    result = {}

    # === 1. yfinance 美股 + 匯率 ===
    log(f"\n[1/3] 抓 yfinance 美股 + 匯率（{START} ~ {END}）...")
    yf_tickers = {
        "TSM": "TSM",        # 台積電 ADR
        "SOXX": "SOXX",       # 半導體 ETF
        "NVDA": "NVDA",
        "TWD": "TWD=X",       # USD/TWD
    }
    for name, tk in yf_tickers.items():
        log(f"  抓 {name} ({tk})...")
        df = fetch_yfinance(tk, START, END)
        if df is None or df.empty:
            log(f"    ❌ 失敗")
            continue
        if "Close" not in df.columns:
            log(f"    ❌ 沒有 Close 欄位 columns={list(df.columns)}")
            continue
        close = df["Close"].dropna()
        if len(close) == 0:
            log(f"    ❌ Close 全 NaN")
            continue
        change_pct = close.pct_change() * 100
        result[f"{name.lower()}_close"] = close
        result[f"{name.lower()}_change_pct"] = change_pct
        log(f"    ✅ {len(close)} 天，最後 {close.index[-1].date()} = {close.iloc[-1]:.2f}")

    # === 2. FinMind 選擇權 PCR ===
    log(f"\n[2/3] 測試 FinMind 選擇權 free tier...")
    pcr_dataset, pcr_ok = test_finmind_options()
    if pcr_ok:
        log(f"  → 抓 PCR 全期...")
        import requests
        BASE = "https://api.finmindtrade.com/api/v4/data"
        try:
            r = requests.get(BASE, params={
                "dataset": pcr_dataset,
                "data_id": "TXO",
                "start_date": START,
                "end_date": END,
            }, timeout=120)
            j = r.json()
            if j.get("status") == 200:
                df = pd.DataFrame(j.get("data", []))
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    log(f"    ✅ {len(df)} rows, columns = {list(df.columns)}")
                    result["pcr_raw"] = df
                else:
                    log(f"    ⚠️ data 空")
            else:
                log(f"    ❌ status {j.get('status')}: {j.get('msg')}")
        except Exception as e:
            log(f"    exception: {e}")

    # === 3. VIX TW ===
    log(f"\n[3/3] 測試 VIXTWN 資料源...")
    vix_ds, vix_ok = test_finmind_vix()
    if vix_ok:
        log(f"  → 抓 VIX 全期...")
        import requests
        BASE = "https://api.finmindtrade.com/api/v4/data"
        try:
            r = requests.get(BASE, params={
                "dataset": vix_ds,
                "start_date": START,
                "end_date": END,
            }, timeout=120)
            j = r.json()
            if j.get("status") == 200:
                df = pd.DataFrame(j.get("data", []))
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    log(f"    ✅ {len(df)} rows")
                    result["vix_raw"] = df
        except Exception as e:
            log(f"    exception: {e}")

    # === 存檔 ===
    log(f"\n=== 完成 ===")
    log(f"已抓 {len(result)} 個 keys: {list(result.keys())}")
    with open(OUT, "wb") as f:
        pickle.dump(result, f)
    log(f"存到 {OUT}")

    # 摘要
    log(f"\n=== 摘要 ===")
    for k, v in result.items():
        if isinstance(v, pd.Series):
            log(f"  {k}: Series len {len(v)}, range {v.index[0].date()} ~ {v.index[-1].date()}")
        elif isinstance(v, pd.DataFrame):
            log(f"  {k}: DataFrame {v.shape}, range {v.index[0].date()} ~ {v.index[-1].date()}")


if __name__ == "__main__":
    main()
