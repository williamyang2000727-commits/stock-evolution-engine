"""手動 append cache 到今天（舊資料完全不動）。

用法：python update_cache.py

會印出更新前後「最後一天」分布對照，確認舊資料沒被動到。
中斷也安全（原子寫入，不會損壞 cache）。

V2（2026-04-26）：append 完雙源驗證 yfinance close vs TWSE/TPEx 官方 close
- 差距 > 1% → 用官方價覆蓋 cache + Telegram 警報
- 防 yfinance adjusted close drift / 抓到半成品
"""
import sys, types
# mock cupy 避免 module load 時的 GPU 初始化（我們只要 append 函式）
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

import os, pickle, json, ssl, urllib.request
from collections import Counter
from gpu_cupy_evolve import append_new_days, CACHE_PATH


# ───────────────────────────────────────────────────────────────────
# 雙源驗證：抓 TWSE / TPEx 官方當日收盤，跟 cache 最後一天 close 比對
# ───────────────────────────────────────────────────────────────────
def fetch_twse_close_today():
    """回 {ticker_no_suffix: {'close': float, 'date': 'YYYYMMDD'}}（上市全市場）"""
    ctx = ssl._create_unverified_context()
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
    try:
        r = urllib.request.urlopen(url, context=ctx, timeout=15)
        d = json.loads(r.read())
    except Exception as e:
        print(f"  ⚠️ TWSE STOCK_DAY_ALL fetch fail: {e}")
        return {}, None
    out = {}
    date = d.get("date")  # 'YYYYMMDD'
    for row in d.get("data", []):
        try:
            tk = row[0].strip()
            close_str = row[7].replace(",", "")
            if close_str in ("--", "", "0"): continue
            out[tk] = {"close": float(close_str), "date": date}
        except Exception:
            continue
    return out, date


def fetch_tpex_close_today():
    """回 {ticker_no_suffix: {'close': float, 'date': 'YYYYMMDD'}}（上櫃全市場）"""
    ctx = ssl._create_unverified_context()
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    try:
        r = urllib.request.urlopen(url, context=ctx, timeout=15)
        d = json.loads(r.read())
    except Exception as e:
        print(f"  ⚠️ TPEx daily close fetch fail: {e}")
        return {}, None
    if not isinstance(d, list) or not d:
        return {}, None
    date = d[0].get("Date")  # 'YYYMMDD' 民國年
    out = {}
    for row in d:
        try:
            tk = row.get("SecuritiesCompanyCode", "").strip()
            close = row.get("Close", "").strip()
            if not tk or not close or close in ("--", "0"): continue
            out[tk] = {"close": float(close), "date": date}
        except Exception:
            continue
    return out, date


def send_telegram_warning(msg):
    """推 Telegram 警報（同 GPU bot）"""
    BOT = os.environ.get("TELEGRAM_BOT_TOKEN", "8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI")
    CHAT = os.environ.get("TELEGRAM_CHAT_ID", "5785839733")
    try:
        ctx = ssl._create_unverified_context()
        url = f"https://api.telegram.org/bot{BOT}/sendMessage"
        data = json.dumps({"chat_id": CHAT, "text": msg[:4000]}).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, context=ctx, timeout=10)
    except Exception as e:
        print(f"  ⚠️ Telegram fail: {e}")


def cross_validate_cache(cache, threshold_pct=1.0):
    """雙源驗證 cache 最後一天 close vs TWSE/TPEx 官方

    threshold_pct: 差距 > 此 % 視為飄
    回 (n_checked, n_drift, drift_list, fixed)
    fixed = 修正後的 cache（in-place modified）
    """
    print()
    print("=== 雙源驗證（yfinance cache vs TWSE/TPEx 官方）===")
    twse_close, twse_date = fetch_twse_close_today()
    tpex_close, tpex_date = fetch_tpex_close_today()
    print(f"  TWSE: {len(twse_close)} 檔, date={twse_date}")
    print(f"  TPEx: {len(tpex_close)} 檔, date={tpex_date}")
    if not twse_close and not tpex_close:
        print("  ⚠️ 兩個官方源都掛了，跳過驗證")
        return 0, 0, [], cache

    n_checked = 0
    n_drift = 0
    drift_list = []  # [(ticker, yf_close, official_close, diff_pct)]
    n_fixed = 0

    for ticker, df in cache.items():
        if df is None or len(df) == 0: continue
        # 拆 ticker：6213.TW → 6213, 2484.TW → 2484, 3264.TWO → 3264
        if "." in ticker:
            tk_no_suffix, suffix = ticker.split(".", 1)
        else:
            tk_no_suffix = ticker
            suffix = "TW"

        # 找官方 close
        official = twse_close.get(tk_no_suffix) or tpex_close.get(tk_no_suffix)
        if not official:
            continue  # 不在 TWSE/TPEx（可能是 ETF 或下市）

        # cache 最後一天 close
        try:
            yf_close = float(df["Close"].iloc[-1])
        except Exception:
            continue
        if yf_close <= 0: continue

        n_checked += 1
        official_close = official["close"]
        diff_pct = abs(yf_close - official_close) / official_close * 100

        if diff_pct > threshold_pct:
            n_drift += 1
            drift_list.append((ticker, yf_close, official_close, diff_pct))
            # 直接修 cache 最後一天 close（用官方價）
            df.iloc[-1, df.columns.get_loc("Close")] = official_close
            n_fixed += 1

    print(f"  ✅ 檢查 {n_checked} 檔")
    if n_drift == 0:
        print(f"  ✅ 全部通過（threshold {threshold_pct}%）")
    else:
        print(f"  ❌ {n_drift} 檔飄超過 {threshold_pct}%（已用官方價覆蓋 cache）")
        # 印前 10 大
        drift_list.sort(key=lambda x: -x[3])
        for tk, yf_c, off_c, dp in drift_list[:10]:
            print(f"    {tk}: yf={yf_c:.2f} vs 官方={off_c:.2f} (差 {dp:.2f}%)")
        # 推 Telegram
        msg_lines = [f"⚠️ update_cache 雙源驗證 {n_drift} 檔飄 > {threshold_pct}%（已修）"]
        for tk, yf_c, off_c, dp in drift_list[:5]:
            msg_lines.append(f"  {tk}: yf={yf_c:.2f} → 官方={off_c:.2f} (-{dp:.1f}%)")
        if n_drift > 5:
            msg_lines.append(f"  ... 還有 {n_drift - 5} 檔")
        send_telegram_warning("\n".join(msg_lines))

    return n_checked, n_drift, drift_list, cache

def dump_summary(label, cache):
    if not cache:
        print(f"  {label}: (無 cache)")
        return
    c = Counter(df.index[-1].date() for df in cache.values() if len(df))
    print(f"  {label}: 共 {len(cache)} 檔")
    for dt, n in sorted(c.items())[-6:]:
        print(f"    {dt}: {n} 檔")

def main():
    """主流程，包進 main 避免 import 時被觸發"""
    print(f"Cache path: {CACHE_PATH}")
    print()

    if not os.path.exists(CACHE_PATH):
        print("cache 不存在，無法 append")
        sys.exit(1)

# 讀更新前狀態
    with open(CACHE_PATH, "rb") as f:
        before = pickle.load(f)
    print("=== 更新前 ===")
    dump_summary("最後一天分布", before)

    # 重複日期檢查 + 自動清理（修舊 bug 造成的重複 row）
    print()
    dup_count = 0
    cleaned = 0
    for t in before:
        df = before[t]
        if df is None or len(df) == 0: continue
        n_dup = df.index.duplicated().sum()
        if n_dup > 0:
            dup_count += n_dup
            before[t] = df[~df.index.duplicated(keep='first')]
            cleaned += 1
    if cleaned > 0:
        print(f"⚠️ 發現 {cleaned} 檔 cache 有重複日期（共 {dup_count} 筆），已用 keep='first' 清理（保留最早那筆）")
        # 寫回
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(before, f)
        os.replace(tmp, CACHE_PATH)
        print(f"✅ 清理後寫回 cache")
    else:
        print(f"✅ 無重複日期，cache 乾淨")

    # 抓幾檔樣本存起來比對（確認舊資料沒變）
    sample_tickers = list(before.keys())[:3]
    sample_before = {t: before[t].copy() for t in sample_tickers}

    print()
    print("=== 開始 append（只抓新天，舊不動）===")
    cache, n = append_new_days(CACHE_PATH)
    print(f"  {n} 檔有新資料被 append")

    if cache is None:
        print("cache 讀取失敗")
        sys.exit(1)

    print()
    print("=== 更新後 ===")
    dump_summary("最後一天分布", cache)

    # 驗證舊資料未變（兩邊都剝 tz 再比對，避免 tz-aware vs naive 衝突）
    print()
    print("=== 舊資料未變驗證（樣本）===")
    for t in sample_tickers:
        old_df = sample_before[t].copy()
        new_df = cache[t].copy()
        # 統一 naive
        if old_df.index.tz is not None:
            old_df.index = old_df.index.tz_localize(None)
        if new_df.index.tz is not None:
            new_df.index = new_df.index.tz_localize(None)
        # 取新 cache 裡對應舊長度的前 N 列（舊資料應該原封不動在最前面）
        new_head = new_df.iloc[:len(old_df)]
        # 比對 Close 值（最敏感指標）
        match = (
            len(old_df) == len(new_head)
            and (old_df.index.values == new_head.index.values).all()
            and (old_df["Close"].values == new_head["Close"].values).all()
        )
        added = len(new_df) - len(old_df)
        print(f"  {t}: 舊 {len(old_df)} 天 {'✅ 完全一致' if match else '❌ 被動到了!'}, 新增 {added} 天")

    # ───────────────────────────────────────────────────────────────────
    # 雙源驗證（V2 新增，2026-04-26）
    # ───────────────────────────────────────────────────────────────────
    n_checked, n_drift, drift_list, cache = cross_validate_cache(cache, threshold_pct=1.0)

    # 如果有飄就重寫 cache
    if n_drift > 0:
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp, CACHE_PATH)
        print(f"  ✅ 雙源驗證後重寫 cache（修正 {n_drift} 檔最後一天 close）")


if __name__ == "__main__":
    main()
