"""
Cache 完整性檢查 — 找最近 60 天每個 ticker 的資料 gap

背景：2026-04-25 發現達邁(3645.TW) 4/18 整天 K 線從 cache 消失
    → realistic_test 用今天 cache 重模擬時 04-18 沒達邁訊號
    → 換股軌跡分岔，4/21 持倉變聯茂而非 Tab 3 的達邁

這個 script 找：
  1. 最近 60 天每個 ticker 缺哪些交易日
  2. 用「全市場交易日聯集」當基準（多數股都有 = 真交易日）
  3. 列出 gap 嚴重的 ticker（缺 ≥ 1 天）
"""
import os, pickle, sys
from collections import Counter
import pandas as pd

USER_SE = r"C:\stock-evolution" if os.name == "nt" else os.path.expanduser("~/stock-evolution")
if not os.path.isdir(USER_SE):
    USER_SE = os.path.dirname(os.path.abspath(__file__))

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
print(f"載 cache: {CACHE_PATH}")
raw = pickle.load(open(CACHE_PATH, "rb"))
print(f"  共 {len(raw)} 檔股票")

# 1. 找全市場最近 60 天的「真交易日聯集」（多數股有的日子）
WINDOW = 60
all_dates = Counter()
for ticker, df in raw.items():
    if len(df) < WINDOW:
        continue
    recent = df.tail(WINDOW)
    for d in recent.index:
        d_norm = d.normalize() if hasattr(d, "tz") and d.tz else pd.Timestamp(d).normalize()
        if hasattr(d_norm, "tz_localize") and d_norm.tz is not None:
            d_norm = d_norm.tz_localize(None)
        all_dates[d_norm.date()] += 1

# 真交易日 = 出現在 ≥ 50% 股票裡的日期
threshold = len(raw) * 0.5
true_trading_days = sorted([d for d, c in all_dates.items() if c >= threshold])
print(f"\n真交易日數（出現在 ≥ 50% 股票）：{len(true_trading_days)}")
print(f"範圍：{true_trading_days[0]} ~ {true_trading_days[-1]}")

# 只看最近 30 個真交易日
recent_30 = set(true_trading_days[-30:])
print(f"\n檢查最近 30 個交易日：{min(recent_30)} ~ {max(recent_30)}")

# 2. 對每檔股票檢查 gap
gaps_by_ticker = {}
for ticker, df in raw.items():
    if len(df) < WINDOW:
        continue
    recent = df.tail(WINDOW * 2)  # 拉廣一點再切
    df_dates = set()
    for d in recent.index:
        d_norm = d.normalize() if hasattr(d, "tz") and d.tz else pd.Timestamp(d).normalize()
        if hasattr(d_norm, "tz_localize") and d_norm.tz is not None:
            d_norm = d_norm.tz_localize(None)
        df_dates.add(d_norm.date())

    missing = recent_30 - df_dates
    if missing:
        # 看是「整段缺」還是「中間 gap」
        df_recent_dates = sorted([d for d in df_dates if d in recent_30 or d <= max(recent_30)])
        if df_recent_dates:
            last_in_df = max(df_recent_dates)
            # 只算 last_in_df 之前缺的（中間 gap）
            true_mid = {d for d in missing if d <= last_in_df}
            gaps_by_ticker[ticker] = sorted(true_mid)

# 3. 排序 + 列出
print(f"\n{'─' * 70}")
print(f"【最近 30 個交易日有 gap 的 ticker】")
print(f"{'─' * 70}")
sorted_gaps = sorted(gaps_by_ticker.items(), key=lambda kv: -len(kv[1]))
n_with_gaps = sum(1 for v in gaps_by_ticker.values() if v)
print(f"  共 {n_with_gaps}/{len(raw)} 檔有 gap ({n_with_gaps/len(raw)*100:.1f}%)")

if n_with_gaps == 0:
    print(f"  ✅ 沒任何 gap")
else:
    # 印前 30 個 gap 最多的
    print(f"\n  Top 30 gap 最多的：")
    print(f"  {'Ticker':<10} {'Gap數':>5}  缺哪些天")
    print("  " + "─" * 66)
    for tk, miss_days in sorted_gaps[:30]:
        if not miss_days:
            continue
        days_str = ", ".join(str(d) for d in miss_days[:5])
        if len(miss_days) > 5:
            days_str += f" ... (+{len(miss_days)-5})"
        print(f"  {tk:<10} {len(miss_days):>5}  {days_str}")

# 4. 特別檢查最有可能影響 Tab 3 的關鍵 ticker
print(f"\n{'─' * 70}")
print(f"【關鍵股檢查（最近 Tab 3 持倉 + 訊號附近）】")
print(f"{'─' * 70}")
key_tickers = ["3645.TW", "6213.TW", "2484.TW", "6217.TWO", "3498.TWO", "6530.TWO"]
for tk in key_tickers:
    if tk not in raw:
        print(f"  {tk}: ❌ 不在 cache")
        continue
    df = raw[tk]
    df_dates = set()
    for d in df.tail(WINDOW * 2).index:
        d_norm = d.normalize() if hasattr(d, "tz") and d.tz else pd.Timestamp(d).normalize()
        if hasattr(d_norm, "tz_localize") and d_norm.tz is not None:
            d_norm = d_norm.tz_localize(None)
        df_dates.add(d_norm.date())
    missing = sorted(recent_30 - df_dates)
    df_recent = sorted([d for d in df_dates if d in recent_30])
    if missing and df_recent:
        true_mid = [d for d in missing if d <= max(df_recent)]
        if true_mid:
            print(f"  {tk}: ⚠️ 缺 {len(true_mid)} 天 → {true_mid}")
        else:
            print(f"  {tk}: ✅ 完整（最後 {df_recent[-1]}）")
    else:
        print(f"  {tk}: ✅ 完整（最後 {max(df_recent) if df_recent else 'N/A'}）")

# 5. 4/18 那天到底缺哪些股
target_date = pd.Timestamp("2026-04-18").date()
print(f"\n{'─' * 70}")
print(f"【4/18 那天缺資料的股票（聚焦驗證達邁案例）】")
print(f"{'─' * 70}")
missing_on_418 = []
for ticker, df in raw.items():
    df_dates = set()
    for d in df.tail(60).index:
        d_norm = d.normalize() if hasattr(d, "tz") and d.tz else pd.Timestamp(d).normalize()
        if hasattr(d_norm, "tz_localize") and d_norm.tz is not None:
            d_norm = d_norm.tz_localize(None)
        df_dates.add(d_norm.date())
    df_recent = [d for d in df_dates if d >= pd.Timestamp("2026-04-15").date() and d <= pd.Timestamp("2026-04-24").date()]
    if df_recent and target_date not in df_dates and max(df_recent) > target_date:
        missing_on_418.append(ticker)

print(f"  共 {len(missing_on_418)} 檔缺 4/18（中間 gap）")
if missing_on_418:
    print(f"  全部：{missing_on_418[:50]}{'...' if len(missing_on_418) > 50 else ''}")

print(f"\n{'─' * 70}")
print("【建議】")
print(f"{'─' * 70}")
if "3645.TW" in missing_on_418 or "3645.TW" in [tk for tk, _ in sorted_gaps if _]:
    print(f"  🔴 達邁 4/18 缺資料確認 → 跑 update_cache.py 強制重抓近 30 天")
if n_with_gaps > len(raw) * 0.05:
    print(f"  🔴 {n_with_gaps} 檔有 gap ({n_with_gaps/len(raw)*100:.1f}%) → cache 完整性差，重抓全 universe 30 天")
elif n_with_gaps > 0:
    print(f"  🟡 {n_with_gaps} 檔有 gap → 重抓那些 ticker 即可")
else:
    print(f"  ✅ Cache 健康")
