"""
基本面挑戰 — 月營收 sanity test
用法：C:\\stock-evolution> python sanity_test_revenue.py

目的：30 分鐘內驗證「月營收 YoY」對 89.90 的 133 筆 trades 有沒有 alpha

流程：
  1. 確認 FinMind free 支援 TaiwanStockMonthRevenue（首先驗證資料源）
  2. 對 89.90 全期 133 筆 trades，每筆查 buy_date 前最近一次公佈的月營收 YoY
  3. Spearman correlation: YoY% vs trade actual return
  4. Conditional WR: YoY > 0 vs YoY <= 0
  5. 判定：|Spearman| >= 0.05 → GREEN（值得進 backfill）
            < 0.05 → RED（沒 alpha，跟 Kronos 同下場）

學乖（V34/V36/V38 教訓）：
  - 不用 80/20 split（V36 假象）
  - 全期 Spearman + conditional（V34 sanity 風格）
  - GREEN 才繼續，否則 30 分鐘下定論
"""
import os, sys, time, json, pickle
import requests
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
REVENUE_PKL = os.path.join(USER_SE, "monthly_revenue.pkl")
BASE = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockMonthRevenue"


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def fetch_revenue_one(stock_id: str, start: str = "2019-01-01"):
    """抓單檔月營收歷史"""
    params = {
        "dataset": DATASET,
        "data_id": stock_id,
        "start_date": start,
    }
    r = requests.get(BASE, params=params, timeout=30)
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    j = r.json()
    if j.get("status") != 200:
        return None, j.get("msg", "unknown")
    data = j.get("data", [])
    if not data:
        return None, "no data"
    df = pd.DataFrame(data)
    return df, "ok"


def main():
    print("=" * 70)
    print("基本面挑戰 — 月營收 sanity test")
    print("=" * 70)

    # === Step 1: FinMind 可用性測試 ===
    print(f"\n[1/5] 測試 FinMind free 是否支援 TaiwanStockMonthRevenue...")
    test_df, msg = fetch_revenue_one("2330", start="2024-01-01")
    if test_df is None:
        print(f"  ❌ FinMind free 不支援月營收：{msg}")
        print(f"  → 需要 sponsor 1099 NTD/月才能用，跟 V37 broker 一樣下場")
        print(f"  → 接受 89.90 final，停手")
        return
    print(f"  ✅ FinMind free 支援！測試 2330 拿到 {len(test_df)} 筆")
    print(f"  欄位: {list(test_df.columns)}")
    print(f"  範例: {test_df.iloc[0].to_dict()}")

    # === Step 2: 跑 89.90 拿 trades ===
    print(f"\n[2/5] 跑 89.90 cpu_replay...")
    params = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    print(f"  89.90 trades: {len(completed)} 筆")

    # 收集所有出現的 ticker（去除 .TW/.TWO 後綴）
    unique_stocks = sorted(set(t.get("ticker", "").split(".")[0] for t in completed if t.get("ticker")))
    unique_stocks = [s for s in unique_stocks if s and len(s) == 4 and s.isdigit()]
    print(f"  涉及 {len(unique_stocks)} 檔股票")

    # === Step 3: 抓這些股票的月營收 ===
    print(f"\n[3/5] 抓 {len(unique_stocks)} 檔月營收（FinMind free 600/hr，~4s/檔）...")
    if os.path.exists(REVENUE_PKL):
        print(f"  發現 cache {REVENUE_PKL}")
        revenue_data = pickle.load(open(REVENUE_PKL, "rb"))
        # 補抓沒有的
        missing = [s for s in unique_stocks if s not in revenue_data]
        print(f"  cache 已有 {len(revenue_data)} 檔，缺 {len(missing)} 檔")
    else:
        revenue_data = {}
        missing = unique_stocks

    if missing:
        for i, sid in enumerate(missing):
            df, msg = fetch_revenue_one(sid, start="2019-01-01")
            if df is not None:
                revenue_data[sid] = df
                print(f"  [{i+1}/{len(missing)}] {sid} ✅ ({len(df)} 筆)")
            else:
                print(f"  [{i+1}/{len(missing)}] {sid} ❌ {msg}")
                if "402" in str(msg) or "limit" in str(msg).lower():
                    print(f"  → rate limit 達到，存當前進度後等下次再跑")
                    break
            time.sleep(4)  # FinMind rate limit
            # 每 30 檔存一次（防中斷）
            if (i + 1) % 30 == 0:
                pickle.dump(revenue_data, open(REVENUE_PKL, "wb"))

        pickle.dump(revenue_data, open(REVENUE_PKL, "wb"))
        print(f"  ✅ 抓完，存到 {REVENUE_PKL}")

    print(f"  最終 {len(revenue_data)} 檔有月營收資料")

    # === Step 4: 對每筆 trade 計算 buy_date 前最近的 YoY ===
    print(f"\n[4/5] 對 {len(completed)} 筆 trades 計算 YoY...")
    rows = []
    for t in completed:
        ticker_full = t.get("ticker", "")
        ticker = ticker_full.split(".")[0]
        if ticker not in revenue_data:
            continue
        rev_df = revenue_data[ticker].copy()
        bd_str = t.get("buy_date", "")
        try:
            bd = pd.to_datetime(bd_str)
        except Exception:
            continue

        # FinMind 月營收欄位通常：date, stock_id, country, revenue, revenue_month, revenue_year, ...
        # 'date' 通常是公佈日期，'revenue_month' 是該營收所屬月份
        # YoY = 本月營收 / 去年同月營收 - 1

        # 確保有 date column
        if "date" not in rev_df.columns:
            continue
        rev_df["date"] = pd.to_datetime(rev_df["date"])

        # 找 buy_date 之前最近一次 announcement
        before = rev_df[rev_df["date"] < bd].sort_values("date")
        if len(before) < 13:  # 至少 13 個月才能算 YoY
            continue
        latest = before.iloc[-1]
        # 找 12 個月前的同月營收
        if "revenue_year" in latest and "revenue_month" in latest:
            ry = int(latest["revenue_year"])
            rm = int(latest["revenue_month"])
            yoy_match = rev_df[(rev_df["revenue_year"] == ry - 1) & (rev_df["revenue_month"] == rm)]
            if len(yoy_match) == 0:
                continue
            cur_rev = float(latest["revenue"])
            prev_rev = float(yoy_match.iloc[0]["revenue"])
            if prev_rev <= 0:
                continue
            yoy_pct = (cur_rev / prev_rev - 1) * 100
        else:
            continue

        rows.append({
            "ticker": ticker_full,
            "buy_date": bd_str,
            "rev_announcement": str(latest["date"].date()),
            "yoy_pct": yoy_pct,
            "actual_return": float(t.get("return", 0)),
        })

    df_yoy = pd.DataFrame(rows)
    print(f"  最終 {len(df_yoy)} 筆有 YoY 資料")

    if len(df_yoy) < 30:
        print(f"  ❌ 樣本太少，無法可靠 sanity")
        return

    # === Step 5: Sanity 統計 ===
    print(f"\n[5/5] Sanity 統計...")
    yoy = df_yoy["yoy_pct"].values
    rets = df_yoy["actual_return"].values

    print(f"\n  YoY 分布: min={yoy.min():.1f}%, p25={np.percentile(yoy,25):.1f}%, "
          f"median={np.median(yoy):.1f}%, p75={np.percentile(yoy,75):.1f}%, max={yoy.max():.1f}%")
    print(f"  Return 分布: min={rets.min():.1f}%, p25={np.percentile(rets,25):.1f}%, "
          f"median={np.median(rets):.1f}%, p75={np.percentile(rets,75):.1f}%, max={rets.max():.1f}%")

    # Spearman
    sp_corr, sp_p = scipy_stats.spearmanr(yoy, rets)
    print(f"\n  Spearman correlation: {sp_corr:+.4f} (p={sp_p:.4f})")

    # Conditional
    pos_rets = rets[yoy > 0]
    neg_rets = rets[yoy <= 0]
    if len(pos_rets) >= 5 and len(neg_rets) >= 5:
        wr_pos = (pos_rets > 0).mean() * 100
        wr_neg = (neg_rets > 0).mean() * 100
        avg_pos = pos_rets.mean()
        avg_neg = neg_rets.mean()
        print(f"  Conditional WR (YoY > 0 vs <= 0):")
        print(f"    YoY > 0  ({len(pos_rets):>3} 筆): wr {wr_pos:.1f}%, avg {avg_pos:+.2f}%")
        print(f"    YoY <= 0 ({len(neg_rets):>3} 筆): wr {wr_neg:.1f}%, avg {avg_neg:+.2f}%")
        print(f"    Diff: wr {wr_pos - wr_neg:+.1f}%, avg {avg_pos - avg_neg:+.2f}%")

    # 多 threshold sweep
    print(f"\n  Threshold sweep (YoY > X)：")
    for th in [-20, -10, 0, 10, 20, 30, 50]:
        keep = yoy > th
        if keep.sum() < 5: continue
        kept_rets = rets[keep]
        wr = (kept_rets > 0).mean() * 100
        avg = kept_rets.mean()
        kept_pct = keep.sum() / len(yoy) * 100
        n_yr = keep.sum() / 6.0  # 6 年資料
        print(f"    YoY > {th:+>3}%: n={keep.sum():>3} ({kept_pct:.1f}%), wr {wr:.1f}%, avg {avg:+.2f}%, 年化 {n_yr:.1f} 筆")

    # === 判定 ===
    print()
    print("=" * 70)
    print("📊 基本面 sanity 判定")
    print("=" * 70)
    print(f"\n  Spearman = {sp_corr:+.4f}")
    if abs(sp_corr) >= 0.10:
        print(f"  🟢🟢 強 alpha！值得進 backfill + CPCV 驗證")
    elif abs(sp_corr) >= 0.05:
        print(f"  🟢 中等 alpha，值得做 backfill")
    elif abs(sp_corr) >= 0.03:
        print(f"  🟡 邊際 alpha（V34 margin 也是這級，後來失敗）")
    else:
        print(f"  🔴 沒 alpha，跟 V37 跨資產同下場")
        print(f"     → 接受 89.90 final，停手")

    # 存
    out_path = os.path.join(USER_SE, "revenue_sanity_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "spearman": sp_corr, "p_value": sp_p,
            "n": len(df_yoy),
            "yoy_dist": {
                "min": float(yoy.min()), "p25": float(np.percentile(yoy,25)),
                "median": float(np.median(yoy)), "p75": float(np.percentile(yoy,75)),
                "max": float(yoy.max())
            },
        }, f, indent=2)
    df_yoy.to_csv(os.path.join(USER_SE, "revenue_yoy_results.csv"), index=False)
    print(f"\n結果存到 revenue_sanity_results.json + revenue_yoy_results.csv")


if __name__ == "__main__":
    main()
