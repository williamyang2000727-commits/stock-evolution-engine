"""
驗證盤中買 vs D+1 開盤買 vs D+1 收盤買的真實效益

問題：
  William 觀察「訊號隔天容易回檔」→ 盤中買可能反而買在高點
  我之前默認「隔夜跳空高」→ 盤中買贏

  哪個對？用 89.90 過去 133 筆實證。

做法：
  1. 跑一次 89.90 cpu_replay → 拿 trades
  2. 對每筆 trade，從 cache 撈：
     - D 收盤價（訊號日 close，現在 16:35 daily_scan 用的）
     - D+1 開盤價（隔天開盤）
     - D+1 收盤價（cpu_replay 目前 buy_price）
     - D+1 high / low（盤中區間）
  3. 對比三種情境下的「買入價 vs 後續 sell_price」

  情境 A：D 收盤買（intraday_scanner 09:30 抓到訊號當天就進）
  情境 B：D+1 開盤買（隔天 09:00 開盤掛單）
  情境 C：D+1 收盤買（cpu_replay 目前的設定）

  統計：
  - D+1 跳空高開（D+1 open > D close）的比例
  - D+1 回檔（D+1 open < D close）的比例
  - 平均/中位數差異
  - 對最終總報酬的影響
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

USER_SE = os.path.expanduser(r"C:\stock-evolution") if os.name == "nt" else os.path.expanduser("~/stock-evolution")
if not os.path.isdir(USER_SE):
    USER_SE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def main():
    print("=" * 70)
    print("驗證盤中買 vs D+1 開盤 vs D+1 收盤")
    print("=" * 70)

    # 1. 載 cache + 跑 89.90
    print("\n[1/4] 載 cache + 跑 89.90 cpu_replay...")
    raw = pickle.load(open(CACHE_PATH, "rb"))
    params = fetch_gist_strategy()
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    trades = base.cpu_replay(pre, params)
    completed = [t for t in trades if t.get("sell_date")]
    print(f"  共 {len(completed)} 筆完成交易")

    # Debug：第一筆結構 + raw cache 樣本 key
    if completed:
        print(f"  trade keys: {list(completed[0].keys())}")
        print(f"  trade[0] sample: ticker={completed[0].get('ticker')} buy_date={completed[0].get('buy_date')}")
    raw_sample_keys = list(raw.keys())[:5]
    print(f"  raw cache sample keys: {raw_sample_keys}")

    # 2. 對每筆 trade 撈 D close / D+1 open / D+1 close
    print("\n[2/4] 對每筆 trade 撈四個價格...")
    rows = []
    n_skip_ticker, n_skip_date, n_skip_idx = 0, 0, 0
    for t in completed:
        # ticker 可能是 "ticker" 或 "stock" 或 "code"
        ticker = t.get("ticker") or t.get("stock") or t.get("code") or t.get("name")
        buy_date_str = t.get("buy_date")  # D+1
        sell_price = t.get("sell_price", 0)
        if not ticker or not buy_date_str or not sell_price:
            n_skip_ticker += 1; continue
        # 嘗試多種 ticker 格式
        df = None
        for cand in [ticker, f"{ticker}.TW", f"{ticker}.TWO", str(ticker).replace(".TW", "").replace(".TWO", "")]:
            if cand in raw:
                df = raw[cand]; break
        if df is None:
            n_skip_ticker += 1; continue
        # 確保 datetime index + normalize（移除時間部分，只比對日期）
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        df_dates_normalized = df.index.normalize()
        buy_date = pd.Timestamp(buy_date_str).normalize()
        matches = np.where(df_dates_normalized == buy_date)[0]
        if len(matches) == 0:
            if n_skip_date == 0:
                print(f"  DEBUG: ticker={ticker} buy_date={buy_date_str}")
                print(f"    df.index dtype={df.index.dtype}")
                print(f"    df.index range: {df.index.min()} ~ {df.index.max()}")
                print(f"    df.index sample (last 3): {list(df.index[-3:])}")
            n_skip_date += 1; continue
        idx_buy = int(matches[0])
        if idx_buy <= 0:
            n_skip_idx += 1; continue
        # D = idx_buy - 1（訊號日）/ D+1 = idx_buy（買入日）
        d_close = float(df["Close"].iloc[idx_buy - 1])
        d1_open = float(df["Open"].iloc[idx_buy])
        d1_high = float(df["High"].iloc[idx_buy])
        d1_low = float(df["Low"].iloc[idx_buy])
        d1_close = float(df["Close"].iloc[idx_buy])
        if d_close <= 0 or d1_open <= 0:
            continue
        rows.append({
            "ticker": ticker,
            "signal_date": str(df.index[idx_buy - 1].date()),
            "buy_date": buy_date_str,
            "sell_price": sell_price,
            "d_close": d_close,
            "d1_open": d1_open,
            "d1_high": d1_high,
            "d1_low": d1_low,
            "d1_close": d1_close,
            # 跳空 % = (D+1 open - D close) / D close
            "gap_pct": (d1_open - d_close) / d_close * 100,
            # D+1 盤中均價估算（high + low + close + open）/ 4
            "d1_intraday_avg": (d1_open + d1_high + d1_low + d1_close) / 4,
            # 三情境報酬（扣 0.585% 摩擦）
            "ret_A_d_close": (sell_price / d_close - 1) * 100 - 0.585,
            "ret_B_d1_open": (sell_price / d1_open - 1) * 100 - 0.585,
            "ret_C_d1_close": (sell_price / d1_close - 1) * 100 - 0.585,
            "ret_D_d1_avg": (sell_price / ((d1_open + d1_high + d1_low + d1_close) / 4) - 1) * 100 - 0.585,
        })
    df_r = pd.DataFrame(rows)
    print(f"  成功對齊 {len(df_r)} 筆 (skip ticker={n_skip_ticker} skip_date={n_skip_date} skip_idx={n_skip_idx})")
    if len(df_r) == 0:
        print("\n❌ 0 筆對齊，無法統計。check trade keys vs raw cache keys 不匹配")
        return

    # 3. 統計
    print("\n[3/4] 統計 gap 分布 + 三情境報酬")
    print()
    print(f"{'─' * 70}")
    print("【跳空分布】D+1 開盤 vs D 收盤")
    print(f"{'─' * 70}")
    gap_high = (df_r["gap_pct"] > 0).sum()
    gap_low = (df_r["gap_pct"] < 0).sum()
    gap_flat = (df_r["gap_pct"] == 0).sum()
    n = len(df_r)
    print(f"  跳空高開 (D+1 open > D close)：{gap_high} 筆 ({gap_high/n*100:.1f}%)")
    print(f"  跳空低開 (D+1 open < D close)：{gap_low} 筆 ({gap_low/n*100:.1f}%)")
    print(f"  平盤      (D+1 open = D close)：{gap_flat} 筆 ({gap_flat/n*100:.1f}%)")
    print(f"  跳空平均：{df_r['gap_pct'].mean():+.3f}%")
    print(f"  跳空中位數：{df_r['gap_pct'].median():+.3f}%")
    print(f"  跳空 std：{df_r['gap_pct'].std():.3f}%")

    print()
    print(f"{'─' * 70}")
    print("【四種進場情境的單筆報酬（扣 0.585% 摩擦）】")
    print(f"{'─' * 70}")
    for label, col in [
        ("A. D 收盤買 (intraday_scanner)", "ret_A_d_close"),
        ("B. D+1 開盤買 (隔天 09:00 掛)", "ret_B_d1_open"),
        ("C. D+1 收盤買 (cpu_replay 目前)", "ret_C_d1_close"),
        ("D. D+1 盤中均價買 (高低開收均值)", "ret_D_d1_avg"),
    ]:
        s = df_r[col]
        wr = (s > 0).mean() * 100
        print(f"  {label}")
        print(f"    平均報酬：{s.mean():+.2f}% / 中位數：{s.median():+.2f}% / 勝率：{wr:.1f}% / 總和：{s.sum():+.1f}%")

    print()
    print(f"{'─' * 70}")
    print("【關鍵對比】")
    print(f"{'─' * 70}")
    diff_AvsB = df_r["ret_A_d_close"] - df_r["ret_B_d1_open"]
    diff_AvsC = df_r["ret_A_d_close"] - df_r["ret_C_d1_close"]
    print(f"  A (D 收盤) vs B (D+1 開盤)：差 {diff_AvsB.mean():+.3f}% / 筆 (平均)")
    print(f"  A (D 收盤) vs C (D+1 收盤)：差 {diff_AvsC.mean():+.3f}% / 筆 (平均)")
    n_A_better = (diff_AvsB > 0).sum()
    n_B_better = (diff_AvsB < 0).sum()
    print(f"    A 贏 B 筆數：{n_A_better}/{n} ({n_A_better/n*100:.1f}%)")
    print(f"    B 贏 A 筆數：{n_B_better}/{n} ({n_B_better/n*100:.1f}%)")

    # 4. 結論
    print()
    print(f"{'─' * 70}")
    print("【結論】")
    print(f"{'─' * 70}")
    avg_diff = diff_AvsB.mean()
    if avg_diff > 0.3:
        print(f"  ✅ A (D 收盤盤中買) 平均贏 B {avg_diff:+.2f}% / 筆")
        print(f"     → intraday_scanner 有意義，預期一年 22 筆 × {avg_diff:.2f}% = +{22 * avg_diff:.1f}% 總報酬")
        print(f"     → 建議做 (b)")
    elif avg_diff < -0.3:
        print(f"  ❌ B (D+1 開盤買) 平均贏 A {-avg_diff:+.2f}% / 筆")
        print(f"     → William 直覺對！訊號隔天回檔居多，盤中買反而吃虧")
        print(f"     → 不該做 (b) intraday_scanner，應該做「D+1 等回檔」優化")
        print(f"     → 一年 22 筆 × {-avg_diff:.2f}% = +{22 * (-avg_diff):.1f}% 總報酬潛力 (反向)")
    else:
        print(f"  🟡 兩者差距 {avg_diff:+.3f}% / 筆（< 0.3%），統計上不顯著")
        print(f"     → (b) 做了也沒大幫助，不值得花時間")
        print(f"     → 集中精力做 (c) Phase 1 lob_collector")

    # 存 csv 給 William 自己看
    out_csv = os.path.join(USER_SE, "intraday_advantage_results.csv")
    df_r.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  📁 詳細資料存到 {out_csv}")


if __name__ == "__main__":
    main()
