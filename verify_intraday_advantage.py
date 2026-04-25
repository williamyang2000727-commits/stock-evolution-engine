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

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
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
        # 確保 datetime index + 處理 timezone（raw cache 帶 Asia/Taipei tz）
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        # 把 raw 的 tz 拔掉再 normalize，buy_date 是 tz-naive
        if df.index.tz is not None:
            df_dates = df.index.tz_localize(None).normalize()
        else:
            df_dates = df.index.normalize()
        buy_date = pd.Timestamp(buy_date_str).normalize()
        matches = np.where(df_dates == buy_date)[0]
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
        # D 日 OHLC（訊號日，盤中 09:30-13:25 能掃描到訊號的那天）
        d_open = float(df["Open"].iloc[idx_buy - 1])
        d_high = float(df["High"].iloc[idx_buy - 1])
        d_low = float(df["Low"].iloc[idx_buy - 1])
        if d_open <= 0 or d_high <= 0:
            continue
        # 盤中均價估算 = (open + high + low + close) / 4
        d_intraday_avg = (d_open + d_high + d_low + d_close) / 4
        # 中點估算 = (open + close) / 2
        d_midpoint = (d_open + d_close) / 2
        rows.append({
            "ticker": ticker,
            "signal_date": str(df.index[idx_buy - 1].date()),
            "buy_date": buy_date_str,
            "sell_price": sell_price,
            "d_open": d_open, "d_high": d_high, "d_low": d_low, "d_close": d_close,
            "d1_open": d1_open, "d1_high": d1_high, "d1_low": d1_low, "d1_close": d1_close,
            "gap_pct": (d1_open - d_close) / d_close * 100,
            # === 誠實版：D 盤中能拿到的代理價（4 種）===
            # A1: D 開盤買（09:30 訊號早出，第一筆掛單）— 最樂觀
            "ret_A1_d_open": (sell_price / d_open - 1) * 100 - 0.585,
            # A2: D 高點買（09:30-13:25 最差成交，悲觀上限）
            "ret_A2_d_high": (sell_price / d_high - 1) * 100 - 0.585,
            # A3: D 中點 (open+close)/2 買 — 居中估
            "ret_A3_d_mid": (sell_price / d_midpoint - 1) * 100 - 0.585,
            # A4: D OHLC 均價買 — 保守居中
            "ret_A4_d_ohlc_avg": (sell_price / d_intraday_avg - 1) * 100 - 0.585,
            # === 對照組（時間旅行版，當 reference）===
            "ret_REF_d_close": (sell_price / d_close - 1) * 100 - 0.585,
            # === D+1 各時點（你 SOP 是 D+1 13:25 前買）===
            "ret_B_d1_open": (sell_price / d1_open - 1) * 100 - 0.585,        # D+1 09:00 開盤掛
            "ret_B2_d1_high": (sell_price / d1_high - 1) * 100 - 0.585,        # D+1 盤中最高（最壞）
            "ret_B3_d1_mid": (sell_price / ((d1_open + d1_close) / 2) - 1) * 100 - 0.585,  # D+1 11:00 中點
            "ret_B4_d1_ohlc_avg": (sell_price / ((d1_open + d1_high + d1_low + d1_close) / 4) - 1) * 100 - 0.585,  # D+1 OHLC 均
            "ret_C_d1_close": (sell_price / d1_close - 1) * 100 - 0.585,       # D+1 13:25 收盤前買 = 現在 SOP
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
    print("【正確版：D+1 不同時點進場（回測邏輯：D 訊號 → D+1 買入）】")
    print(f"  你 SOP：D+1 13:25 前買 ≈ C (D+1 收盤價)")
    print(f"  intraday_scanner 提案：D+1 早盤就買 (B 或 B3)")
    print(f"{'─' * 70}")
    scenarios = [
        ("B.  D+1 開盤 09:00 掛 [現在 SOP 早盤版]", "ret_B_d1_open"),
        ("B3. D+1 中點 11:00 進 [中盤]", "ret_B3_d1_mid"),
        ("B4. D+1 OHLC 均價 [盤中均勻分布]", "ret_B4_d1_ohlc_avg"),
        ("B2. D+1 高點 [盤中最差成交]", "ret_B2_d1_high"),
        ("C.  D+1 收盤 13:25 前買 [你現在 SOP] ⭐", "ret_C_d1_close"),
        ("─" * 50, None),
        ("(參考) D 訊號日當天買 — 需即時計算指標 (intraday_scanner 想做的)", None),
        ("A1. D 開盤 09:30 [需 D-1 訊號 + D 開盤]", "ret_A1_d_open"),
        ("A4. D OHLC 均價 [盤中均]", "ret_A4_d_ohlc_avg"),
        ("REF. D 收盤 [作弊參考，不可實現]", "ret_REF_d_close"),
    ]
    for label, col in scenarios:
        if col is None:
            print(f"  {label}")
            continue
        s = df_r[col]
        wr = (s > 0).mean() * 100
        print(f"  {label}")
        print(f"      平均：{s.mean():+.2f}% / 中位數：{s.median():+.2f}% / 勝率：{wr:.1f}% / 總和：{s.sum():+.1f}%")

    print()
    print(f"{'─' * 70}")
    print("【關鍵對比 vs C (你現在 SOP = D+1 13:25 前買)】")
    print(f"{'─' * 70}")
    for label, col in [
        ("B (D+1 開盤 09:00) vs C (你 SOP)", "ret_B_d1_open"),
        ("B3 (D+1 中點 11:00) vs C", "ret_B3_d1_mid"),
        ("B4 (D+1 OHLC均) vs C", "ret_B4_d1_ohlc_avg"),
        ("B2 (D+1 高點 最差) vs C", "ret_B2_d1_high"),
    ]:
        diff = df_r[col] - df_r["ret_C_d1_close"]
        n_win = (diff > 0).sum()
        flag = "✅" if diff.mean() > 0.3 else ("❌" if diff.mean() < -0.3 else "🟡")
        print(f"  {flag} {label}: 平均差 {diff.mean():+.3f}% / 筆 / 早買贏 {n_win}/{n} ({n_win/n*100:.1f}%)")

    # 4. 真結論
    print()
    print(f"{'─' * 70}")
    print("【真結論：D+1 早盤買 vs 你 SOP (D+1 收盤前買)】")
    print(f"{'─' * 70}")
    bvc = (df_r["ret_B_d1_open"] - df_r["ret_C_d1_close"]).mean()
    b3vc = (df_r["ret_B3_d1_mid"] - df_r["ret_C_d1_close"]).mean()
    b4vc = (df_r["ret_B4_d1_ohlc_avg"] - df_r["ret_C_d1_close"]).mean()
    b2vc = (df_r["ret_B2_d1_high"] - df_r["ret_C_d1_close"]).mean()
    print(f"  D+1 09:00 開盤買 vs 你 SOP：{bvc:+.2f}% / 筆")
    print(f"  D+1 11:00 中點買 vs 你 SOP：{b3vc:+.2f}% / 筆")
    print(f"  D+1 OHLC 均買 vs 你 SOP：{b4vc:+.2f}% / 筆")
    print(f"  D+1 高點買 vs 你 SOP：{b2vc:+.2f}% / 筆")
    print()
    if bvc > 0.3:
        print(f"  ✅ D+1 早盤買贏你 SOP {bvc:+.2f}%/筆 → 改 SOP 提早買")
    elif bvc < -0.3:
        print(f"  ❌ D+1 早盤買輸你 SOP {-bvc:.2f}%/筆 → 你直覺對，訊號隔天回檔，等收盤前買最好")
        print(f"     不該做 (b) intraday_scanner，現在 SOP 已最優")
        print(f"     一年 22 筆 × {-bvc:.2f}% = +{22*(-bvc):.0f}% 反向損失（如果改成早盤買的話）")
    else:
        print(f"  🟡 D+1 開盤 vs 收盤差距 {bvc:+.3f}%（< 0.3% 不顯著）")
        print(f"     早晚買差不多，(b) 沒意義，跳到 (c)")

    # 存 csv 給 William 自己看
    out_csv = os.path.join(USER_SE, "intraday_advantage_results.csv")
    df_r.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  📁 詳細資料存到 {out_csv}")


if __name__ == "__main__":
    main()
