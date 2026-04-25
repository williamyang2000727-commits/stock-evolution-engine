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
            # === 現有 SOP ===
            "ret_B_d1_open": (sell_price / d1_open - 1) * 100 - 0.585,
            "ret_C_d1_close": (sell_price / d1_close - 1) * 100 - 0.585,
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
    print("【誠實版：D 盤中真實能拿到的價 vs 現有 SOP】")
    print(f"  ⚠️ D 收盤價 13:30 後才已知，盤中 09:30-13:25 拿不到")
    print(f"  → 用 D 開盤/高點/中點/OHLC 均價估算盤中能拿到的價")
    print(f"{'─' * 70}")
    scenarios = [
        ("A1. D 開盤買 (09:30 訊號早出)   [盤中最樂觀]", "ret_A1_d_open"),
        ("A2. D 高點買 (盤中最差成交)     [盤中悲觀上限]", "ret_A2_d_high"),
        ("A3. D 中點 (open+close)/2 買   [盤中居中]", "ret_A3_d_mid"),
        ("A4. D OHLC 均價買              [盤中保守均]", "ret_A4_d_ohlc_avg"),
        ("─" * 50, None),
        ("REF. D 收盤買 (時間旅行作弊版)  [不可實現]", "ret_REF_d_close"),
        ("B.  D+1 開盤買 (現有 SOP)      [對照基準]", "ret_B_d1_open"),
        ("C.  D+1 收盤買 (cpu_replay 計算用)", "ret_C_d1_close"),
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
    print("【關鍵對比 vs B (現有 SOP D+1 開盤)】")
    print(f"{'─' * 70}")
    for label, col in [
        ("A1 開盤 vs B", "ret_A1_d_open"),
        ("A2 高點 vs B", "ret_A2_d_high"),
        ("A3 中點 vs B", "ret_A3_d_mid"),
        ("A4 OHLC均 vs B", "ret_A4_d_ohlc_avg"),
        ("REF 收盤 vs B (作弊參考)", "ret_REF_d_close"),
    ]:
        diff = df_r[col] - df_r["ret_B_d1_open"]
        n_win = (diff > 0).sum()
        flag = "✅" if diff.mean() > 0.3 else ("❌" if diff.mean() < -0.3 else "🟡")
        print(f"  {flag} {label}: 平均差 {diff.mean():+.3f}% / 筆 / A 贏 B {n_win}/{n} ({n_win/n*100:.1f}%)")

    # 4. 結論
    print()
    print(f"{'─' * 70}")
    print("【誠實結論】")
    print(f"{'─' * 70}")
    a1 = (df_r["ret_A1_d_open"] - df_r["ret_B_d1_open"]).mean()
    a2 = (df_r["ret_A2_d_high"] - df_r["ret_B_d1_open"]).mean()
    a3 = (df_r["ret_A3_d_mid"] - df_r["ret_B_d1_open"]).mean()
    a4 = (df_r["ret_A4_d_ohlc_avg"] - df_r["ret_B_d1_open"]).mean()
    print(f"  最樂觀 (D 開盤)：{a1:+.2f}% / 筆")
    print(f"  悲觀上限 (D 高點)：{a2:+.2f}% / 筆")
    print(f"  中點：{a3:+.2f}% / 筆")
    print(f"  OHLC 均：{a4:+.2f}% / 筆")
    print()
    if a2 > 0.3:
        print(f"  ✅ 連 D 高點（盤中最差成交）都贏 B → (b) intraday_scanner 真的有用")
        print(f"     一年 22 筆 × {a2:.2f}% (悲觀) ~ {a1:.2f}% (樂觀) = +{22*a2:.0f}% ~ +{22*a1:.0f}% 總報酬")
    elif a2 < -0.3 and a1 > 0.3:
        print(f"  🟡 早盤訊號(A1)贏，但晚盤訊號(A2)輸")
        print(f"     建議 intraday_scanner 只信 09:30-10:30 訊號，下午訊號跳過")
    elif a1 < -0.3:
        print(f"  ❌ 連最樂觀(D 開盤)都輸 B → 盤中買真的不划算")
        print(f"     William 直覺對，不該做 (b)，跳到 (c)")
    else:
        print(f"  🟡 樂觀贏悲觀輸，邊際效益不明，看時段而定")
        print(f"     建議：盤中 09:30 訊號才推 Telegram，避開 13:00 後訊號")

    # 存 csv 給 William 自己看
    out_csv = os.path.join(USER_SE, "intraday_advantage_results.csv")
    df_r.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  📁 詳細資料存到 {out_csv}")


if __name__ == "__main__":
    main()
