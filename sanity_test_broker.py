"""
V37 分點券商主力 sanity test — 用 Spearman correlation 跨全期評估
用法：C:\\stock-evolution> python sanity_test_broker.py

學乖了：不用 80/20 split（V36 教訓）
        改用 Spearman correlation（V34 margin sanity 同方法）

判定：
  🟢 |mean Spearman| ≥ 0.05 AND 方向一致率 ≥ 60% → 跑 V37 GPU
  🟡 0.03-0.05 → 邊際值得試
  🔴 < 0.03 → 訊號太弱，跳 Kronos

5 個分點指標：
  1. broker_concentration: top-15 主力買超集中度（Herfindahl）
  2. broker_follow_rate: 高 follow-rate 分點（過去 30 天買股 avg return 高）今日有沒買
  3. broker_persistence: 連續 N 天主力買超的分點數
  4. broker_top_buy_ratio: top-3 主力今日買量佔當日總買量
  5. broker_net_concentration: 加權買賣超集中度
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
from scipy import stats

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
BROKER_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "broker_data_full.pkl")

HORIZONS = [5, 10, 30]
MIN_VALID = 100
ALPHA_THRESHOLD = 0.05
EDGE_THRESHOLD = 0.03


def forward_returns(close: np.ndarray, h: int) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float32)
    if len(close) > h:
        out[:-h] = (close[h:] / np.where(close[:-h] > 0, close[:-h], np.nan) - 1) * 100
    return out


def compute_indicators_per_stock(df: pd.DataFrame):
    """
    對單一股票的 broker DataFrame，算每天的 5 個指標
    df 預期欄位：date, BrokerID, BrokerName, buy(買進股數), sell(賣出股數), price...
    或 FinMind 真實欄位 — 第一次跑要看實際 schema

    Returns: DataFrame index=date, columns=5 個指標
    """
    if df.empty:
        return pd.DataFrame()

    # 確認 schema
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # FinMind 欄位猜測（實跑後可能要調）
    # 可能是：buy / sell / 買進 / 賣出 / Buy / Sell
    buy_col = None
    sell_col = None
    for col in df.columns:
        cl = col.lower()
        if "buy" in cl and "sell" not in cl and buy_col is None:
            buy_col = col
        elif "sell" in cl and sell_col is None:
            sell_col = col
    # 中文後備
    if buy_col is None and "買進" in df.columns:
        buy_col = "買進"
    if sell_col is None and "賣出" in df.columns:
        sell_col = "賣出"
    if buy_col is None or sell_col is None:
        # 實跑後再 fix schema
        return pd.DataFrame()

    df["net"] = df[buy_col] - df[sell_col]

    daily_indicators = []
    for date, group in df.groupby("date"):
        if len(group) == 0:
            continue
        # 過濾零交易分點
        active = group[(group[buy_col] > 0) | (group[sell_col] > 0)]
        if len(active) < 3:
            continue

        # 1. broker_concentration: 買超 top-15 的 Herfindahl
        net_pos = active[active["net"] > 0].nlargest(15, "net")
        total_net_buy = net_pos["net"].sum() if len(net_pos) > 0 else 0
        if total_net_buy > 0:
            shares = net_pos["net"] / total_net_buy
            concentration = (shares ** 2).sum()  # Herfindahl 0-1
        else:
            concentration = 0

        # 2. broker_top_buy_ratio: top-3 買量 / 總買量
        top3_buy = active.nlargest(3, buy_col)[buy_col].sum()
        total_buy = active[buy_col].sum()
        top3_ratio = top3_buy / total_buy if total_buy > 0 else 0

        # 3. broker_buy_count: 多少分點買超 ≥ 100 張（10 萬股）
        big_buyers = (active["net"] >= 100000).sum()

        # 4. broker_net_imbalance: top-15 買超 - top-15 賣超 / 總成交
        top15_buy = active.nlargest(15, "net")["net"].sum()  # 含負數會被排到後面
        top15_sell = active.nsmallest(15, "net")["net"].sum()  # 最負的（賣超最大）
        total_volume = active[buy_col].sum() + active[sell_col].sum()
        net_imbalance = (top15_buy + top15_sell) / total_volume if total_volume > 0 else 0

        # 5. broker_n_active: 當天交易分點數（活躍度）
        n_active = len(active)

        daily_indicators.append({
            "date": pd.Timestamp(date),
            "concentration": concentration,
            "top3_ratio": top3_ratio,
            "big_buyers": big_buyers,
            "net_imbalance": net_imbalance,
            "n_active": n_active,
        })

    return pd.DataFrame(daily_indicators).set_index("date") if daily_indicators else pd.DataFrame()


def main():
    print("=" * 60)
    print("V37 分點券商主力 sanity test (Spearman 全期)")
    print("=" * 60)

    if not os.path.exists(BROKER_PATH):
        print(f"❌ 找不到 {BROKER_PATH}")
        print(f"   先跑：python fetch_broker_history.py")
        return

    print(f"\n[1/3] 載入 broker data...")
    broker_data = pickle.load(open(BROKER_PATH, "rb"))
    print(f"  共 {len(broker_data)} 檔股票")
    if broker_data:
        first_key = list(broker_data.keys())[0]
        first_df = broker_data[first_key]
        print(f"  範例 {first_key}: {len(first_df)} rows, columns = {list(first_df.columns)}")
        if len(first_df) > 0:
            print(f"    sample row: {first_df.iloc[0].to_dict()}")

    print(f"\n[2/3] 載入 OHLCV cache...")
    ohlcv = pickle.load(open(CACHE_PATH, "rb"))
    print(f"  共 {len(ohlcv)} 檔")

    # === 計算指標 ===
    print(f"\n[3/3] 對每檔股票算指標 + Spearman 相關...")
    print(f"      Horizons: {HORIZONS} 天 forward return")
    print(f"      Threshold: |mean Spearman| ≥ {ALPHA_THRESHOLD} → GREEN")
    print()

    indicator_names = ["concentration", "top3_ratio", "big_buyers", "net_imbalance", "n_active"]
    per_stock_corr = {(n, h): [] for n in indicator_names for h in HORIZONS}
    n_processed = 0
    n_skipped = 0

    for ticker_short in broker_data:
        ticker_yf = f"{ticker_short}.TW"
        ticker_yfo = f"{ticker_short}.TWO"
        if ticker_yf in ohlcv:
            ohlcv_df = ohlcv[ticker_yf]
        elif ticker_yfo in ohlcv:
            ohlcv_df = ohlcv[ticker_yfo]
        else:
            n_skipped += 1
            continue

        broker_df = broker_data[ticker_short]
        ind_df = compute_indicators_per_stock(broker_df)
        if ind_df.empty:
            n_skipped += 1
            continue

        # 對齊 broker indicators 跟 OHLCV
        ohlcv_df_2 = ohlcv_df.copy()
        if hasattr(ohlcv_df_2.index[0], 'tz') and ohlcv_df_2.index[0].tz is not None:
            ohlcv_df_2.index = ohlcv_df_2.index.tz_localize(None)
        common = ind_df.index.intersection(ohlcv_df_2.index)
        if len(common) < MIN_VALID:
            n_skipped += 1
            continue

        ind_aligned = ind_df.loc[common]
        close_aligned = ohlcv_df_2.loc[common, "Close"].values.astype(np.float32)

        # 對每個 horizon + indicator 算 Spearman
        for h in HORIZONS:
            fr = forward_returns(close_aligned, h)
            mask = ~np.isnan(fr)
            if mask.sum() < MIN_VALID:
                continue
            for ind_name in indicator_names:
                x = ind_aligned[ind_name].values
                m2 = mask & ~np.isnan(x)
                if m2.sum() < MIN_VALID:
                    continue
                if x[m2].std() < 1e-6:
                    continue
                rho, _ = stats.spearmanr(x[m2], fr[m2])
                if not np.isnan(rho):
                    per_stock_corr[(ind_name, h)].append(rho)

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  處理 {n_processed} / {len(broker_data)}")

    print(f"\n處理完：{n_processed} 檔有效，{n_skipped} 檔跳過")

    # === 結果 ===
    print(f"\n=== Spearman 相關係數平均 ===")
    print(f"{'指標':<18s}", end="")
    for h in HORIZONS:
        print(f"{'h='+str(h):>10s}", end="")
    print()
    print("-" * 50)

    any_alpha = False
    max_abs_mean = 0.0
    best_ind, best_h = None, None

    for ind_name in indicator_names:
        line = f"{ind_name:<18s}"
        for h in HORIZONS:
            arr = np.array(per_stock_corr[(ind_name, h)])
            if len(arr) == 0:
                line += f"{'N/A':>10s}"
                continue
            mean = arr.mean()
            line += f"{mean:>+10.4f}"
            if abs(mean) >= ALPHA_THRESHOLD:
                any_alpha = True
            if abs(mean) > max_abs_mean:
                max_abs_mean = abs(mean)
                best_ind = ind_name
                best_h = h
        print(line)

    print()
    print(f"=== 方向一致率（多少 % 股票 |corr| > 0.05 且方向相同）===")
    for ind_name in indicator_names:
        for h in HORIZONS:
            arr = np.array(per_stock_corr[(ind_name, h)])
            if len(arr) == 0:
                continue
            pos_pct = (arr > 0.05).mean() * 100
            neg_pct = (arr < -0.05).mean() * 100
            print(f"  {ind_name:<18s} h={h:>3d}: >+0.05 {pos_pct:5.1f}% / <-0.05 {neg_pct:5.1f}% / std={arr.std():.3f}")

    # === Go/no-go ===
    print(f"\n=== V37 Go/no-go 裁決 ===")
    if any_alpha:
        print(f"🟢 GREEN — 有 alpha 訊號")
        print(f"  最強：{best_ind} h={best_h} mean={max_abs_mean:.4f} (>= {ALPHA_THRESHOLD})")
        print(f"  → 值得寫 V37 GPU 跑 24h")
    elif max_abs_mean < EDGE_THRESHOLD:
        print(f"🔴 RED — 無 alpha")
        print(f"  最強只 {max_abs_mean:.4f} < {EDGE_THRESHOLD}")
        print(f"  → 跳 V37，做 Kronos 或 forward test")
    else:
        print(f"🟡 YELLOW — 邊際")
        print(f"  最強 {max_abs_mean:.4f} 在 0.03-0.05 之間")
        print(f"  → 可選擇做或不做")


if __name__ == "__main__":
    main()
