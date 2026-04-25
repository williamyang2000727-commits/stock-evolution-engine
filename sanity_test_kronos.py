"""
V38 Kronos sanity test — 對 89.90 全期 trades 評估 Kronos 預測能力
用法：C:\\stock-evolution> python sanity_test_kronos.py

學乖了：不用 80/20 split（V36 教訓）
        用 Spearman 全期評估（V34 margin / V37 cross-asset 同方法）

設計：
  對 89.90 全期 133 trades 中的每筆：
    1. 取 buy_date 前 60 K 線
    2. Kronos 預測未來 5-10 天（含 sell_date）
    3. 拿預測的「次日 close」算 P_predicted_return = (pred_close[0] / today_close - 1) * 100
    4. Spearman: P_predicted_return vs trade actual return

  判定：
    🟢 |Spearman| ≥ 0.10 → 過 CPCV LOO 驗證（嚴於 V37 因 Kronos 是 SOTA）
    🟡 0.05-0.10 → 邊際
    🔴 < 0.05 → 跳 V39 forward test
"""
import os, sys, pickle, time, json
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
KRONOS_DIR = os.path.join(USER_SE, "Kronos")
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
PRED_LEN = 5  # 預測未來 5 天
LOOKBACK = 60  # 用前 60 天當 input


def fetch_gist_strategy():
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s), s.get("score", "N/A")


def main():
    print("=" * 70)
    print("V38 Kronos Sanity Test — 對 89.90 全期 trades 跑預測")
    print("=" * 70)

    # === 1. 載 Kronos ===
    print(f"\n[1/4] 載入 Kronos model...")
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        print(f"❌ Kronos model module 找不到。先跑 setup_kronos.py")
        return

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print(f"  ✅ Kronos-small + Tokenizer-base loaded")

    # === 2. 載 89.90 + cpu_replay ===
    print(f"\n[2/4] 載入 89.90 + cpu_replay...")
    params, score = fetch_gist_strategy()
    print(f"  89.90 score = {score}")

    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data)

    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    print(f"  完成交易 {len(completed)} 筆")

    # === 3. 對每筆 trade 跑 Kronos ===
    print(f"\n[3/4] 對 133 trades 逐一跑 Kronos 預測...")
    print(f"  Input: 前 {LOOKBACK} 天 K 線 / Output: 未來 {PRED_LEN} 天")
    print(f"  CPU 跑預估 ~5-10 秒/筆 × 133 = 10-25 分鐘")
    print(f"  GPU 跑預估 ~0.3-1 秒/筆 × 133 = 1-2 分鐘")

    rows = []
    failed = 0
    t_start = time.time()

    for i, trade in enumerate(completed):
        ticker = trade.get("ticker", "")
        bd_str = trade.get("buy_date", "")
        if ticker not in raw:
            failed += 1
            continue
        df_full = raw[ticker]

        # 找 buy_date index
        bd = pd.to_datetime(bd_str)
        # tz-naive 處理
        if df_full.index.tz is not None:
            df_full = df_full.copy()
            df_full.index = df_full.index.tz_localize(None)

        # 找 buy_date 之前最近 60 天（不含 buy_date 當天本身，因為 89.90 是 D 收盤訊號 D+1 開盤買）
        # 我們要的是 D 收盤後的預測，所以 input = up to D（含 D），y_timestamp 從 D+1 開始
        if bd not in df_full.index:
            # 如果剛好不是交易日，找之前最近的
            mask = df_full.index <= bd
            if mask.sum() < LOOKBACK:
                failed += 1
                continue
            d_idx = mask.sum() - 1
        else:
            d_idx = df_full.index.get_loc(bd)
            if d_idx < LOOKBACK:
                failed += 1
                continue

        # x_df = bd_idx - LOOKBACK + 1 ~ bd_idx (含 bd 共 LOOKBACK 天)
        x_df = df_full.iloc[d_idx - LOOKBACK + 1:d_idx + 1].copy()
        x_df.columns = [c.lower() for c in x_df.columns]
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in x_df.columns for c in required):
            failed += 1
            continue
        x_df = x_df[required].reset_index(drop=True)
        x_timestamp = df_full.index[d_idx - LOOKBACK + 1:d_idx + 1].to_series().reset_index(drop=True)

        # y_timestamp = bd_idx+1 ~ +PRED_LEN
        if d_idx + 1 + PRED_LEN > len(df_full):
            # 已經到最近期，沒未來資料
            failed += 1
            continue
        y_timestamp = df_full.index[d_idx + 1:d_idx + 1 + PRED_LEN].to_series().reset_index(drop=True)

        try:
            pred = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=PRED_LEN,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            # P_pred = (predicted next-day close / today close - 1) * 100
            today_close = float(x_df["close"].iloc[-1])
            pred_next_close = float(pred["close"].iloc[0])
            pred_5d_close = float(pred["close"].iloc[-1])
            p_next_pct = (pred_next_close / today_close - 1) * 100
            p_5d_pct = (pred_5d_close / today_close - 1) * 100
        except Exception as e:
            if i < 3:
                print(f"  [{i}] {ticker} predict 失敗: {e}")
            failed += 1
            continue

        ret = float(trade.get("return", 0))
        rows.append({
            "ticker": ticker,
            "buy_date": bd_str,
            "today_close": today_close,
            "pred_next_pct": p_next_pct,
            "pred_5d_pct": p_5d_pct,
            "actual_return": ret,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(completed) - i - 1) / rate
            print(f"  [{i+1}/{len(completed)}] {rate:.2f}筆/秒，ETA {eta/60:.1f}分鐘 (failed {failed})")

    df = pd.DataFrame(rows)
    print(f"\n  完成 {len(df)} 筆，failed {failed} 筆")

    if len(df) < 30:
        print(f"❌ 有效樣本太少 ({len(df)} < 30)，無法評估")
        return

    # === 4. Spearman + 判定 ===
    print(f"\n[4/4] 評估 Kronos 預測能力...")
    print()

    for sig_col, label in [("pred_next_pct", "次日預測 %"), ("pred_5d_pct", "5日預測 %")]:
        rho, pval = stats.spearmanr(df[sig_col], df["actual_return"])
        print(f"=== {label} vs actual return ===")
        print(f"  Spearman = {rho:+.4f}, p-value = {pval:.4f}")
        if abs(rho) >= 0.10:
            print(f"  🟢 GREEN（|ρ| ≥ 0.10）")
        elif abs(rho) >= 0.05:
            print(f"  🟡 YELLOW（|ρ| in [0.05, 0.10]）")
        else:
            print(f"  🔴 RED（|ρ| < 0.05）")

        # Conditional：高 vs 低
        thresh = df[sig_col].median()
        high = df[df[sig_col] > thresh]
        low = df[df[sig_col] <= thresh]
        if len(high) >= 10 and len(low) >= 10:
            wr_high = (high["actual_return"] > 0).mean() * 100
            wr_low = (low["actual_return"] > 0).mean() * 100
            print(f"  Conditional: P > median wr={wr_high:.1f}% (n={len(high)})  vs  ≤ median wr={wr_low:.1f}% (n={len(low)})")
            print(f"               wr diff = {wr_high - wr_low:+.1f}%")
        print()

    # 判定
    rho_next, _ = stats.spearmanr(df["pred_next_pct"], df["actual_return"])
    rho_5d, _ = stats.spearmanr(df["pred_5d_pct"], df["actual_return"])
    best_rho = max(abs(rho_next), abs(rho_5d))

    print("=" * 70)
    print("📋 V38 Kronos Sanity Go/no-go 裁決")
    print("=" * 70)
    if best_rho >= 0.10:
        print(f"🟢 GREEN — 進階做 CPCV LOO 驗證")
        print(f"  最強 |Spearman| = {best_rho:.4f} ≥ 0.10")
        print(f"  → 寫 cpcv_test_kronos.py 跑 15 path leave-one-out")
    elif best_rho >= 0.05:
        print(f"🟡 YELLOW — 邊際")
        print(f"  最強 |Spearman| = {best_rho:.4f} in [0.05, 0.10]")
        print(f"  → 可選擇做 CPCV 或跳 V39")
    else:
        print(f"🔴 RED — Kronos zero-shot 在台股無 alpha")
        print(f"  最強 |Spearman| = {best_rho:.4f} < 0.05")
        print(f"  → 跳 V38，做 V39 forward test（最後手段）")

    # 存結果
    out_path = os.path.join(USER_SE, "kronos_sanity_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
