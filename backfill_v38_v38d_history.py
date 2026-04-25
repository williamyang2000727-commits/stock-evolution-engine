"""
歷史回填 Track B/C — 立刻看結果（不用等 2 個月 paper trading）

邏輯：
  1. 跑 89.90 cpu_replay 拿全期 133 筆完整 trades（含 actual return）
  2. 對每筆抽 19 features + Kronos prediction（已有 csv）
  3. 套 V38 rule + V38d ML head → 算每筆「假如那天有 V38d，會不會買」
  4. 統計三軌 wr / total / avg

注意：
  - 這是 in-sample retrospective（V38d 用全期訓練，看全期回測）
  - 跟真實 forward test 不同，會偏樂觀
  - **CPCV LOO 才是真實表現**：mean wr↑ +18% (V38d) / +14% (V38 zero-shot)
  - 這個 backfill 只是「立刻看數字」用
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base
from metalabel_features import extract_features_for_trades, FEATURE_NAMES

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
SANITY_CSV = os.path.join(USER_SE, "kronos_sanity_results.csv")
KRONOS_NEXT_TH = 0.8


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s), s.get("score", "N/A")


def stats_of(rets):
    if len(rets) == 0:
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0, "max_dd": 0.0}
    r = np.array(rets)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return {
        "n": len(r),
        "wr": float((r > 0).mean() * 100),
        "total": float(r.sum()),
        "avg": float(r.mean()),
        "max_dd": float(dd.min()) if len(dd) > 0 else 0.0,
    }


def main():
    print("=" * 70)
    print("Backfill Track B/C 歷史 — 立刻看結果")
    print("=" * 70)

    # === 1. 讀 sanity + 跑 89.90 ===
    if not os.path.exists(SANITY_CSV):
        print(f"❌ {SANITY_CSV} 不存在")
        return
    df_sanity = pd.read_csv(SANITY_CSV)
    print(f"\n[1/5] sanity CSV: {len(df_sanity)} 筆 Kronos predictions")

    print(f"\n[2/5] 跑 89.90 cpu_replay...")
    params, _ = fetch_gist_strategy()
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

    # === 3. 抽 features + 合併 Kronos ===
    print(f"\n[3/5] 抽 features + 合併 Kronos...")
    X19, _, keep_indices = extract_features_for_trades(pre, completed)
    trades_kept = [completed[i] for i in keep_indices]
    df_trades = pd.DataFrame([{
        "ticker": t.get("ticker"),
        "buy_date": t.get("buy_date"),
        "sell_date": t.get("sell_date"),
        "actual_return": float(t.get("return", 0)),
    } for t in trades_kept]).reset_index(drop=True)

    df_merged = df_trades.merge(
        df_sanity[["ticker", "buy_date", "pred_next_pct", "pred_5d_pct"]],
        on=["ticker", "buy_date"], how="inner"
    )
    trade_key = [(t.get("ticker"), t.get("buy_date")) for t in trades_kept]
    merged_keys = list(zip(df_merged["ticker"], df_merged["buy_date"]))
    aligned_idx = [trade_key.index(k) for k in merged_keys]
    X19_m = X19[aligned_idx]
    rets_m = df_merged["actual_return"].values
    p_next = df_merged["pred_next_pct"].values
    p_5d = df_merged["pred_5d_pct"].values
    print(f"  合併後 {len(df_merged)} 筆")

    # === 4. 套 V38 + V38d 規則 ===
    print(f"\n[4/5] 套規則計算 Track B/C 決策...")

    # V38: pred_next > 0.8 AND pred_5d > full-period median
    full_5d_median = float(np.median(p_5d))
    v38_pass = (p_next > KRONOS_NEXT_TH) & (p_5d > full_5d_median)
    print(f"  V38 (rule only): {v38_pass.sum()}/{len(df_merged)} 筆 pass ({v38_pass.mean()*100:.1f}%)")

    # V38d: V38 pass AND ML proba > train median
    # ML 訓練用 V38 子集
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_v38 = X19_m[v38_pass]
    y_v38 = (rets_m[v38_pass] > 0).astype(int)
    scaler = StandardScaler()
    X_v38_s = scaler.fit_transform(X_v38)
    model = LogisticRegression(C=1.0, max_iter=500, penalty="l2")
    model.fit(X_v38_s, y_v38)
    proba_train = model.predict_proba(X_v38_s)[:, 1]
    train_proba_median = float(np.median(proba_train))

    # 對所有 trades 套 V38d
    X19_s_all = scaler.transform(X19_m)
    proba_all = model.predict_proba(X19_s_all)[:, 1]
    v38d_pass = v38_pass & (proba_all > train_proba_median)
    print(f"  V38d (V38 + ML head): {v38d_pass.sum()}/{len(df_merged)} 筆 pass ({v38d_pass.mean()*100:.1f}%)")

    # === 5. 統計三軌 ===
    print(f"\n[5/5] 三軌統計...")
    a_stats = stats_of(rets_m)
    b_stats = stats_of(rets_m[v38_pass])
    c_stats = stats_of(rets_m[v38d_pass])

    print()
    print("=" * 70)
    print("📊 三軌歷史回填結果（in-sample，含 actual return）")
    print("=" * 70)
    print(f"\n【Track A】89.90 全買")
    print(f"  n = {a_stats['n']}, wr = {a_stats['wr']:.1f}%, avg = {a_stats['avg']:+.2f}%, total = {a_stats['total']:+.0f}%")
    print(f"  max_dd (cumulative) = {a_stats['max_dd']:.0f}%")

    print(f"\n【Track B】V38 zero-shot (next > 0.8 AND 5d > {full_5d_median:.2f})")
    print(f"  n = {b_stats['n']}, wr = {b_stats['wr']:.1f}%, avg = {b_stats['avg']:+.2f}%, total = {b_stats['total']:+.0f}%")
    print(f"  保留率: {b_stats['n']/a_stats['n']*100:.1f}%")
    print(f"  vs Track A: wr {b_stats['wr']-a_stats['wr']:+.1f}%, avg {b_stats['avg']-a_stats['avg']:+.2f}%")

    print(f"\n【Track C】V38d (V38 + ML proba > {train_proba_median:.3f})")
    print(f"  n = {c_stats['n']}, wr = {c_stats['wr']:.1f}%, avg = {c_stats['avg']:+.2f}%, total = {c_stats['total']:+.0f}%")
    print(f"  保留率: {c_stats['n']/a_stats['n']*100:.1f}%")
    print(f"  vs Track A: wr {c_stats['wr']-a_stats['wr']:+.1f}%, avg {c_stats['avg']-a_stats['avg']:+.2f}%")
    print(f"  vs Track B: wr {c_stats['wr']-b_stats['wr']:+.1f}%, avg {c_stats['avg']-b_stats['avg']:+.2f}%")

    # 配置：實盤該選哪個
    print(f"\n=" * 35 + " 結論 " + "=" * 35)
    print(f"\n⚠️ 注意：這是 in-sample retrospective（V38d 用全期 train + 全期 test）")
    print(f"   實際 CPCV LOO（避免 leak）的結果：")
    print(f"   - V38 zero-shot: mean wr↑ +14.28%, p25 +5.58%（穩健）")
    print(f"   - V38d:          mean wr↑ +18.02%, p25 +0.54%（mean ↑ 但 p25 ↓）")

    if b_stats["wr"] - a_stats["wr"] > 5 and c_stats["wr"] - b_stats["wr"] > 3:
        print(f"\n🟢 V38d 顯著勝 V38 in-sample → CPCV 也支持，可上線")
    elif b_stats["wr"] - a_stats["wr"] > 3:
        print(f"\n🟡 V38 過濾有效，V38d 邊際進步 → 建議用 V38 上線（V38d 加為次要追蹤）")
    else:
        print(f"\n🔴 V38 改善小，過濾代價高（kept rate {b_stats['n']/a_stats['n']*100:.0f}%）")

    # 存
    out_path = os.path.join(USER_SE, "backfill_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "track_A": a_stats, "track_B": b_stats, "track_C": c_stats,
            "v38_5d_median": full_5d_median,
            "v38d_proba_median": train_proba_median,
        }, f, indent=2)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
