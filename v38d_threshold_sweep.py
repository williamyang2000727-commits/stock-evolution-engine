"""
V38d threshold sweep — 找平衡頻率/勝率的最佳閾值
用法：C:\\stock-evolution> python v38d_threshold_sweep.py

問題：V38d 預設 threshold = train proba median = 0.850 太嚴格
    → 一年只買 1.7 筆，實盤不可行

解法：sweep ML threshold 從 0.30 到 0.90，看每個 threshold 的：
    - n_kept（年化推算）
    - wr（in-sample 偏樂觀）
    - avg / total
    - 跟 89.90、V38 比較

讓使用者自己挑「能接受的頻率 + 還算好的 wr」
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base
from metalabel_features import extract_features_for_trades

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
SANITY_CSV = os.path.join(USER_SE, "kronos_sanity_results.csv")
KRONOS_NEXT_TH = 0.8


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def stats_of(rets):
    if len(rets) == 0:
        return {"n": 0, "wr": 0.0, "total": 0.0, "avg": 0.0}
    r = np.array(rets)
    return {
        "n": len(r),
        "wr": float((r > 0).mean() * 100),
        "total": float(r.sum()),
        "avg": float(r.mean()),
    }


def main():
    print("=" * 70)
    print("V38d Threshold Sweep — 找平衡的最佳閾值")
    print("=" * 70)

    df_sanity = pd.read_csv(SANITY_CSV)
    print(f"\n[1/3] Sanity CSV: {len(df_sanity)} 筆")

    print(f"\n[2/3] 跑 89.90 cpu_replay + 抽 features...")
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

    X19, _, keep_indices = extract_features_for_trades(pre, completed)
    trades_kept = [completed[i] for i in keep_indices]
    df_trades = pd.DataFrame([{
        "ticker": t.get("ticker"),
        "buy_date": t.get("buy_date"),
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
    print(f"  最終 {len(df_merged)} 筆")

    # === V38 + V38d ===
    full_5d_median = float(np.median(p_5d))
    v38_pass = (p_next > KRONOS_NEXT_TH) & (p_5d > full_5d_median)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_v38 = X19_m[v38_pass]
    y_v38 = (rets_m[v38_pass] > 0).astype(int)
    scaler = StandardScaler()
    X_v38_s = scaler.fit_transform(X_v38)
    model = LogisticRegression(C=1.0, max_iter=500, penalty="l2")
    model.fit(X_v38_s, y_v38)

    X_all_s = scaler.transform(X19_m)
    proba_all = model.predict_proba(X_all_s)[:, 1]

    # 估計 trades/year（資料 1500 天 ~ 6 年）
    n_years = 1500 / 250  # 250 trading days/year
    print(f"\n  資料 = {n_years:.1f} 年（1500 交易日）")

    # === 3. Sweep thresholds ===
    print(f"\n[3/3] Threshold Sweep")
    print()
    print(f"{'Threshold':<12} {'n':<6} {'kept%':<8} {'wr':<8} {'avg':<8} {'total':<10} {'年化筆數':<10} {'評價'}")
    print("-" * 90)

    # Track A baseline
    a_stats = stats_of(rets_m)
    print(f"{'A (全買)':<12} {a_stats['n']:<6} {100:<8.1f} {a_stats['wr']:<8.1f} "
          f"{a_stats['avg']:<+8.2f} {a_stats['total']:<+10.0f} {a_stats['n']/n_years:<10.1f} "
          f"基準")

    # V38 zero-shot
    b_stats = stats_of(rets_m[v38_pass])
    b_kept = b_stats["n"] / a_stats["n"] * 100
    b_yearly = b_stats["n"] / n_years
    print(f"{'B (V38)':<12} {b_stats['n']:<6} {b_kept:<8.1f} {b_stats['wr']:<8.1f} "
          f"{b_stats['avg']:<+8.2f} {b_stats['total']:<+10.0f} {b_yearly:<10.1f} "
          f"V38 zero-shot")

    # V38d 不同 threshold
    print()
    sweep_results = []
    for th in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        keep = v38_pass & (proba_all > th)
        s = stats_of(rets_m[keep])
        if s["n"] == 0:
            continue
        kept_pct = s["n"] / a_stats["n"] * 100
        yearly = s["n"] / n_years

        # 評價（in-sample 偏樂觀，但仍可看相對排名）
        if yearly < 3:
            eval_str = "🔴 太低（年 < 3 筆）"
        elif yearly < 6:
            eval_str = "🟡 偏低（每 2 月 1 筆）"
        elif yearly < 12:
            eval_str = "🟢 合理（每月 1 筆）"
        elif yearly < 22:
            eval_str = "🟢 接近 89.90"
        else:
            eval_str = "—"

        sweep_results.append({
            "th": th, "n": s["n"], "kept_pct": kept_pct,
            "wr": s["wr"], "avg": s["avg"], "total": s["total"],
            "yearly": yearly,
        })

        print(f"V38d th={th:<5.2f} {s['n']:<6} {kept_pct:<8.1f} {s['wr']:<8.1f} "
              f"{s['avg']:<+8.2f} {s['total']:<+10.0f} {yearly:<10.1f} "
              f"{eval_str}")

    # === 推薦 ===
    print()
    print("=" * 70)
    print("📊 推薦 threshold")
    print("=" * 70)

    # 找年化 8-15 筆 + wr 最高的
    sweet = [s for s in sweep_results if 8 <= s["yearly"] <= 15]
    if sweet:
        best_sweet = max(sweet, key=lambda s: s["wr"])
        print(f"\n🟢 平衡選項：threshold = {best_sweet['th']}")
        print(f"   n={best_sweet['n']}, kept {best_sweet['kept_pct']:.1f}%, wr {best_sweet['wr']:.1f}%, "
              f"avg {best_sweet['avg']:+.2f}%, total {best_sweet['total']:+.0f}%")
        print(f"   年化 {best_sweet['yearly']:.1f} 筆（每月 ~{best_sweet['yearly']/12:.1f} 筆）")

    # 找年化 15-22 + wr 最高
    high_freq = [s for s in sweep_results if 15 <= s["yearly"] <= 22]
    if high_freq:
        best_hf = max(high_freq, key=lambda s: s["wr"])
        print(f"\n🟢 高頻率選項：threshold = {best_hf['th']}")
        print(f"   n={best_hf['n']}, kept {best_hf['kept_pct']:.1f}%, wr {best_hf['wr']:.1f}%, "
              f"avg {best_hf['avg']:+.2f}%, total {best_hf['total']:+.0f}%")
        print(f"   年化 {best_hf['yearly']:.1f} 筆（接近 89.90 頻率）")

    # 跟 89.90 比 total
    print(f"\n📌 對照：89.90 全買 total {a_stats['total']:+.0f}% / 年化 {a_stats['n']/n_years:.1f} 筆")
    print(f"   想要 V38d total >= 89.90 而且還能打 wr → 通常找 yearly 接近 22 的")

    # 存
    out_path = os.path.join(USER_SE, "v38d_threshold_sweep.json")
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
