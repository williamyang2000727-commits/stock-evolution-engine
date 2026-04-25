"""
V38d: V38 zero-shot + ML 過濾 false positive（最後一試）

設計哲學：
  V38 zero-shot ensemble (next > 0.8 AND 5d > median) 已經是最強過濾規則
  → 不取代它，而是「在 V38 通過的子集上再用 ML 過濾」

  V38 zero-shot 攻擊面：n_break 11/14 但仍有 1/14 虧
  → 用 ML 找出「V38 想買但實際會虧」的 pattern → 二次過濾

階段：
  1. 對 89.90 的 133 筆 trades 套 V38 rule（next > 0.8 AND 5d > path-internal median）
     → 子集大約 21 筆 (15.8% kept)
  2. 在這個子集上訓練 ML（小樣本）
     → 如果 V38 子集本身就 90%+ wr，ML 大概學不到東西
     → 如果還有 ~30% 失敗 → 有空間

CPCV LOO 14 path：
  - train = 13 path 的 V38 子集
  - test = 1 path 的 V38 子集
  - 在 test V38 子集上跑 ML proba > X 過濾

判定：
  🟢🟢 mean wr↑ > 14.28% AND p25 > 5.58% → 比 V38 zero-shot 嚴格更強
  🟢 mean wr↑ > 14.28% （但 p25 可能更差） → 攻擊面取捨
  🟡 mean wr↑ > 8% → 不確定
  🔴 其他 → V38 zero-shot final，停止 fine-tune 方向
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from itertools import combinations

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base
from metalabel_features import extract_features_for_trades, FEATURE_NAMES

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
SANITY_CSV = os.path.join(USER_SE, "kronos_sanity_results.csv")
N_GROUPS = 6
K_TEST = 2
WARMUP = 60
KRONOS_NEXT_TH = 0.8


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s), s.get("score", "N/A")


def split_into_groups(n_days, warmup, n_groups):
    g_size = (n_days - warmup) // n_groups
    return [(warmup + i * g_size, warmup + (i + 1) * g_size if i < n_groups - 1 else n_days)
            for i in range(n_groups)]


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
    print("V38d: V38 zero-shot + ML 二次過濾 — Quick Test")
    print("=" * 70)
    print()
    print("V38c 失敗（mean +11.53% < zero-shot +14.28%）")
    print("V38d 改攻擊面：不取代 V38 rule，而是在 V38 子集上做 ML 過濾")
    print()

    # === 1. 讀 sanity + 跑 89.90 ===
    if not os.path.exists(SANITY_CSV):
        print(f"❌ {SANITY_CSV} 不存在")
        return
    df_sanity = pd.read_csv(SANITY_CSV)
    print(f"[1/6] 讀 sanity CSV：{len(df_sanity)} 筆 ✅")

    print(f"\n[2/6] 跑 89.90 cpu_replay...")
    params, _ = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    n_days = pre["n_days"]
    dates = pre["dates"]
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}

    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]
    print(f"  89.90 trades: {len(completed)} 筆")

    # === 2. 抽 19 features + 合併 Kronos ===
    print(f"\n[3/6] 抽 features + 合併 Kronos...")
    X19, y, keep_indices = extract_features_for_trades(pre, completed)
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

    day_idx = np.array([date_to_day.get(bd, -1) for bd in df_merged["buy_date"]])
    valid_mask = day_idx >= 0
    X19_m = X19_m[valid_mask]
    rets_m = rets_m[valid_mask]
    p_next = p_next[valid_mask]
    p_5d = p_5d[valid_mask]
    day_idx = day_idx[valid_mask]
    print(f"  最終 {len(X19_m)} 筆")

    # === 3. CPCV LOO 評估「V38 zero-shot baseline」+「V38d ML refine」===
    print(f"\n[4/6] CPCV LOO baseline（V38 zero-shot）+ V38d ML refine...")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def eval_v38_zero_shot():
        """V38 zero-shot baseline: next > 0.8 AND 5d > path-internal median"""
        per_path = []
        for pi, gi in enumerate(test_combos):
            ranges = [groups[g] for g in gi]
            in_test = np.zeros(len(X19_m), dtype=bool)
            for s, e in ranges:
                in_test |= (day_idx >= s) & (day_idx < e)
            if in_test.sum() < 5: continue
            rets_te = rets_m[in_test]
            p_n_te = p_next[in_test]
            p_5_te = p_5d[in_test]
            med_5d = np.median(p_5_te)
            keep = (p_n_te > KRONOS_NEXT_TH) & (p_5_te > med_5d)
            if keep.sum() < 3: continue
            raw_s = stats_of(rets_te)
            filt_s = stats_of(rets_te[keep])
            wr_imp = filt_s["wr"] - raw_s["wr"]
            kept_pct = filt_s["n"] / raw_s["n"] * 100
            per_path.append({
                "wr_imp": wr_imp, "kept_pct": kept_pct,
                "raw_wr": raw_s["wr"], "filt_wr": filt_s["wr"],
                "n_in_filt": filt_s["n"],
            })
        return per_path

    def eval_v38d_refine(ml_th_offset, C):
        """V38d: 先套 V38 rule 拿子集，再用 ML 過濾子集中的 false positive"""
        per_path = []
        for pi, gi in enumerate(test_combos):
            ranges = [groups[g] for g in gi]
            in_test = np.zeros(len(X19_m), dtype=bool)
            for s, e in ranges:
                in_test |= (day_idx >= s) & (day_idx < e)
            train_mask = ~in_test
            test_mask = in_test
            if train_mask.sum() < 15 or test_mask.sum() < 5: continue

            # train set V38 子集
            p_n_tr = p_next[train_mask]
            p_5_tr = p_5d[train_mask]
            med_5d_tr = np.median(p_5_tr)
            v38_pass_tr = (p_n_tr > KRONOS_NEXT_TH) & (p_5_tr > med_5d_tr)
            if v38_pass_tr.sum() < 8:  # train V38 子集太少
                continue
            X_tr_v38 = X19_m[train_mask][v38_pass_tr]
            y_tr_v38 = (rets_m[train_mask][v38_pass_tr] > 0).astype(int)
            if y_tr_v38.sum() == 0 or y_tr_v38.sum() == len(y_tr_v38):
                # 全贏或全輸，沒法訓練
                continue

            # test set V38 子集
            rets_te = rets_m[test_mask]
            p_n_te = p_next[test_mask]
            p_5_te = p_5d[test_mask]
            med_5d_te = np.median(p_5_te)
            v38_pass_te = (p_n_te > KRONOS_NEXT_TH) & (p_5_te > med_5d_te)
            if v38_pass_te.sum() < 3: continue

            X_te_v38 = X19_m[test_mask][v38_pass_te]
            rets_te_v38 = rets_te[v38_pass_te]

            # ML
            scaler = StandardScaler()
            try:
                X_tr_s = scaler.fit_transform(X_tr_v38)
                X_te_s = scaler.transform(X_te_v38)
                m = LogisticRegression(C=C, max_iter=500, penalty="l2")
                m.fit(X_tr_s, y_tr_v38)
                proba_tr = m.predict_proba(X_tr_s)[:, 1]
                proba_te = m.predict_proba(X_te_s)[:, 1]
            except Exception:
                continue

            th = float(np.median(proba_tr)) + ml_th_offset
            keep_ml = proba_te > th
            if keep_ml.sum() < 2: continue

            # 完整路徑統計（vs raw test 不過濾）
            raw_s = stats_of(rets_te)
            filt_s = stats_of(rets_te_v38[keep_ml])
            wr_imp = filt_s["wr"] - raw_s["wr"]
            kept_pct = filt_s["n"] / raw_s["n"] * 100
            per_path.append({
                "wr_imp": wr_imp, "kept_pct": kept_pct,
                "raw_wr": raw_s["wr"], "filt_wr": filt_s["wr"],
                "n_in_filt": filt_s["n"],
            })
        return per_path

    def summarize(per_path, name):
        if not per_path:
            return None
        wr_imps = np.array([p["wr_imp"] for p in per_path])
        kept_pcts = np.array([p["kept_pct"] for p in per_path])
        n_break = int((wr_imps >= 5).sum())
        n_pos = int((wr_imps > 0).sum())
        return {
            "name": name,
            "n_valid": len(per_path),
            "n_break": n_break,
            "n_pos": n_pos,
            "mean_wr_imp": float(wr_imps.mean()),
            "median_wr_imp": float(np.median(wr_imps)),
            "p25_wr_imp": float(np.percentile(wr_imps, 25)),
            "min_wr_imp": float(wr_imps.min()),
            "max_wr_imp": float(wr_imps.max()),
            "mean_kept_pct": float(kept_pcts.mean()),
        }

    # Baseline V38 zero-shot
    baseline = summarize(eval_v38_zero_shot(), "V38 zero-shot (baseline 重算)")
    print(f"\n  [Baseline] V38 zero-shot 重算：")
    if baseline:
        print(f"    n_break {baseline['n_break']}/{baseline['n_valid']}, "
              f"mean {baseline['mean_wr_imp']:+5.2f}%, p25 {baseline['p25_wr_imp']:+5.2f}%, "
              f"kept {baseline['mean_kept_pct']:.0f}%, n_pos {baseline['n_pos']}/{baseline['n_valid']}")

    # V38d sweep
    print(f"\n  [V38d] ML refine 多配置 sweep：")
    configs = []
    for C in [0.1, 0.3, 1.0]:
        for off in [-0.05, 0.0, 0.05, 0.10]:
            configs.append({"C": C, "off": off})

    best = None
    for cfg in configs:
        per_path = eval_v38d_refine(cfg["off"], cfg["C"])
        s = summarize(per_path, f"V38d-C{cfg['C']}-off{cfg['off']:+.2f}")
        if s is None:
            print(f"  ⚠️ V38d-C{cfg['C']}-off{cfg['off']:+.2f}: 樣本不足")
            continue
        flag = "🟢" if s["mean_wr_imp"] >= 14.28 else ("🟡" if s["mean_wr_imp"] >= 8 else "🔴")
        print(f"  {flag} {s['name']:<28s}: n_break {s['n_break']:>2d}/{s['n_valid']:>2d}, "
              f"mean {s['mean_wr_imp']:+5.2f}%, p25 {s['p25_wr_imp']:+5.2f}%, "
              f"kept {s['mean_kept_pct']:.0f}%, n_pos {s['n_pos']}/{s['n_valid']}")
        if best is None or (s["n_break"], s["mean_wr_imp"]) > (best["n_break"], best["mean_wr_imp"]):
            best = s

    # === 5. 判定 ===
    print(f"\n[5/6] 判定...")
    print()
    print("=" * 70)
    print("📊 V38d 結果")
    print("=" * 70)

    if baseline:
        print(f"\n【V38 zero-shot 本次重算 baseline】")
        print(f"  n_break {baseline['n_break']}/{baseline['n_valid']}, mean {baseline['mean_wr_imp']:+.2f}%, p25 {baseline['p25_wr_imp']:+.2f}%, n_pos {baseline['n_pos']}/{baseline['n_valid']}")

    print(f"\n【V38d 之前已知】mean +14.28%, p25 +5.58%, n_pos 13/14（V38 paper trading 用的）")

    if best:
        print(f"\n【V38d best】 {best['name']}")
        print(f"  n_break = {best['n_break']}/{best['n_valid']}")
        print(f"  mean wr↑ = {best['mean_wr_imp']:+.2f}%")
        print(f"  p25 wr↑ = {best['p25_wr_imp']:+.2f}%")
        print(f"  median = {best['median_wr_imp']:+.2f}%")
        print(f"  min/max = {best['min_wr_imp']:+.2f}% / {best['max_wr_imp']:+.2f}%")
        print(f"  positive paths = {best['n_pos']}/{best['n_valid']}")
        print(f"  mean kept = {best['mean_kept_pct']:.1f}%")

        if baseline and best["mean_wr_imp"] > baseline["mean_wr_imp"] + 2 and best["p25_wr_imp"] > baseline["p25_wr_imp"]:
            print(f"\n  🟢🟢 V38d 顯著比 V38 zero-shot 強！")
            print(f"      → 整合到 production，daily_scan 加二層 ML 過濾")
        elif baseline and best["mean_wr_imp"] > baseline["mean_wr_imp"]:
            print(f"\n  🟡 V38d 略強，但邊際小")
            print(f"      → 不值得增加複雜度，繼續用 V38 zero-shot")
        else:
            print(f"\n  🔴 V38d 沒贏 V38 zero-shot")
            print(f"      → 接受 V38 zero-shot final，停止 fine-tune 方向")
    else:
        print(f"\n🔴 V38d 全失敗（樣本不足或訓練失敗）")
        print(f"   → 接受 V38 zero-shot final")

    out_path = os.path.join(USER_SE, "v38d_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump({"baseline": baseline, "best": best}, f, indent=2, default=str)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
