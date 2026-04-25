"""
V37 跨資產 sanity test — Spearman 全期評估（學乖不用 80/20）
用法：C:\\stock-evolution> python sanity_test_cross_asset.py

對 89.90 的 133 trades，每筆 trade 看「buy_date 當天的跨資產訊號」vs trade return

判定：
  🟢 任一 signal |Spearman| ≥ 0.05 → V37 GPU 跑 24h
  🟡 0.03-0.05 → 邊際
  🔴 < 0.03 → 跳 Kronos
"""
import os, sys, pickle, json
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path: sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(_USER_SE, "stock_data_cache.pkl")
CROSS_PATH = os.path.join(_USER_SE, "cross_asset_data.pkl")


def fetch_gist_strategy():
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"
    r = urllib.request.urlopen(urllib.request.Request(GIST_URL), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s), s.get("score", "N/A")


def main():
    print("=" * 60)
    print("V37 跨資產 Gate sanity test (Spearman)")
    print("=" * 60)

    if not os.path.exists(CROSS_PATH):
        print(f"❌ 找不到 {CROSS_PATH}")
        print(f"   先跑：python fetch_cross_asset.py")
        return

    # === Step 1: 載入跨資產資料 ===
    print(f"\n[1/4] 載入跨資產資料...")
    cross = pickle.load(open(CROSS_PATH, "rb"))
    print(f"  keys: {list(cross.keys())}")

    # 建立 signal DataFrame（index=date）
    signals = {}
    for k, v in cross.items():
        if isinstance(v, pd.Series):
            signals[k] = v
        elif isinstance(v, pd.DataFrame):
            # PCR / VIX raw — 看欄位
            print(f"  {k} columns: {list(v.columns)}")
            # 暫時不加，需手動定義要用哪欄

    # 重要 signals：
    # tsm_change_pct: TSM 隔夜漲跌（buy_date 用前一天的，因為 buy 是 D+1 開盤）
    # tsm_close: 沒用，不要
    # soxx_change_pct: 同 TSM
    # nvda_change_pct: 同
    # twd_change_pct: USD/TWD 升貶（昨日）

    signal_names = []
    for s_name in ["tsm_change_pct", "soxx_change_pct", "nvda_change_pct", "twd_change_pct"]:
        if s_name in signals:
            signal_names.append(s_name)

    if not signal_names:
        print("❌ 沒有可用 signals")
        return
    print(f"\n  可用 signals: {signal_names}")

    # === Step 2: 載入 89.90 trades ===
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
    print(f"  完成交易: {len(completed)} 筆")

    # === Step 3: 對每筆 trade，抽 buy_date 前一天的跨資產 signal ===
    print(f"\n[3/4] 對齊跨資產 signals 到 89.90 trades...")
    rows = []
    for t in completed:
        bd_str = t.get("buy_date", "")
        if not bd_str:
            continue
        bd = pd.to_datetime(bd_str)
        # buy 是 D+1 開盤，所以 D 收盤後（前一個交易日的跨資產收盤）才是真正可用
        # signal date = buy_date - 1 trading day（簡化：找 bd 之前最近的有資料日）

        ret = float(t.get("return", 0))
        row = {"return": ret, "buy_date": bd}

        for s_name in signal_names:
            s = signals[s_name]
            # 找 bd 之前最近的非 NaN 值（避免 lookahead）
            mask = (s.index < bd) & ~s.isna()
            if mask.sum() > 0:
                row[s_name] = s[mask].iloc[-1]
            else:
                row[s_name] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  對齊後 {len(df)} 筆 trades")
    nans_per_signal = df[signal_names].isna().sum()
    for s in signal_names:
        print(f"    {s}: {nans_per_signal[s]} NaN")

    # === Step 4: 算 Spearman correlation（每個 signal vs return）===
    print(f"\n[4/4] Spearman correlation 計算...")
    print()
    print(f"{'Signal':<22s}{'Spearman':<12s}{'p-value':<12s}{'judgment':<15s}")
    print("-" * 60)

    any_alpha = False
    max_abs = 0.0
    best_sig = None

    for s_name in signal_names:
        valid = df[~df[s_name].isna()]
        if len(valid) < 30:
            print(f"{s_name:<22s}{'N/A (<30)':<12s}")
            continue
        rho, pval = stats.spearmanr(valid[s_name], valid["return"])
        if abs(rho) >= 0.05:
            judgment = "🟢 GREEN"
            any_alpha = True
        elif abs(rho) >= 0.03:
            judgment = "🟡 YELLOW"
        else:
            judgment = "🔴 RED"

        if abs(rho) > max_abs:
            max_abs = abs(rho)
            best_sig = s_name

        print(f"{s_name:<22s}{rho:+.4f}      {pval:.4f}      {judgment}")

    print()
    print("=" * 60)
    print("📋 V37 跨資產 Gate Go/no-go 裁決")
    print("=" * 60)
    if any_alpha:
        print(f"🟢 GREEN — 至少一個 signal 有 alpha")
        print(f"   最強：{best_sig} (|Spearman| = {max_abs:.4f})")
        print(f"   → 寫 V37 GPU monkey-patch + CPCV 驗證")
    elif max_abs >= 0.03:
        print(f"🟡 YELLOW — 邊際")
        print(f"   最強：{best_sig} (|Spearman| = {max_abs:.4f})")
        print(f"   → 可選擇做或不做")
    else:
        print(f"🔴 RED — 無 alpha")
        print(f"   最強只 {max_abs:.4f} < 0.03")
        print(f"   → 跳 V37，做 V38 Kronos 或 forward test")

    # 額外洞察：不同 horizon 的 forward correlation
    print(f"\n=== 額外：多空訊號 conditional 比較 ===")
    print(f"（看 signal 為正/負時 89.90 wr 是否不同）")
    for s_name in signal_names:
        valid = df[~df[s_name].isna()]
        if len(valid) < 30:
            continue
        thresh = valid[s_name].median()
        high = valid[valid[s_name] > thresh]
        low = valid[valid[s_name] <= thresh]
        if len(high) >= 10 and len(low) >= 10:
            wr_high = (high["return"] > 0).mean() * 100
            wr_low = (low["return"] > 0).mean() * 100
            print(f"  {s_name:<22s}: signal > median wr={wr_high:.1f}% (n={len(high)})  vs  ≤ median wr={wr_low:.1f}% (n={len(low)})  diff={wr_high-wr_low:+.1f}%")


if __name__ == "__main__":
    main()
