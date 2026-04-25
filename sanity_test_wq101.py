"""
WorldQuant 101 Alphas — sanity test on 89.90 trades
用法：C:\\stock-evolution> python sanity_test_wq101.py

目的：對 89.90 全期 ~133 筆 trades，每個 alpha 在 buy_date 前一天的值算
       Spearman vs actual_return，找出有 |corr| ≥ 0.10 的「下一階段值得進 GPU 的指標」

精選 35 個 (Kakushadze 2015 "101 Formulaic Alphas") 適合單股日線 OHLCV 的 alphas：
  - 排除需要 cap weighting / industry classification / sector neutralize 的 (大概 60+ 個)
  - 只保留純 OHLCV (open, high, low, close, volume, vwap proxy) 可算的
  - 把 cross-sectional rank 改成 time-series rank (rolling pct rank)，因我們是 per-stock 評估

學乖（V34/V36/V38 教訓）：
  - 不用 80/20 split (V36 假象)
  - 全期 Spearman + conditional WR (V34 sanity 風格)
  - 多 alpha 並聯，找 top N 個有 alpha 的進 GPU PARAMS_SPACE

判定（per-alpha）：
  |Spearman| ≥ 0.10 → 🟢🟢 強 alpha (V39 月營收 +0.1433 級)
  |Spearman| ≥ 0.05 → 🟢 中等 (V34 margin -0.1056 級)
  |Spearman| ≥ 0.03 → 🟡 marginal
  < 0.03 → 🔴 noise

最終輸出：
  - per-alpha Spearman 排行榜
  - top 5 alphas 的 conditional WR (above-median vs below-median)
  - alpha 組合（top 3 pca / equal weight）的 Spearman
  - 寫 wq101_sanity_results.json + wq101_alphas.csv
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


# ============================================================
# WorldQuant 101 alpha helpers (time-series version, per-stock)
# ============================================================

def ts_rank(x, w):
    """Time-series rank 0~1 over window w (fraction of values <= today)"""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(w - 1, n):
        window = x[i - w + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        cur = window[-1]
        if np.isnan(cur):
            continue
        out[i] = (valid <= cur).sum() / len(valid)
    return out


def ts_argmax(x, w):
    """Days since max in last w days (0 = max is today)"""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(w - 1, n):
        window = x[i - w + 1:i + 1]
        if np.all(np.isnan(window)):
            continue
        out[i] = w - 1 - np.nanargmax(window)
    return out


def ts_argmin(x, w):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(w - 1, n):
        window = x[i - w + 1:i + 1]
        if np.all(np.isnan(window)):
            continue
        out[i] = w - 1 - np.nanargmin(window)
    return out


def ts_corr(x, y, w):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(w - 1, n):
        xs = x[i - w + 1:i + 1]
        ys = y[i - w + 1:i + 1]
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() < 3:
            continue
        sx = xs[mask].std()
        sy = ys[mask].std()
        if sx == 0 or sy == 0:
            continue
        out[i] = np.corrcoef(xs[mask], ys[mask])[0, 1]
    return out


def ts_cov(x, y, w):
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(w - 1, n):
        xs = x[i - w + 1:i + 1]
        ys = y[i - w + 1:i + 1]
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() < 3:
            continue
        out[i] = np.cov(xs[mask], ys[mask])[0, 1]
    return out


def ts_sum(x, w):
    return pd.Series(x).rolling(w, min_periods=1).sum().values.astype(np.float32)


def ts_mean(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values.astype(np.float32)


def ts_std(x, w):
    return pd.Series(x).rolling(w, min_periods=1).std().values.astype(np.float32)


def ts_min(x, w):
    return pd.Series(x).rolling(w, min_periods=1).min().values.astype(np.float32)


def ts_max(x, w):
    return pd.Series(x).rolling(w, min_periods=1).max().values.astype(np.float32)


def delay(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    return out


def delta(x, k):
    return x - delay(x, k)


def returns(close):
    out = np.full_like(close, np.nan)
    out[1:] = close[1:] / close[:-1] - 1
    return out


def signed_power(x, p):
    return np.sign(x) * (np.abs(x) ** p)


def stock_vwap_proxy(open_, high, low, close, volume):
    """近似 vwap: typical price * vol 的 5 日均，再除以 vol 5 日均"""
    typ = (high + low + close) / 3.0
    pv = typ * volume
    pv_ma = pd.Series(pv).rolling(5, min_periods=1).sum().values
    v_ma = pd.Series(volume).rolling(5, min_periods=1).sum().values
    return np.where(v_ma > 0, pv_ma / v_ma, typ).astype(np.float32)


# ============================================================
# Alpha definitions (35 selected, single-stock daily OHLCV only)
# ============================================================

def compute_all_alphas(open_, high, low, close, volume):
    """
    Compute 35 alphas for single stock time series.
    Returns dict {alpha_name: array of shape (n_days,)}
    NaN for warmup period.
    """
    o, h, l, c, v = open_, high, low, close, volume
    vwap = stock_vwap_proxy(o, h, l, c, v)
    ret = returns(c)

    A = {}

    # --- Alpha #1: ts_argmax(signed_power(stddev(returns,20) if ret<0 else close, 2), 5) - 0.5
    # 簡化：ret<0 取 std, 否則取 close, 平方後 5 日 argmax
    std20 = ts_std(ret, 20)
    feat1 = np.where(ret < 0, std20, c)
    A["a001"] = ts_argmax(signed_power(feat1, 2), 5) - 0.5

    # --- Alpha #2: -ts_corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)
    log_v = np.log(np.where(v > 0, v, 1))
    co_ratio = np.where(o > 0, (c - o) / o, 0)
    A["a002"] = -ts_corr(ts_rank(delta(log_v, 2), 20), ts_rank(co_ratio, 20), 6)

    # --- Alpha #3: -ts_corr(rank(open), rank(volume), 10)
    A["a003"] = -ts_corr(ts_rank(o, 20), ts_rank(v, 20), 10)

    # --- Alpha #4: -ts_rank(low, 9)
    A["a004"] = -ts_rank(l, 9)

    # --- Alpha #5: rank((open - sum(vwap,10)/10)) * (-abs(rank((close-vwap))))
    # 簡化單股版
    A["a005"] = (o - ts_mean(vwap, 10)) * (-np.abs(c - vwap))

    # --- Alpha #6: -ts_corr(open, volume, 10)
    A["a006"] = -ts_corr(o, v, 10)

    # --- Alpha #7: 簡化版本
    # adv20<volume ? -ts_rank(abs(delta(close,7)),60) * sign(delta(close,7)) : -1
    adv20 = ts_mean(v, 20)
    d7 = delta(c, 7)
    A["a007"] = np.where(adv20 < v,
                         -ts_rank(np.abs(d7), 60) * np.sign(d7),
                         -1.0)

    # --- Alpha #8: -((sum(open,5) * sum(returns,5)) - delay(sum(open,5)*sum(returns,5),10))
    so5 = ts_sum(o, 5)
    sr5 = ts_sum(ret, 5)
    A["a008"] = -((so5 * sr5) - delay(so5 * sr5, 10))

    # --- Alpha #9: 條件式 d1 close
    d1 = delta(c, 1)
    cond_pos = ts_min(d1, 5) > 0
    cond_neg = ts_max(d1, 5) < 0
    A["a009"] = np.where(cond_pos, d1, np.where(cond_neg, d1, -d1))

    # --- Alpha #12: sign(delta(volume,1)) * (-delta(close,1))
    A["a012"] = np.sign(delta(v, 1)) * (-delta(c, 1))

    # --- Alpha #14: -delta(returns,3) * ts_corr(open, volume, 10)
    A["a014"] = -delta(ret, 3) * ts_corr(o, v, 10)

    # --- Alpha #15: -sum(corr(rank(high), rank(volume), 3), 3)
    A["a015"] = -ts_sum(ts_corr(ts_rank(h, 20), ts_rank(v, 20), 3), 3)

    # --- Alpha #16: -corr(rank(high), rank(volume), 5)
    A["a016"] = -ts_corr(ts_rank(h, 20), ts_rank(v, 20), 5)

    # --- Alpha #18: -((stddev(abs(close-open),5) + (close-open) + corr(close,open,10)))
    A["a018"] = -(ts_std(np.abs(c - o), 5) + (c - o) + ts_corr(c, o, 10))

    # --- Alpha #19: sign((close-delay(close,7)) + delta(close,7)) * (1 + rank(1+sum(returns,250)))
    s_term = (c - delay(c, 7)) + delta(c, 7)
    sum_r = ts_sum(ret, 250)
    A["a019"] = -np.sign(s_term) * (1 + ts_rank(1 + sum_r, 250))

    # --- Alpha #21: 條件式 ma 比較
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)
    sma2 = ts_mean(c, 2)
    cond1 = (sma8 + std8) < sma2
    cond2 = sma2 < (sma8 - std8)
    cond3 = (1 < (v / adv20)) | ((v / adv20) == 1)
    a21 = np.where(cond1, -1.0,
          np.where(cond2, 1.0,
          np.where(cond3, 1.0, -1.0)))
    A["a021"] = a21.astype(np.float32)

    # --- Alpha #22: -delta(corr(high,volume,5),5) * rank(stddev(close,20))
    A["a022"] = -delta(ts_corr(h, v, 5), 5) * ts_rank(ts_std(c, 20), 50)

    # --- Alpha #23: sum(high,20)/20 < high ? -delta(high,2) : 0
    sma_h20 = ts_mean(h, 20)
    A["a023"] = np.where(sma_h20 < h, -delta(h, 2), 0.0)

    # --- Alpha #24: 條件 ma 衰退
    sma100 = ts_mean(c, 100)
    delta100 = delta(sma100, 100)
    delay100 = delay(c, 100)
    cond = np.where(delay100 != 0, delta100 / delay100 <= 0.05, False)
    A["a024"] = np.where(cond, -(c - ts_min(c, 100)), -delta(c, 3))

    # --- Alpha #26: -ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)
    A["a026"] = -ts_max(ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5), 3)

    # --- Alpha #28: corr(adv20, low, 5) + (high+low)/2 - close
    A["a028"] = ts_corr(adv20, l, 5) + (h + l) / 2 - c

    # --- Alpha #32: scale(sum(close,7)/7 - close) + scale(corr(vwap, delay(close,5), 230))
    A["a032"] = (ts_mean(c, 7) - c) + 20 * ts_corr(vwap, delay(c, 5), 230)

    # --- Alpha #33: -((1 - open/close)^1)
    A["a033"] = -(1 - np.where(c > 0, o / c, 1))

    # --- Alpha #34: 1 - rank(stddev(returns,2)/stddev(returns,5)) + 1 - rank(delta(close,1))
    std_r2 = ts_std(ret, 2)
    std_r5 = ts_std(ret, 5)
    ratio = np.where(std_r5 > 0, std_r2 / std_r5, 1)
    A["a034"] = (1 - ts_rank(ratio, 50)) + (1 - ts_rank(delta(c, 1), 50))

    # --- Alpha #35: ts_rank(volume,32) * (1-ts_rank(close+high-low,16)) * (1-ts_rank(returns,32))
    A["a035"] = ts_rank(v, 32) * (1 - ts_rank(c + h - l, 16)) * (1 - ts_rank(ret, 32))

    # --- Alpha #37: corr(delay(open-close,1), close, 200) + rank(open-close)
    A["a037"] = ts_corr(delay(o - c, 1), c, 200) + (o - c)

    # --- Alpha #38: -ts_rank(close,10) * rank(close/open)
    A["a038"] = -ts_rank(c, 10) * np.where(o > 0, c / o, 1)

    # --- Alpha #40: -(rank(stddev(high,10)) * corr(high,volume,10))
    A["a040"] = -(ts_rank(ts_std(h, 10), 50) * ts_corr(h, v, 10))

    # --- Alpha #41: ((high*low)^0.5) - vwap
    A["a041"] = np.sqrt(np.maximum(h * l, 0)) - vwap

    # --- Alpha #42: rank(vwap-close)/rank(vwap+close)
    rk_diff = ts_rank(vwap - c, 50)
    rk_sum = ts_rank(vwap + c, 50)
    A["a042"] = np.where(rk_sum > 0, rk_diff / rk_sum, 0)

    # --- Alpha #43: ts_rank(volume/adv20, 20) * ts_rank(-delta(close,7), 8)
    A["a043"] = ts_rank(np.where(adv20 > 0, v / adv20, 1), 20) * ts_rank(-delta(c, 7), 8)

    # --- Alpha #44: -corr(high, rank(volume), 5)
    A["a044"] = -ts_corr(h, ts_rank(v, 50), 5)

    # --- Alpha #46: 條件式 close 趨勢加速
    delay20 = delay(c, 20)
    delay10 = delay(c, 10)
    diff = ((delay20 - delay10) / 10.0) - ((delay10 - c) / 10.0)
    A["a046"] = np.where(0.25 < diff, -1.0,
                np.where(diff < 0, 1.0, -(c - delay(c, 1))))

    # --- Alpha #49: 同 #46 變體
    A["a049"] = np.where(diff < -0.1, 1.0, -(c - delay(c, 1)))

    # --- Alpha #51: 同 #46 變體
    A["a051"] = np.where(diff < -0.05, 1.0, -(c - delay(c, 1)))

    # --- Alpha #54: -((low-close)*open^5) / ((low-high)*close^5)
    num = -((l - c) * (o ** 5))
    den = (l - h) * (c ** 5)
    A["a054"] = np.where(den != 0, num / den, 0)

    # --- Alpha #101: (close-open)/(high-low+0.001)
    A["a101"] = (c - o) / (h - l + 0.001)

    return A


def main():
    print("=" * 80)
    print("WorldQuant 101 Alphas — sanity test on 89.90 trades")
    print("=" * 80)

    if not os.path.exists(CACHE_PATH):
        print(f"❌ {CACHE_PATH} 不存在")
        return

    # === Step 1: 89.90 trades ===
    print(f"\n[1/4] 跑 89.90 cpu_replay...")
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
    print(f"  89.90 trades: {len(completed)} 筆完成")

    tickers = pre["tickers"]
    dates = pre["dates"]
    close_arr = pre["close"]   # (n_stocks, n_days)
    high_arr = pre["high"] if "high" in pre else None
    low_arr = pre["low"] if "low" in pre else None

    # 自己讀 OHLCV (precompute 內 high/low/open 不一定 export)
    print(f"\n[2/4] 重建 OHLCV arrays...")
    ns = len(tickers)
    nd = len(dates)
    o_arr = np.zeros((ns, nd), dtype=np.float32)
    h_arr = np.zeros((ns, nd), dtype=np.float32)
    l_arr = np.zeros((ns, nd), dtype=np.float32)
    v_arr = np.zeros((ns, nd), dtype=np.float32)
    for si, t in enumerate(tickers):
        df = data_dict[t].tail(nd)
        o_arr[si] = df["Open"].values[-nd:]
        h_arr[si] = df["High"].values[-nd:]
        l_arr[si] = df["Low"].values[-nd:]
        v_arr[si] = df["Volume"].values[-nd:]
    c_arr = close_arr

    # date → idx
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # === Step 3: 對每筆 trade 算 alphas at buy_date - 1 (D-1, 訊號日) ===
    print(f"\n[3/4] 對每筆 trade 計算 35 個 alphas (at D-1, 即訊號日)...")
    rows = []
    skipped = 0
    for t in completed:
        ticker = t.get("ticker", "")
        bd_str = t.get("buy_date", "")
        if ticker not in ticker_to_idx or bd_str not in date_to_day:
            skipped += 1
            continue
        si = ticker_to_idx[ticker]
        bd_idx = date_to_day[bd_str]
        # 訊號是 D-1 收盤計算，D 開盤買 → 評估 alpha 必須用 bd_idx-1
        signal_idx = bd_idx - 1
        if signal_idx < 250:  # 需 250 天 warmup (sum_returns 250)
            skipped += 1
            continue

        # 算這支股票的全 alpha
        alphas = compute_all_alphas(
            o_arr[si], h_arr[si], l_arr[si], c_arr[si], v_arr[si]
        )

        row = {"ticker": ticker, "buy_date": bd_str, "actual_return": float(t.get("return", 0))}
        for name, arr in alphas.items():
            v_at = arr[signal_idx] if signal_idx < len(arr) else np.nan
            row[name] = float(v_at) if np.isfinite(v_at) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} 筆有 alpha 值 (skip {skipped})")

    if len(df) < 30:
        print(f"  ❌ 樣本太少")
        return

    # === Step 4: per-alpha Spearman ===
    print(f"\n[4/4] Per-alpha Spearman correlation...\n")
    # 只抓 a + 3 位數字（a001-a999），避免吃到 actual_return
    import re
    alpha_pattern = re.compile(r"^a\d{3}$")
    alpha_names = [c for c in df.columns if alpha_pattern.match(c)]
    rets = df["actual_return"].values

    results = []
    for name in alpha_names:
        vals = df[name].values
        mask = np.isfinite(vals) & np.isfinite(rets)
        if mask.sum() < 30:
            continue
        sp_corr, sp_p = scipy_stats.spearmanr(vals[mask], rets[mask])
        if not np.isfinite(sp_corr):
            continue
        # conditional WR (above median vs below median)
        med = np.median(vals[mask])
        above = rets[mask][vals[mask] > med]
        below = rets[mask][vals[mask] <= med]
        wr_above = (above > 0).mean() * 100 if len(above) > 0 else 0
        wr_below = (below > 0).mean() * 100 if len(below) > 0 else 0
        avg_above = above.mean() if len(above) > 0 else 0
        avg_below = below.mean() if len(below) > 0 else 0

        results.append({
            "alpha": name,
            "spearman": float(sp_corr),
            "p_value": float(sp_p),
            "n": int(mask.sum()),
            "wr_above_med": float(wr_above),
            "wr_below_med": float(wr_below),
            "wr_diff": float(wr_above - wr_below),
            "avg_above_med": float(avg_above),
            "avg_below_med": float(avg_below),
            "avg_diff": float(avg_above - avg_below),
        })

    results.sort(key=lambda r: -abs(r["spearman"]))

    # === 輸出排行榜 ===
    print(f"{'Rank':<5} {'Alpha':<8} {'|Sp|':<8} {'Spear':<10} {'p':<8} "
          f"{'wr↑med':<8} {'wr↓med':<8} {'Δwr':<8} {'Δavg':<8} {'Eval'}")
    print("-" * 95)

    n_strong = 0
    n_mid = 0
    n_marginal = 0

    for i, r in enumerate(results, 1):
        absp = abs(r["spearman"])
        if absp >= 0.10:
            tag = "🟢🟢"
            n_strong += 1
        elif absp >= 0.05:
            tag = "🟢"
            n_mid += 1
        elif absp >= 0.03:
            tag = "🟡"
            n_marginal += 1
        else:
            tag = "🔴"
        print(f"{i:<5} {r['alpha']:<8} {absp:<8.4f} {r['spearman']:<+10.4f} "
              f"{r['p_value']:<8.4f} {r['wr_above_med']:<8.1f} {r['wr_below_med']:<8.1f} "
              f"{r['wr_diff']:<+8.1f} {r['avg_diff']:<+8.2f} {tag}")

    # === 組合 alpha (top 5 equal-weight z-score) ===
    print()
    print("=" * 80)
    print("Top 5 alphas combination (equal-weight z-score sum)")
    print("=" * 80)

    top5 = results[:5]
    if len(top5) >= 3:
        combo = np.zeros(len(df))
        valid_count = np.zeros(len(df))
        for r in top5:
            vals = df[r["alpha"]].values
            mask = np.isfinite(vals)
            if mask.sum() < 10:
                continue
            mean_v = np.nanmean(vals)
            std_v = np.nanstd(vals)
            if std_v == 0:
                continue
            z = np.where(mask, (vals - mean_v) / std_v, 0)
            # 反向：負 spearman 的 alpha 取負號
            sign = -1 if r["spearman"] < 0 else 1
            combo += sign * z
            valid_count += mask.astype(float)
        combo = np.where(valid_count > 0, combo / np.maximum(valid_count, 1), 0)
        mask_combo = valid_count >= 3
        if mask_combo.sum() >= 30:
            sp_c, sp_pc = scipy_stats.spearmanr(combo[mask_combo], rets[mask_combo])
            print(f"  Combined (signed z-sum, n={mask_combo.sum()}): Spearman {sp_c:+.4f} (p={sp_pc:.4f})")

    # === 判定 ===
    print()
    print("=" * 80)
    print("📊 WQ101 Sanity 判定")
    print("=" * 80)
    print(f"\n  測了 {len(results)} 個 alphas")
    print(f"  🟢🟢 強 (|Sp|≥0.10): {n_strong}")
    print(f"  🟢   中 (|Sp|≥0.05): {n_mid}")
    print(f"  🟡   邊際(|Sp|≥0.03): {n_marginal}")
    print(f"  🔴   雜訊 (|Sp|<0.03): {len(results) - n_strong - n_mid - n_marginal}")

    if n_strong >= 3:
        print(f"\n  🟢🟢🟢 多個強 alpha → 值得加進 GPU PARAMS_SPACE 跑 5090")
    elif n_strong >= 1:
        print(f"\n  🟢🟢 至少 1 個強 alpha → 跑 backfill + CPCV 驗證 top 3")
    elif n_mid >= 5:
        print(f"\n  🟢 多個中等 alpha → 試組合 sanity (已輸出)")
    elif n_mid >= 1:
        print(f"\n  🟡 marginal — 跟 V34 margin 同級，CPCV 驗證後決定")
    else:
        print(f"\n  🔴 全部 noise — WQ101 對 89.90 沒額外 alpha，89.90 已吃完技術面")
        print(f"     → 接受 89.90 final，把 5090 算力用在 Kronos fine-tune")

    # 存
    out_json = os.path.join(USER_SE, "wq101_sanity_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_trades": len(df),
            "summary": {"strong": n_strong, "mid": n_mid, "marginal": n_marginal,
                        "noise": len(results) - n_strong - n_mid - n_marginal},
            "alphas": results,
        }, f, indent=2)
    df.to_csv(os.path.join(USER_SE, "wq101_alphas.csv"), index=False)
    print(f"\n結果存到 wq101_sanity_results.json + wq101_alphas.csv")


if __name__ == "__main__":
    main()
