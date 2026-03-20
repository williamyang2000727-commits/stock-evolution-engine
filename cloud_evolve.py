#!/usr/bin/env python3
"""
雲端進化引擎 — 跑在 GitHub Actions 上
自己下載資料、回測、推播 Telegram
"""

import numpy as np
import json
import os
import sys
import time
import requests
import yfinance as yf
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Telegram（從環境變數讀，不寫死在公開 repo）===
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = os.environ.get("TELEGRAM_CHAT_IDS", "").split(",")

def telegram_push(msg):
    for cid in CHAT_IDS:
        cid = cid.strip()
        if not cid or not BOT_TOKEN:
            continue
        try:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": cid, "text": msg}, timeout=10
            )
        except:
            pass

# === 中文名 ===
CN_NAMES = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "2303.TW": "聯電",
    "2382.TW": "廣達", "3231.TW": "緯創", "2353.TW": "宏碁", "2357.TW": "華碩",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "2412.TW": "中華電", "1301.TW": "台塑", "2603.TW": "長榮", "2609.TW": "陽明",
    "1216.TW": "統一", "2002.TW": "中鋼", "2308.TW": "台達電", "3711.TW": "日月光投控",
    "2409.TW": "友達", "3481.TW": "群創", "2356.TW": "英業達", "2324.TW": "仁寶",
    "4938.TW": "和碩", "2337.TW": "旺宏", "2344.TW": "華邦電", "3037.TW": "欣興",
    "6770.TW": "力積電", "3576.TW": "聯合再生", "1802.TW": "台玻", "8039.TW": "台虹",
    "2485.TW": "兆赫", "1711.TW": "永光", "1717.TW": "長興", "2313.TW": "聯電",
    "6505.TW": "台塑化", "1303.TW": "南亞", "2406.TW": "國碩", "8150.TW": "南茂",
    "2615.TW": "萬海", "2618.TW": "長榮航", "2610.TW": "華航", "2912.TW": "統一超",
    "1101.TW": "台泥", "2880.TW": "華南金", "2885.TW": "元大金", "2890.TW": "永豐金",
}

def get_name(t):
    n = CN_NAMES.get(t, "")
    if not n or " " in n or (len(n) > 0 and n[0].isupper()):
        return t.replace(".TW", "")
    return n

# === 下載股票資料 ===
TW_TICKERS = [
    "2330.TW","2454.TW","2317.TW","2303.TW","2382.TW","3231.TW","2353.TW","2357.TW",
    "2881.TW","2882.TW","2891.TW","2886.TW","2412.TW","1301.TW","2603.TW","2609.TW",
    "1216.TW","2002.TW","2308.TW","3711.TW","2409.TW","3481.TW","2356.TW","2324.TW",
    "4938.TW","2337.TW","2344.TW","3037.TW","6770.TW","3576.TW","1802.TW","8039.TW",
    "2485.TW","1711.TW","1717.TW","2313.TW","6505.TW","1303.TW","2406.TW","8150.TW",
    "2615.TW","2618.TW","2610.TW","2912.TW","1101.TW","2880.TW","2885.TW","2890.TW",
    "2884.TW","2892.TW","1326.TW","2345.TW","3017.TW","6669.TW","2379.TW","3034.TW",
    "2408.TW","3661.TW","2301.TW","2395.TW","2327.TW","3008.TW","2634.TW","1513.TW",
    "2049.TW","1504.TW","2207.TW","1476.TW","9910.TW","2368.TW","2449.TW","2383.TW",
]

def download_data():
    data = {}
    for ticker in TW_TICKERS:
        try:
            h = yf.Ticker(ticker).history(period="6mo")
            if len(h) >= 40:
                data[ticker] = h
        except:
            continue
    return data

def filter_top_volume(data, top_n=50):
    vol_rank = {}
    for t, h in data.items():
        if "Volume" in h.columns and len(h) >= 20:
            vol_rank[t] = h["Volume"].tail(20).mean()
    top = sorted(vol_rank, key=vol_rank.get, reverse=True)[:top_n]
    return {k: data[k] for k in top}

# === 預算指標 ===
def precompute(data):
    tickers = list(data.keys())
    min_len = min(len(data[t]) for t in tickers)
    min_len = min(min_len, 120)
    n = len(tickers)

    close = np.zeros((n, min_len))
    high = np.zeros((n, min_len))
    low = np.zeros((n, min_len))
    opn = np.zeros((n, min_len))
    volume = np.zeros((n, min_len))
    dates = None

    for si, t in enumerate(tickers):
        h = data[t]
        close[si] = h["Close"].values[-min_len:]
        high[si] = h["High"].values[-min_len:]
        low[si] = h["Low"].values[-min_len:]
        opn[si] = h["Open"].values[-min_len:]
        volume[si] = h["Volume"].values[-min_len:]
        if dates is None:
            dates = h.index[-min_len:]

    ind = {"close": close, "high": high, "low": low, "open": opn, "volume": volume}

    # RSI
    delta = np.diff(close, axis=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_g = np.zeros_like(close)
    avg_l = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        if i == 14:
            avg_g[:, i] = np.mean(gain[:, :14], axis=1)
            avg_l[:, i] = np.mean(loss[:, :14], axis=1)
        else:
            avg_g[:, i] = (avg_g[:, i-1] * 13 + gain[:, i-1]) / 14
            avg_l[:, i] = (avg_l[:, i-1] * 13 + loss[:, i-1]) / 14
    rs = np.where(avg_l > 0, avg_g / avg_l, 100)
    ind["rsi"] = 100 - 100 / (1 + rs)

    # MA
    for w in [3, 5, 8, 10, 15, 20, 30, 60]:
        ma = np.zeros_like(close)
        for i in range(w, close.shape[1]):
            ma[:, i] = np.mean(close[:, i-w:i], axis=1)
        ind[f"ma{w}"] = ma

    # BB
    bb_mid = ind["ma20"]
    bb_std = np.zeros_like(close)
    for i in range(20, close.shape[1]):
        bb_std[:, i] = np.std(close[:, i-20:i], axis=1)
    bb_u = bb_mid + 2 * bb_std
    bb_l = bb_mid - 2 * bb_std
    bb_r = bb_u - bb_l
    ind["bb_pos"] = np.where(bb_r > 0, (close - bb_l) / bb_r, 0.5)
    ind["bb_width"] = np.where(bb_mid > 0, bb_r / bb_mid, 0)

    # Volume ratio
    vol_ma = np.zeros_like(volume)
    for i in range(20, volume.shape[1]):
        vol_ma[:, i] = np.mean(volume[:, i-20:i], axis=1)
    ind["vol_ratio"] = np.where(vol_ma > 0, volume / vol_ma, 1)

    # MACD
    e12 = np.zeros_like(close); e26 = np.zeros_like(close)
    e12[:, 0] = close[:, 0]; e26[:, 0] = close[:, 0]
    for i in range(1, close.shape[1]):
        e12[:, i] = e12[:, i-1] * (1 - 2/13) + close[:, i] * 2/13
        e26[:, i] = e26[:, i-1] * (1 - 2/27) + close[:, i] * 2/27
    ml = e12 - e26
    ms = np.zeros_like(close); ms[:, 0] = ml[:, 0]
    for i in range(1, close.shape[1]):
        ms[:, i] = ms[:, i-1] * (1 - 2/10) + ml[:, i] * 2/10
    ind["macd_line"] = ml
    ind["macd_hist"] = ml - ms

    # KD
    l9 = np.zeros_like(close); h9 = np.zeros_like(close)
    for i in range(9, close.shape[1]):
        l9[:, i] = np.min(low[:, i-9:i+1], axis=1)
        h9[:, i] = np.max(high[:, i-9:i+1], axis=1)
    rsv = np.where((h9 - l9) > 0, (close - l9) / (h9 - l9) * 100, 50)
    kv = np.zeros_like(close); dv = np.zeros_like(close)
    kv[:, 0] = 50; dv[:, 0] = 50
    for i in range(1, close.shape[1]):
        kv[:, i] = kv[:, i-1] * 2/3 + rsv[:, i] * 1/3
        dv[:, i] = dv[:, i-1] * 2/3 + kv[:, i] * 1/3
    ind["k_val"] = kv; ind["d_val"] = dv

    # Momentum
    for d in [3, 5, 10]:
        m = np.zeros_like(close)
        m[:, d:] = (close[:, d:] / close[:, :-d] - 1) * 100
        ind[f"mom_{d}"] = m

    ind["is_green"] = (close > opn).astype(np.float64)
    gap = np.zeros_like(close)
    gap[:, 1:] = (opn[:, 1:] / close[:, :-1] - 1) * 100
    ind["gap"] = gap

    h20 = np.zeros_like(close)
    for i in range(20, close.shape[1]):
        h20[:, i] = np.max(high[:, i-20:i+1], axis=1)
    ind["near_high"] = np.where(h20 > 0, (close / h20 - 1) * 100, 0)

    vp = np.zeros_like(ind["vol_ratio"])
    vp[:, 1:] = ind["vol_ratio"][:, :-1]
    ind["vol_prev"] = vp

    return {"tickers": tickers, "dates": dates, "n_stocks": n, "n_days": min_len, "ind": ind}

# === 回測（單組參數）===
def backtest_one(args):
    p, pre = args
    ind = pre["ind"]
    ns, nd = pre["n_stocks"], pre["n_days"]

    mfw = p.get("ma_fast_w", 5)
    msw = p.get("ma_slow_w", 20)
    if mfw >= msw: return None
    maf = ind.get(f"ma{mfw}", ind["ma5"])
    mas = ind.get(f"ma{msw}", ind["ma20"])
    ma60 = ind.get("ma60", ind["ma20"])
    md = p.get("momentum_days", 5)
    mom = ind.get(f"mom_{md}", ind["mom_5"])

    trades = []
    holding = None

    for day in range(30, nd - 1):
        if holding:
            si = holding["si"]
            cur = ind["close"][si, day]
            dh = day - holding["bd"]
            ret = (cur / holding["bp"] - 1) * 100
            if dh < 1: continue
            if cur > holding["pk"]: holding["pk"] = cur
            sell = None

            if ret <= p["stop_loss"]: sell = "停損"
            if not sell and p.get("use_take_profit", 1) and ret >= p["take_profit"]: sell = "停利"
            ts = p.get("trailing_stop", 0)
            if not sell and ts > 0 and holding["pk"] > holding["bp"]:
                if (cur / holding["pk"] - 1) * 100 <= -ts: sell = "移動停利"
            if not sell and p.get("use_rsi_sell", 1) and ind["rsi"][si, day] >= p.get("rsi_sell", 90): sell = "RSI超買"
            if not sell and p.get("use_macd_sell", 0) and day >= 1:
                if ind["macd_hist"][si, day] < 0 and ind["macd_hist"][si, day-1] >= 0: sell = "MACD死叉"
            if not sell and p.get("use_kd_sell", 0) and day >= 1:
                if ind["k_val"][si, day] < ind["d_val"][si, day] and ind["k_val"][si, day-1] >= ind["d_val"][si, day-1]: sell = "KD死叉"
            svs = p.get("sell_vol_shrink", 0)
            if not sell and svs > 0 and dh >= 2 and ind["vol_ratio"][si, day] < svs: sell = "量縮"
            if not sell and dh >= p["hold_days"]: sell = "到期"

            if sell:
                trades.append({"si": si, "bd": holding["bd"], "sd": day, "bp": holding["bp"], "sp": float(cur), "ret": ret, "dh": dh, "reason": sell})
                holding = None
            continue

        best_si, best_v = -1, 0
        for si in range(ns):
            ok = True
            if p.get("use_rsi_buy", 1) and ind["rsi"][si, day] < p["rsi_buy"]: ok = False
            if ok and p.get("use_bb_buy", 1) and ind["bb_pos"][si, day] < p["bb_buy"]: ok = False
            if ok and p.get("use_vol_filter", 1) and ind["vol_ratio"][si, day] < p["vol_filter"]: ok = False
            if ok and p.get("require_ma_bull", 0) and ind["close"][si, day] < maf[si, day]: ok = False
            if ok and p.get("use_macd", 0):
                mm = p.get("macd_mode", 2)
                if mm == 0 and not (ind["macd_hist"][si, day] > 0 and ind["macd_hist"][si, day-1] <= 0): ok = False
                elif mm == 1 and ind["macd_line"][si, day] <= 0: ok = False
            if ok and p.get("use_kd", 0):
                if ind["k_val"][si, day] < p.get("kd_buy_k", 50): ok = False
                if ok and p.get("kd_cross", 0):
                    if not (ind["k_val"][si, day] > ind["d_val"][si, day] and ind["k_val"][si, day-1] <= ind["d_val"][si, day-1]): ok = False
            mm_val = p.get("momentum_min", 0)
            if ok and mm_val > 0 and mom[si, day] < mm_val: ok = False
            vid = p.get("vol_increase_days", 0)
            if ok and vid >= 2:
                for v in range(vid):
                    if day - v < 1 or ind["vol_ratio"][si, day-v] < 1.0: ok = False; break
            if ok and p.get("bb_width_min", 0) > 0 and ind["bb_width"][si, day] < p["bb_width_min"]: ok = False
            cg = p.get("consecutive_green", 0)
            if ok and cg >= 1:
                for g in range(cg):
                    if day - g < 0 or ind["is_green"][si, day-g] != 1: ok = False; break
            if ok and p.get("gap_up", 0) and ind["gap"][si, day] < 1.0: ok = False
            nhp = p.get("near_high_pct", 0)
            if ok and nhp > 0 and abs(ind["near_high"][si, day]) > nhp: ok = False
            if ok and p.get("above_ma60", 0) and ind["close"][si, day] < ma60[si, day]: ok = False
            if ok and p.get("require_ma_cross", 0) and maf[si, day] < mas[si, day]: ok = False
            if ok and p.get("vol_gt_yesterday", 0) and day >= 1 and ind["vol_ratio"][si, day] <= ind["vol_prev"][si, day]: ok = False
            if ok and ind["vol_ratio"][si, day] > best_v:
                best_si = si; best_v = ind["vol_ratio"][si, day]

        if best_si >= 0 and day + 1 < nd:
            holding = {"si": best_si, "bp": float(ind["close"][best_si, day+1]), "bd": day+1, "pk": float(ind["close"][best_si, day+1])}

    if len(trades) < 3: return None
    rets = np.array([t["ret"] for t in trades])
    avg_r = np.mean(rets)
    if avg_r < 4 or np.sum(rets > 0)/len(rets)*100 < 40: return None
    avg_hd = np.mean([t["dh"] for t in trades])
    if avg_hd > 15 or avg_hd < 1: return None

    w = rets[rets > 0]; l = rets[rets <= 0]
    wasted = np.sum(rets < 3) / len(rets) * 100
    pf = abs(np.sum(w) / np.sum(l)) if len(l) > 0 and np.sum(l) != 0 else 999

    score = np.sum(rets) * 0.25 + avg_r * 0.30 + (np.sum(rets>0)/len(rets)*100) * 0.15 + min(pf,5)*3*0.10 + np.max(rets)*0.10 - wasted*0.10

    return {"score": float(score), "params": p, "trades": trades,
            "avg_return": float(avg_r), "total_return": float(np.sum(rets)),
            "win_rate": float(np.sum(rets>0)/len(rets)*100), "max_return": float(np.max(rets)),
            "avg_hold": float(avg_hd), "n_trades": len(trades), "pf": float(pf)}

# === 參數空間 ===
PARAMS = {
    "use_rsi_buy": [0,1], "rsi_buy": [40,50,55,60,65,70,75],
    "use_bb_buy": [0,1], "bb_buy": [0.3,0.5,0.6,0.7,0.8,0.9,1.0,1.1],
    "use_vol_filter": [0,1], "vol_filter": [1.5,2.0,2.5,3.0,4.0,5.0],
    "require_ma_bull": [0,1], "ma_fast_w": [3,5,10], "ma_slow_w": [15,20,30,60],
    "use_macd": [0,1], "macd_mode": [0,1,2],
    "use_kd": [0,1], "kd_buy_k": [30,50,60,70,80], "kd_cross": [0,1],
    "momentum_days": [3,5,10], "momentum_min": [0,2,5,8,10],
    "vol_increase_days": [0,2,3], "bb_width_min": [0,0.03,0.05,0.08],
    "consecutive_green": [0,1,2,3], "gap_up": [0,1],
    "near_high_pct": [0,3,5,10], "above_ma60": [0,1],
    "require_ma_cross": [0,1], "vol_gt_yesterday": [0,1],
    "stop_loss": [-3,-5,-7,-10,-12,-15],
    "use_take_profit": [0,1], "take_profit": [10,15,20,30,40,50,60],
    "trailing_stop": [0,3,5,7,10],
    "use_rsi_sell": [0,1], "rsi_sell": [75,80,85,90,95],
    "use_macd_sell": [0,1], "use_kd_sell": [0,1],
    "sell_vol_shrink": [0,0.3,0.5,0.7], "hold_days": [3,5,7,10,15],
}

# === 主程式 ===
def main():
    job_id = os.environ.get("JOB_ID", "0")
    n_tests = int(os.environ.get("N_TESTS", "5000"))

    print(f"[Job {job_id}] 🚀 雲端進化引擎啟動 | {n_tests} 組")

    # 下載資料
    t0 = time.time()
    raw = download_data()
    data = filter_top_volume(raw, 50)
    print(f"[Job {job_id}] 資料下載：{len(data)} 檔 | {time.time()-t0:.1f}秒")

    if len(data) < 10:
        print("資料不足"); return

    pre = precompute(data)
    print(f"[Job {job_id}] 指標預算完成：{pre['n_stocks']}檔 x {pre['n_days']}天")

    # 載入歷史最佳（從環境變數或檔案）
    best_score = float(os.environ.get("BEST_SCORE", "-999999"))
    print(f"[Job {job_id}] 歷史最佳：{best_score:.2f}")

    # 隨機測試
    # 每台機器用不同 seed，且跟本地 Mac 的 seed 完全不重疊
    seed_offset = int(os.environ.get("SEED_OFFSET", "1000000"))
    np.random.seed((int(time.time()) + int(job_id) * 99991 + seed_offset) % 2**31)
    param_sets = []
    for _ in range(n_tests):
        p = {k: np.random.choice(v) for k, v in PARAMS.items()}
        p = {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for k, v in p.items()}
        if p.get("ma_fast_w", 5) >= p.get("ma_slow_w", 20): continue
        param_sets.append((p, pre))

    t1 = time.time()
    best = None
    improved = 0
    tested = 0

    workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(backtest_one, param_sets):
            tested += 1
            if r and r["score"] > best_score:
                best_score = r["score"]
                best = r
                improved += 1
                print(f"  [Job {job_id}] 新紀錄！{r['score']:.1f} | 勝率{r['win_rate']:.0f}% | 平均報酬{r['avg_return']:.1f}% | 總報酬{r['total_return']:.0f}%")

    elapsed = time.time() - t1
    speed = tested / elapsed if elapsed > 0 else 0
    print(f"[Job {job_id}] 完成 | {tested}組 | {elapsed:.1f}秒 | {speed:.0f}組/秒 | 突破{improved}次")

    if best:
        tickers = pre["tickers"]
        dates = pre["dates"]
        cl = pre["ind"]["close"]

        lines = []
        for t in sorted(best["trades"], key=lambda x: x["bd"]):
            si = t["si"]
            tk = tickers[si]
            lines.append(f"  {get_name(tk)}({tk.replace('.TW','')}) | {str(dates[t['bd']].date())[5:]}→{str(dates[t['sd']].date())[5:]} | {t['bp']:.1f}→{t['sp']:.1f} | {t['ret']:+.1f}% | {t['dh']}天 | {t['reason']}")

        msg = (
            f"☁️ 雲端進化突破！Job{job_id}\n"
            f"━━━━━━━━━━━━\n"
            f"平均報酬：{best['avg_return']:.1f}%\n"
            f"總報酬：{best['total_return']:.1f}%\n"
            f"勝率：{best['win_rate']:.0f}% | {best['n_trades']}筆\n"
            f"持有：{best['avg_hold']:.1f}天\n"
            f"⚡ {tested}組/{elapsed:.1f}秒/{speed:.0f}組/秒\n\n"
            f"📋 交易明細：\n" + "\n".join(lines)
        )
        telegram_push(msg)

        # 輸出結果給 GitHub Actions 收集
        with open(f"result_{job_id}.json", "w") as f:
            json.dump({"score": best_score, "job_id": job_id, "result": best}, f)

    print(f"::set-output name=score::{best_score}")
    print(f"::set-output name=improved::{improved}")

if __name__ == "__main__":
    main()
