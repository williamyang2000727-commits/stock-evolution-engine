#!/usr/bin/env python3
"""
雲端進化引擎 — 跑在 GitHub Actions 上
自己下載資料、回測、推播 Telegram
"""

import numpy as np
import numba as nb
import json
import os
import sys
import time
import requests
import yfinance as yf
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
    "2485.TW": "兆赫", "1711.TW": "永光", "1717.TW": "長興", "2313.TW": "華通",
    "6505.TW": "台塑化", "1303.TW": "南亞", "2406.TW": "國碩", "8150.TW": "南茂",
    "2615.TW": "萬海", "2618.TW": "長榮航", "2610.TW": "華航", "2912.TW": "統一超",
    "1101.TW": "台泥", "2880.TW": "華南金", "2885.TW": "元大金", "2890.TW": "永豐金",
    "2301.TW": "光寶科", "2408.TW": "南亞科", "2449.TW": "京元電子",
    "2345.TW": "智邦", "3443.TW": "創意", "2474.TW": "可成",
    "6239.TW": "力成", "3044.TW": "健鼎", "2379.TW": "瑞昱",
    "2395.TW": "研華", "5871.TW": "中租-KY", "1402.TW": "遠東新",
    "1590.TW": "亞德客-KY", "2327.TW": "國巨", "3008.TW": "大立光",
    "2801.TW": "彰銀", "2834.TW": "臺企銀", "2883.TW": "開發金",
    "2884.TW": "玉山金", "2887.TW": "台新金", "2892.TW": "第一金", "3189.TW": "景碩",
}

NAMES_CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_names.json")

def load_names_cache():
    """從快取檔讀取股票名稱"""
    if os.path.exists(NAMES_CACHE_PATH):
        try:
            with open(NAMES_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
                CN_NAMES.update(cached)
        except: pass

def save_names_cache():
    """儲存股票名稱快取"""
    try:
        os.makedirs(os.path.dirname(NAMES_CACHE_PATH), exist_ok=True)
        with open(NAMES_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(CN_NAMES, f, ensure_ascii=False, indent=2)
    except: pass

def auto_fetch_names(tickers):
    """自動從 yfinance 抓缺漏的中文名"""
    missing = [t for t in tickers if t not in CN_NAMES or CN_NAMES[t] == t.replace(".TW","")]
    if not missing: return
    for t in missing:
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName", "") or info.get("longName", "")
            if name and not name[0].isdigit():
                CN_NAMES[t] = name
        except: pass
    save_names_cache()

load_names_cache()

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

CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")
DATA_GIST_ID = "a300b9e29372ac76f79eda39a2a86321"

def download_data():
    import pickle, base64
    # 有快取且不到 24 小時就直接讀
    if os.path.exists(CACHE_PATH):
        age_hours = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age_hours < 24:
            try:
                with open(CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                if len(data) >= 10:
                    print(f"[快取] 讀取 {len(data)} 檔 | {age_hours:.1f} 小時前下載")
                    return data
            except:
                pass

    # 方法 1：yfinance 下載（加延遲避免 rate limit）
    data = {}
    for i, ticker in enumerate(TW_TICKERS):
        try:
            h = yf.Ticker(ticker).history(period="2y")
            if len(h) >= 40:
                data[ticker] = h
            if i % 5 == 4:
                time.sleep(1)
        except:
            continue

    # 方法 2：yfinance 失敗就從 Gist raw URL 下載資料快取
    if len(data) < 10:
        print("[yfinance 失敗] 從 Gist 下載資料快取...")
        try:
            gh_token = os.environ.get("GH_TOKEN", "")
            headers = {"Authorization": f"token {gh_token}"} if gh_token else {}
            # 先拿 raw_url（大檔案會 truncated，必須用 raw_url）
            r = requests.get(f"https://api.github.com/gists/{DATA_GIST_ID}",
                headers=headers, timeout=30)
            finfo = list(r.json()["files"].values())[0]
            raw_url = finfo.get("raw_url", "")
            if raw_url:
                r2 = requests.get(raw_url, headers=headers, timeout=60)
                raw = base64.b64decode(r2.text)
            else:
                raw = base64.b64decode(finfo["content"])
            data = pickle.loads(raw)
            print(f"[Gist 快取] 成功讀取 {len(data)} 檔")
        except Exception as e:
            print(f"[Gist 快取失敗] {e}")
            return {}

    # 存本地快取
    if len(data) >= 10:
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(data, f)
        except:
            pass
    return data

def filter_top_volume(data, top_n=50):
    vol_rank = {}
    for t, h in data.items():
        if "Volume" in h.columns and len(h) >= 20:
            vol_rank[t] = h["Volume"].tail(20).mean()
    top = sorted(vol_rank, key=vol_rank.get, reverse=True)[:top_n]
    # 自動抓缺漏的中文名
    auto_fetch_names(top)
    return {k: data[k] for k in top}

# === 預算指標 ===
def precompute(data):
    tickers = list(data.keys())
    min_len = min(len(data[t]) for t in tickers)
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

    # OBV
    obv = np.zeros_like(volume)
    obv[:, 0] = volume[:, 0]
    for i in range(1, close.shape[1]):
        obv[:, i] = np.where(close[:, i] > close[:, i-1], obv[:, i-1] + volume[:, i],
                    np.where(close[:, i] < close[:, i-1], obv[:, i-1] - volume[:, i], obv[:, i-1]))
    obv_ma = np.zeros_like(obv)
    for i in range(10, obv.shape[1]):
        obv_ma[:, i] = np.mean(obv[:, i-10:i], axis=1)
    ind["obv"] = obv; ind["obv_ma"] = obv_ma

    # 前三日低點
    low3 = np.zeros_like(low)
    for i in range(3, low.shape[1]):
        low3[:, i] = np.min(low[:, i-3:i], axis=1)
    ind["low3"] = low3

    # ATR (14)
    tr = np.zeros_like(close)
    tr[:, 1:] = np.maximum(high[:, 1:] - low[:, 1:], np.maximum(np.abs(high[:, 1:] - close[:, :-1]), np.abs(low[:, 1:] - close[:, :-1])))
    atr = np.zeros_like(close)
    for i in range(1, close.shape[1]):
        if i <= 14:
            atr[:, i] = np.mean(tr[:, 1:min(i+1,15)], axis=1)
        else:
            atr[:, i] = (atr[:, i-1] * 13 + tr[:, i]) / 14
    ind["atr"] = atr

    # Bias 乖離率
    for w in [5, 10, 20]:
        ma = ind.get(f"ma{w}", np.zeros_like(close))
        bias = np.where(ma > 0, (close - ma) / ma * 100, 0)
        ind[f"bias{w}"] = bias

    # Williams %R (14)
    low_14 = np.zeros_like(close); high_14 = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        low_14[:, i] = np.min(low[:, i-14:i+1], axis=1)
        high_14[:, i] = np.max(high[:, i-14:i+1], axis=1)
    ind["williams_r"] = np.where((high_14 - low_14) > 0, (high_14 - close) / (high_14 - low_14) * -100, -50)

    return {"tickers": tickers, "dates": dates, "n_stocks": n, "n_days": min_len, "ind": ind}

# === Numba 編譯的核心模擬（七大核心統一版）===
@nb.njit(cache=True)
def simulate_trading(n_stocks, n_days, close, rsi, bb_pos, vol_ratio,
    macd_line, macd_hist, k_val, d_val, momentum, is_green, gap, near_high,
    williams_r, ma_fast, ma_slow, ma60, bb_width, vol_prev,
    use_rsi_buy, rsi_buy, use_bb_buy, bb_buy_th, use_vol, vol_th,
    require_ma_bull, use_macd, macd_mode, use_kd, kd_k_th, kd_cross,
    use_wr_buy, wr_buy_th,
    mom_min, consec_green, use_gap, near_high_pct, above_ma60,
    require_ma_cross, vol_gt_yesterday,
    stop_loss, use_tp, take_profit, trailing_stop,
    use_rsi_sell, rsi_sell_th, use_macd_sell, use_kd_sell,
    sell_vol_shrink, sell_below_ma, hold_days_max):

    MAX_TRADES = 100
    trade_returns = np.zeros(MAX_TRADES)
    trade_stocks = np.zeros(MAX_TRADES, dtype=np.int64)
    trade_buy_days = np.zeros(MAX_TRADES, dtype=np.int64)
    trade_sell_days = np.zeros(MAX_TRADES, dtype=np.int64)
    trade_hold_days = np.zeros(MAX_TRADES, dtype=np.int64)
    trade_reasons = np.zeros(MAX_TRADES, dtype=np.int64)
    n_trades = 0
    holding = -1; buy_price = 0.0; peak_price = 0.0; buy_day = 0

    for day in range(30, n_days - 1):
        if holding >= 0:
            si = holding
            cur = close[si, day]
            dh = day - buy_day
            ret = (cur / buy_price - 1.0) * 100.0
            if dh < 1: continue
            if cur > peak_price: peak_price = cur
            sell = False; reason = 0
            if ret <= stop_loss: sell = True; reason = 2
            if not sell and use_tp == 1 and ret >= take_profit: sell = True; reason = 1
            if not sell and trailing_stop > 0 and peak_price > buy_price:
                if (cur / peak_price - 1.0) * 100.0 <= -trailing_stop: sell = True; reason = 4
            if not sell and use_rsi_sell == 1 and rsi[si, day] >= rsi_sell_th: sell = True; reason = 3
            if not sell and use_macd_sell == 1 and day >= 1:
                if macd_hist[si, day] < 0 and macd_hist[si, day-1] >= 0: sell = True; reason = 5
            if not sell and use_kd_sell == 1 and day >= 1:
                if k_val[si, day] < d_val[si, day] and k_val[si, day-1] >= d_val[si, day-1]: sell = True; reason = 6
            if not sell and sell_vol_shrink > 0 and dh >= 2 and vol_ratio[si, day] < sell_vol_shrink: sell = True; reason = 7
            if not sell and sell_below_ma > 0:
                ma_check = 0.0
                if sell_below_ma == 1: ma_check = ma_fast[si, day]
                elif sell_below_ma == 2: ma_check = ma_slow[si, day]
                elif sell_below_ma == 3: ma_check = ma60[si, day]
                if ma_check > 0 and cur < ma_check: sell = True; reason = 8
            if not sell and dh >= hold_days_max: sell = True; reason = 0
            if sell and n_trades < MAX_TRADES:
                trade_returns[n_trades] = ret
                trade_stocks[n_trades] = si
                trade_buy_days[n_trades] = buy_day
                trade_sell_days[n_trades] = day
                trade_hold_days[n_trades] = dh
                trade_reasons[n_trades] = reason
                n_trades += 1
                holding = -1
            continue

        best_si = -1; best_vol = 0.0
        for si in range(n_stocks):
            buy = True
            if buy and use_rsi_buy == 1 and rsi[si, day] < rsi_buy: buy = False
            if buy and use_bb_buy == 1 and bb_pos[si, day] < bb_buy_th: buy = False
            if buy and use_vol == 1 and vol_ratio[si, day] < vol_th: buy = False
            if buy and require_ma_bull == 1 and close[si, day] < ma_fast[si, day]: buy = False
            if buy and use_macd == 1:
                if macd_mode == 0 and not (macd_hist[si, day] > 0 and macd_hist[si, day-1] <= 0): buy = False
                elif macd_mode == 1 and macd_line[si, day] <= 0: buy = False
            if buy and use_kd == 1:
                if k_val[si, day] < kd_k_th: buy = False
                if buy and kd_cross == 1 and day >= 1:
                    if not (k_val[si, day] > d_val[si, day] and k_val[si, day-1] <= d_val[si, day-1]): buy = False
            if buy and use_wr_buy == 1 and williams_r[si, day] < wr_buy_th: buy = False
            if buy and mom_min > 0 and momentum[si, day] < mom_min: buy = False
            if buy and consec_green >= 1:
                for g in range(consec_green):
                    if day - g < 0 or is_green[si, day - g] != 1: buy = False; break
            if buy and use_gap == 1 and gap[si, day] < 1.0: buy = False
            if buy and near_high_pct > 0 and abs(near_high[si, day]) > near_high_pct: buy = False
            if buy and above_ma60 == 1 and close[si, day] < ma60[si, day]: buy = False
            if buy and require_ma_cross == 1 and ma_fast[si, day] < ma_slow[si, day]: buy = False
            if buy and vol_gt_yesterday == 1 and day >= 1 and vol_ratio[si, day] <= vol_prev[si, day]: buy = False
            if buy and vol_ratio[si, day] > best_vol:
                best_si = si; best_vol = vol_ratio[si, day]
        if best_si >= 0 and day + 1 < n_days:
            holding = best_si
            buy_price = close[best_si, day + 1]
            peak_price = buy_price
            buy_day = day + 1

    return n_trades, trade_returns, trade_stocks, trade_buy_days, trade_sell_days, trade_hold_days, trade_reasons

REASON_NAMES = ["到期", "停利", "停損", "RSI超買", "移動停利", "MACD死叉", "KD死叉", "量縮", "跌破均線"]

# === 回測（單組參數，用 Numba 快版）===
def backtest_one(args):
    p, pre = args
    ind = pre["ind"]
    ns, nd = pre["n_stocks"], pre["n_days"]
    mfw = p.get("ma_fast_w", 5); msw = p.get("ma_slow_w", 20)
    if mfw >= msw: return None
    maf = ind.get(f"ma{mfw}", ind["ma5"])
    mas = ind.get(f"ma{msw}", ind["ma20"])
    ma60 = ind.get("ma60", ind["ma20"])
    md = p.get("momentum_days", 5)
    mom = ind.get(f"mom_{md}", ind["mom_5"])

    n_trades, rets_arr, stocks, buy_days, sell_days, hold_days, reasons = simulate_trading(
        ns, nd, ind["close"], ind["rsi"], ind["bb_pos"], ind["vol_ratio"],
        ind["macd_line"], ind["macd_hist"], ind["k_val"], ind["d_val"],
        mom, ind["is_green"], ind["gap"], ind["near_high"],
        ind["williams_r"], maf, mas, ma60, ind["bb_width"], ind["vol_prev"],
        p.get("use_rsi_buy",1), p.get("rsi_buy",55),
        p.get("use_bb_buy",1), p.get("bb_buy",0.7),
        p.get("use_vol_filter",1), p.get("vol_filter",3.0),
        p.get("require_ma_bull",0), p.get("use_macd",0),
        p.get("macd_mode",2), p.get("use_kd",0),
        p.get("kd_buy_k",50), p.get("kd_cross",0),
        p.get("use_wr_buy",0), p.get("wr_buy",-30),
        p.get("momentum_min",0), p.get("consecutive_green",0),
        p.get("gap_up",0), p.get("near_high_pct",0),
        p.get("above_ma60",0), p.get("require_ma_cross",0),
        p.get("vol_gt_yesterday",0),
        p.get("stop_loss",-10), p.get("use_take_profit",1),
        p.get("take_profit",20), p.get("trailing_stop",0),
        p.get("use_rsi_sell",1), p.get("rsi_sell",90),
        p.get("use_macd_sell",0), p.get("use_kd_sell",0),
        p.get("sell_vol_shrink",0), p.get("sell_below_ma",0), p.get("hold_days",10))

    if n_trades < 10: return None
    rets = rets_arr[:n_trades]; bds = buy_days[:n_trades]
    avg_r = np.mean(rets)
    if avg_r < 6 or np.sum(rets > 0)/n_trades*100 < 50: return None
    avg_hd = np.mean(hold_days[:n_trades].astype(np.float64))
    if avg_hd > 15 or avg_hd < 1: return None

    mid = nd // 2
    first = rets[bds < mid]; second = rets[bds >= mid]
    if len(first) < 2 or len(second) < 2: return None
    if np.mean(first) < 0 or np.mean(second) < 0: return None
    consistency = min(np.mean(first), np.mean(second)) / max(np.mean(first), np.mean(second))

    w = rets[rets > 0]; l = rets[rets <= 0]
    wasted = np.sum(rets < 5) / n_trades * 100
    if wasted > 60: return None
    pf = abs(np.sum(w) / np.sum(l)) if len(l) > 0 and np.sum(l) != 0 else 999
    win_rate = np.sum(rets > 0) / n_trades * 100

    score = (np.sum(rets)*0.15 + avg_r*0.30 + win_rate*0.10 +
             min(pf,5)*3*0.05 + consistency*20*0.10 +
             n_trades*0.5*0.10 - wasted*0.20)

    # 組裝交易明細
    tickers = pre["tickers"]
    trades = []
    for j in range(n_trades):
        si = int(stocks[j])
        trades.append({"si": si, "bd": int(buy_days[j]), "sd": int(sell_days[j]),
            "bp": float(pre["ind"]["close"][si, int(buy_days[j])]),
            "sp": float(pre["ind"]["close"][si, int(sell_days[j])]),
            "ret": float(rets[j]), "dh": int(hold_days[j]),
            "reason": REASON_NAMES[int(reasons[j])]})

    return {"score": float(score), "params": p, "trades": trades,
            "avg_return": float(avg_r), "total_return": float(np.sum(rets)),
            "win_rate": float(win_rate), "max_return": float(np.max(rets)),
            "avg_hold": float(avg_hd), "n_trades": n_trades, "pf": float(pf)}

# === 參數空間 ===
PARAMS = {
    # ====== 七大核心（跟本地 backtest_turbo.py 100% 一致）======
    "use_rsi_buy": [0,1], "rsi_buy": [35,40,45,50,55,60,65,70,75],
    "use_bb_buy": [0,1], "bb_buy": [0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    "use_vol_filter": [0,1], "vol_filter": [1.5,2.0,2.5,3.0,4.0,5.0],
    "require_ma_bull": [0,1], "ma_fast_w": [3,5,10], "ma_slow_w": [15,20,30,60],
    "use_wr_buy": [0,1], "wr_buy": [-10,-20,-30,-40,-50],
    "use_macd": [0,1], "macd_mode": [0,1,2],
    "use_kd": [0,1], "kd_buy_k": [20,30,40,50,60,70,80], "kd_cross": [0,1],
    # 輔助
    "momentum_days": [3,5,10], "momentum_min": [0,3,8],
    "consecutive_green": [0,1,2,3], "gap_up": [0,1],
    "near_high_pct": [0,5,10], "above_ma60": [0,1],
    "require_ma_cross": [0,1], "vol_gt_yesterday": [0,1],
    # 賣出
    "stop_loss": [-5,-7,-10,-15],
    "use_take_profit": [0,1], "take_profit": [10,15,20,30,40,60],
    "trailing_stop": [0,3,5,7,10],
    "use_rsi_sell": [0,1], "rsi_sell": [75,80,85,90,95],
    "use_macd_sell": [0,1], "use_kd_sell": [0,1],
    "sell_vol_shrink": [0,0.3,0.5,0.7],
    "sell_below_ma": [0,1,2,3],  # 0=關 1=快線 2=慢線 3=MA60
    "hold_days": [5,7,10,15],
}

# === 主程式 ===
def main():
    job_id = os.environ.get("JOB_ID", "0")
    max_minutes = int(os.environ.get("MAX_MINUTES", "7"))  # 本地可設更長
    batch_size = int(os.environ.get("N_TESTS", "10000"))  # 每批組數

    print(f"[Job {job_id}] 🚀 雲端進化引擎啟動 | 跑滿 {max_minutes} 分鐘")
    start_time = time.time()

    # 下載資料（只下載一次）
    t0 = time.time()
    raw = download_data()
    data = filter_top_volume(raw, 50)
    print(f"[Job {job_id}] 資料下載：{len(data)} 檔 | {time.time()-t0:.1f}秒")

    if len(data) < 10:
        print("資料不足"); return

    pre = precompute(data)
    print(f"[Job {job_id}] 指標預算完成：{pre['n_stocks']}檔 x {pre['n_days']}天")

    # 載入歷史最佳
    best_score = float(os.environ.get("BEST_SCORE", "-999999"))
    print(f"[Job {job_id}] 歷史最佳：{best_score:.2f}")

    seed_offset = int(os.environ.get("SEED_OFFSET", "1000000"))
    workers = int(os.environ.get("N_WORKERS", str(max(1, os.cpu_count() - 1))))

    best = None
    improved = 0
    tested = 0
    round_num = 0

    # 無限循環跑到 7 分鐘為止
    while (time.time() - start_time) < max_minutes * 60:
        round_num += 1

        # 每輪用不同 seed
        np.random.seed((int(time.time() * 1000) + int(job_id) * 99991 + seed_offset + round_num * 77777) % 2**31)

        param_sets = []
        for _ in range(batch_size):
            p = {k: np.random.choice(v) for k, v in PARAMS.items()}
            p = {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for k, v in p.items()}
            if p.get("ma_fast_w", 5) >= p.get("ma_slow_w", 20): continue
            param_sets.append((p, pre))

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(backtest_one, param_sets):
                tested += 1
                if r and r["score"] > best_score:
                    best_score = r["score"]
                    best = r
                    improved += 1
                    print(f"  [Job {job_id}] R{round_num} 新紀錄！{r['score']:.1f} | 勝率{r['win_rate']:.0f}% | 平均報酬{r['avg_return']:.1f}%")

        elapsed_so_far = time.time() - start_time
        speed = tested / elapsed_so_far if elapsed_so_far > 0 else 0
        print(f"[Job {job_id}] R{round_num} | 累計{tested}組 | {elapsed_so_far:.0f}秒 | {speed:.0f}組/秒")

    elapsed = time.time() - start_time
    speed = tested / elapsed if elapsed > 0 else 0
    print(f"[Job {job_id}] 結束 | {round_num}輪 | {tested}組 | {elapsed:.0f}秒 | {speed:.0f}組/秒 | 突破{improved}次")

    if best:
        tickers = pre["tickers"]
        dates = pre["dates"]
        cl = pre["ind"]["close"]

        lines = []
        for t in sorted(best["trades"], key=lambda x: x["bd"]):
            si = t["si"]
            tk = tickers[si]
            lines.append(f"  {get_name(tk)}({tk.replace('.TW','')}) | {str(dates[t['bd']].date())[5:]}→{str(dates[t['sd']].date())[5:]} | {t['bp']:.1f}→{t['sp']:.1f} | {t['ret']:+.1f}% | {t['dh']}天 | {t['reason']}")

        print(f"[Job {job_id}] 找到候選策略，嘗試同步...")

        # 同步到 GitHub Gist（中央資料庫）
        gist_id = os.environ.get("GIST_ID", "")
        gh_token = os.environ.get("GH_TOKEN", "")

        # 無 GH_TOKEN 模式（第二台 Mac）：讀公開 Gist 比分數，超過就推 Telegram
        if gist_id and not gh_token:
            try:
                r = requests.get(f"https://api.github.com/gists/{gist_id}", timeout=10)
                gist_data = r.json()
                current_gist_score = json.loads(list(gist_data["files"].values())[0]["content"]).get("score", 0)
                if best_score > current_gist_score:
                    telegram_push(
                        f"🖥️ 第二台 Mac 突破！\n"
                        f"分數 {best_score:.2f} > Gist {current_gist_score:.2f}\n"
                        f"勝率 {best['win_rate']:.1f}% | 平均報酬 {best['avg_return']:.1f}%\n"
                        f"總報酬 {best['total_return']:.0f}% | {best['n_trades']}筆\n"
                        f"⚠️ 無 GH_TOKEN，請手動同步或設定 token"
                    )
                    # 存本地檔案備查
                    with open(os.path.expanduser("~/stock-evolution/best_found.json"), "w") as f:
                        json.dump({"score": best_score, "params": best["params"],
                            "backtest": {"avg_return": round(best["avg_return"], 2),
                                "total_return": round(best["total_return"], 2),
                                "win_rate": round(best["win_rate"], 2),
                                "total_trades": best["n_trades"]}}, f, indent=2)
                    print(f"[Job {job_id}] ✅ 超越 Gist！已推 Telegram")
                else:
                    print(f"[Job {job_id}] Gist 分數 {current_gist_score:.2f} 更高，不更新")
            except Exception as e:
                print(f"[Job {job_id}] Gist 讀取失敗: {e}")

        if gist_id and gh_token:
            try:
                # 先讀目前 Gist 的分數
                r = requests.get(f"https://api.github.com/gists/{gist_id}",
                    headers={"Authorization": f"token {gh_token}"}, timeout=10)
                gist_data = r.json()
                current_content = json.loads(list(gist_data["files"].values())[0]["content"])
                current_gist_score = current_content.get("score", 0)

                # 只有比 Gist 裡的更高才更新
                if best_score > current_gist_score:
                    # 組裝完整策略
                    trade_details = []
                    for t in sorted(best["trades"], key=lambda x: x["bd"]):
                        si = t["si"]
                        tk = tickers[si]
                        trade_details.append({
                            "ticker": tk, "name": get_name(tk),
                            "buy_date": str(dates[t["bd"]].date()),
                            "sell_date": str(dates[t["sd"]].date()),
                            "buy_price": round(t["bp"], 2),
                            "sell_price": round(t["sp"], 2),
                            "return": round(t["ret"], 2),
                            "days": t["dh"], "reason": t["reason"],
                        })

                    gist_content = json.dumps({
                        "score": round(best_score, 4),
                        "source": f"cloud_job_{job_id}",
                        "updated_at": datetime.now().isoformat(),
                        "params": best["params"],
                        "backtest": {
                            "avg_return": round(best["avg_return"], 2),
                            "total_return": round(best["total_return"], 2),
                            "win_rate": round(best["win_rate"], 2),
                            "max_return": round(best["max_return"], 2),
                            "avg_hold_days": round(best["avg_hold"], 2),
                            "total_trades": best["n_trades"],
                            "profit_factor": round(best["pf"], 2),
                        },
                        "trade_details": trade_details,
                    }, ensure_ascii=False, indent=2)

                    requests.patch(
                        f"https://api.github.com/gists/{gist_id}",
                        headers={"Authorization": f"token {gh_token}"},
                        json={"files": {"best_strategy.json": {"content": gist_content}}},
                        timeout=10
                    )
                    print(f"[Job {job_id}] ✅ 已同步到 Gist（分數 {best_score:.2f} > {current_gist_score:.2f}）")

                    # 推播完整策略 + 交易明細
                    trade_lines = "\n".join([
                        f"  {t['name']}({t['ticker'].replace('.TW','')}) | {t['buy_date'][5:]}→{t['sell_date'][5:]} | {t['return']:+.1f}% | {t['days']}天 | {t['reason']}"
                        for t in trade_details
                    ])
                    telegram_push(
                        f"🚀 策略突破！（Job {job_id}）\n"
                        f"━━━━━━━━━━━━\n"
                        f"分數：{best_score:.2f} > {current_gist_score:.2f}\n"
                        f"平均報酬：{best['avg_return']:.1f}%\n"
                        f"總報酬：{best['total_return']:.0f}%\n"
                        f"勝率：{best['win_rate']:.0f}% | {best['n_trades']}筆\n"
                        f"⚡ {tested}組/{elapsed:.0f}秒/{speed:.0f}組/秒\n\n"
                        f"📋 交易明細：\n{trade_lines}"
                    )
                else:
                    print(f"[Job {job_id}] Gist 分數 {current_gist_score:.2f} 更高，不更新")
            except Exception as e:
                print(f"[Job {job_id}] Gist 同步失敗: {e}")

if __name__ == "__main__":
    main()
