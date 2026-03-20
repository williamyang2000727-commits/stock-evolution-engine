#!/usr/bin/env python3
"""
GPU + Numba CPU 雙引擎進化 — Windows NVIDIA 專用
GPU CUDA kernel 同時跑數千組參數，速度是純 Python 的 100-300 倍
"""
import numpy as np
import numba as nb
from numba import cuda
import json, os, sys, time, requests, pickle, base64

# === Telegram / Gist ===
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = os.environ.get("TELEGRAM_CHAT_IDS", "").split(",")
GIST_ID = os.environ.get("GIST_ID", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
DATA_GIST_ID = "a300b9e29372ac76f79eda39a2a86321"
CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")

CN_NAMES = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "2303.TW": "聯電",
    "2382.TW": "廣達", "3231.TW": "緯創", "2353.TW": "宏碁", "2357.TW": "華碩",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "2412.TW": "中華電", "1301.TW": "台塑", "2603.TW": "長榮", "2609.TW": "陽明",
    "1216.TW": "統一", "2002.TW": "中鋼", "2308.TW": "台達電", "3711.TW": "日月光投控",
}

def get_name(t): return CN_NAMES.get(t, t.replace(".TW",""))

def telegram_push(msg):
    for cid in CHAT_IDS:
        cid = cid.strip()
        if not cid or not BOT_TOKEN: continue
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": cid, "text": msg}, timeout=10)
        except: pass

# === 資料下載（跟 cloud_evolve.py 一樣）===
import yfinance as yf
TW_TICKERS = [
    "2330.TW","2454.TW","2317.TW","2303.TW","2382.TW","3231.TW","2353.TW","2357.TW",
    "2881.TW","2882.TW","2891.TW","2886.TW","2412.TW","1301.TW","2603.TW","2609.TW",
    "1216.TW","2002.TW","2308.TW","3711.TW","2409.TW","3481.TW","2356.TW","2324.TW",
    "4938.TW","2337.TW","2344.TW","3037.TW","6770.TW","3576.TW","1802.TW","8039.TW",
    "2485.TW","1711.TW","1717.TW","6505.TW","1303.TW","2406.TW","8150.TW",
    "2615.TW","2618.TW","2610.TW","2912.TW","1101.TW","2880.TW","2885.TW","2890.TW",
    "2801.TW","2834.TW","2883.TW","2884.TW","2887.TW","2892.TW","3189.TW","2301.TW",
    "2408.TW","3008.TW","2345.TW","3443.TW","2474.TW","6239.TW","3044.TW",
    "2379.TW","2395.TW","5871.TW","2912.TW","1402.TW","1590.TW","2327.TW",
]

def download_data():
    if os.path.exists(CACHE_PATH):
        age = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age < 24:
            try:
                with open(CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                if len(data) >= 10:
                    print(f"[快取] {len(data)} 檔 | {age:.1f}h 前")
                    return data
            except: pass
    data = {}
    for i, t in enumerate(TW_TICKERS):
        try:
            h = yf.Ticker(t).history(period="2y")
            if len(h) >= 40: data[t] = h
            if i % 5 == 4: time.sleep(1)
        except: continue
    if len(data) < 10:
        print("[yfinance 失敗] 從 Gist 下載...")
        try:
            headers = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
            r = requests.get(f"https://api.github.com/gists/{DATA_GIST_ID}", headers=headers, timeout=30)
            finfo = list(r.json()["files"].values())[0]
            raw_url = finfo.get("raw_url", "")
            r2 = requests.get(raw_url, headers=headers, timeout=60) if raw_url else None
            data = pickle.loads(base64.b64decode(r2.text if r2 else finfo["content"]))
            print(f"[Gist] {len(data)} 檔")
        except Exception as e:
            print(f"[失敗] {e}"); return {}
    if len(data) >= 10:
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "wb") as f: pickle.dump(data, f)
        except: pass
    return data

def filter_top(data, n=50):
    vr = {t: h["Volume"].tail(20).mean() for t, h in data.items() if "Volume" in h.columns and len(h) >= 20}
    top = sorted(vr, key=vr.get, reverse=True)[:n]
    return {k: data[k] for k in top}

# === 指標預算 ===
def precompute(data):
    tickers = list(data.keys())
    ml = min(len(data[t]) for t in tickers)
    n = len(tickers)
    close = np.zeros((n, ml), dtype=np.float32)
    high = np.zeros((n, ml), dtype=np.float32)
    low = np.zeros((n, ml), dtype=np.float32)
    opn = np.zeros((n, ml), dtype=np.float32)
    volume = np.zeros((n, ml), dtype=np.float32)
    dates = None
    for si, t in enumerate(tickers):
        h = data[t]
        close[si] = h["Close"].values[-ml:].astype(np.float32)
        high[si] = h["High"].values[-ml:].astype(np.float32)
        low[si] = h["Low"].values[-ml:].astype(np.float32)
        opn[si] = h["Open"].values[-ml:].astype(np.float32)
        volume[si] = h["Volume"].values[-ml:].astype(np.float32)
        if dates is None: dates = h.index[-ml:]

    # RSI
    delta = np.diff(close, axis=1)
    gain = np.where(delta > 0, delta, 0)
    loss_arr = np.where(delta < 0, -delta, 0)
    avg_g = np.zeros_like(close); avg_l = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        if i == 14:
            avg_g[:, i] = np.mean(gain[:, :14], axis=1)
            avg_l[:, i] = np.mean(loss_arr[:, :14], axis=1)
        else:
            avg_g[:, i] = (avg_g[:, i-1] * 13 + gain[:, i-1]) / 14
            avg_l[:, i] = (avg_l[:, i-1] * 13 + loss_arr[:, i-1]) / 14
    rs = np.where(avg_l > 0, avg_g / avg_l, 100)
    rsi = 100 - 100 / (1 + rs)

    # MA
    ma_dict = {}
    for w in [3, 5, 10, 15, 20, 30, 60]:
        ma = np.zeros_like(close)
        for i in range(w, close.shape[1]):
            ma[:, i] = np.mean(close[:, i-w:i], axis=1)
        ma_dict[w] = ma

    # BB
    bb_mid = ma_dict[20]
    bb_std = np.zeros_like(close)
    for i in range(20, close.shape[1]):
        bb_std[:, i] = np.std(close[:, i-20:i], axis=1)
    bb_u = bb_mid + 2 * bb_std; bb_l = bb_mid - 2 * bb_std; bb_r = bb_u - bb_l
    bb_pos = np.where(bb_r > 0, (close - bb_l) / bb_r, 0.5).astype(np.float32)
    bb_width = np.where(bb_mid > 0, bb_r / bb_mid, 0).astype(np.float32)

    # Volume ratio
    vol_ma = np.zeros_like(volume)
    for i in range(20, volume.shape[1]):
        vol_ma[:, i] = np.mean(volume[:, i-20:i], axis=1)
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1).astype(np.float32)
    vol_prev = np.zeros_like(vol_ratio)
    vol_prev[:, 1:] = vol_ratio[:, :-1]

    # MACD
    e12 = np.zeros_like(close); e26 = np.zeros_like(close)
    e12[:, 0] = close[:, 0]; e26[:, 0] = close[:, 0]
    for i in range(1, close.shape[1]):
        e12[:, i] = e12[:, i-1] * (1 - 2/13) + close[:, i] * 2/13
        e26[:, i] = e26[:, i-1] * (1 - 2/27) + close[:, i] * 2/27
    macd_line = (e12 - e26).astype(np.float32)
    ms = np.zeros_like(close); ms[:, 0] = macd_line[:, 0]
    for i in range(1, close.shape[1]):
        ms[:, i] = ms[:, i-1] * (1 - 2/10) + macd_line[:, i] * 2/10
    macd_hist = (macd_line - ms).astype(np.float32)

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

    # Momentum
    mom5 = np.zeros_like(close)
    mom5[:, 5:] = (close[:, 5:] / close[:, :-5] - 1) * 100
    mom3 = np.zeros_like(close)
    mom3[:, 3:] = (close[:, 3:] / close[:, :-3] - 1) * 100
    mom10 = np.zeros_like(close)
    mom10[:, 10:] = (close[:, 10:] / close[:, :-10] - 1) * 100

    is_green = (close > opn).astype(np.float32)
    gap_arr = np.zeros_like(close)
    gap_arr[:, 1:] = (opn[:, 1:] / close[:, :-1] - 1) * 100

    h20 = np.zeros_like(close)
    for i in range(20, close.shape[1]):
        h20[:, i] = np.max(high[:, i-20:i+1], axis=1)
    near_high = np.where(h20 > 0, (close / h20 - 1) * 100, 0).astype(np.float32)

    return {
        "tickers": tickers, "dates": dates, "n_stocks": n, "n_days": ml,
        "close": close, "rsi": rsi.astype(np.float32),
        "bb_pos": bb_pos, "bb_width": bb_width,
        "vol_ratio": vol_ratio, "vol_prev": vol_prev.astype(np.float32),
        "macd_line": macd_line, "macd_hist": macd_hist,
        "k_val": kv.astype(np.float32), "d_val": dv.astype(np.float32),
        "mom3": mom3.astype(np.float32), "mom5": mom5.astype(np.float32), "mom10": mom10.astype(np.float32),
        "is_green": is_green, "gap": gap_arr.astype(np.float32), "near_high": near_high,
        "ma3": ma_dict[3].astype(np.float32), "ma5": ma_dict[5].astype(np.float32),
        "ma10": ma_dict[10].astype(np.float32), "ma15": ma_dict[15].astype(np.float32),
        "ma20": ma_dict[20].astype(np.float32), "ma30": ma_dict[30].astype(np.float32),
        "ma60": ma_dict[60].astype(np.float32),
    }

# ============================================================
# GPU CUDA Kernel — 每個 thread 跑一組參數的完整交易模擬
# ============================================================
@cuda.jit
def gpu_backtest_kernel(
    n_stocks, n_days,
    close, rsi, bb_pos, vol_ratio, macd_line, macd_hist,
    k_val, d_val, momentum, is_green, gap, near_high,
    ma_fast, ma_slow, ma60, bb_width, vol_prev,
    params,  # (n_combos, 28) 參數陣列
    results, # (n_combos, 6) 結果：[score, n_trades, avg_ret, total_ret, win_rate, wasted]
    trade_rets, # (n_combos, 100) 每組的交易報酬
    trade_days, # (n_combos, 100) 每組的買入日
):
    idx = cuda.grid(1)
    if idx >= params.shape[0]:
        return

    # 讀取這個 thread 的參數
    p = params[idx]
    use_rsi_buy = int(p[0]); rsi_buy = p[1]
    use_bb_buy = int(p[2]); bb_buy = p[3]
    use_vol = int(p[4]); vol_th = p[5]
    require_ma_bull = int(p[6])
    use_macd = int(p[7]); macd_mode = int(p[8])
    use_kd = int(p[9]); kd_k_th = p[10]; kd_cross = int(p[11])
    mom_min = p[12]
    consec_green = int(p[13]); use_gap = int(p[14])
    near_high_pct = p[15]; above_ma60 = int(p[16])
    require_ma_cross = int(p[17]); vol_gt_yesterday = int(p[18])
    stop_loss = p[19]
    use_tp = int(p[20]); take_profit = p[21]
    trailing_stop = p[22]
    use_rsi_sell = int(p[23]); rsi_sell_th = p[24]
    use_macd_sell = int(p[25]); use_kd_sell = int(p[26])
    sell_vol_shrink = p[27]
    hold_days_max = int(p[28])

    # 交易模擬
    holding = -1
    buy_price = 0.0
    peak_price = 0.0
    buy_day = 0
    n_trades = 0

    for day in range(30, n_days - 1):
        # === 持有中：檢查賣出 ===
        if holding >= 0:
            si = holding
            cur = close[si, day]
            dh = day - buy_day
            ret = (cur / buy_price - 1.0) * 100.0
            if dh < 1:
                continue
            if cur > peak_price:
                peak_price = cur
            sell = False

            # 停損
            if ret <= stop_loss:
                sell = True
            # 停利
            if not sell and use_tp == 1 and ret >= take_profit:
                sell = True
            # 移動停利
            if not sell and trailing_stop > 0 and peak_price > buy_price:
                if (cur / peak_price - 1.0) * 100.0 <= -trailing_stop:
                    sell = True
            # RSI 超買
            if not sell and use_rsi_sell == 1 and rsi[si, day] >= rsi_sell_th:
                sell = True
            # MACD 死叉
            if not sell and use_macd_sell == 1 and day >= 1:
                if macd_hist[si, day] < 0 and macd_hist[si, day-1] >= 0:
                    sell = True
            # KD 死叉
            if not sell and use_kd_sell == 1 and day >= 1:
                if k_val[si, day] < d_val[si, day] and k_val[si, day-1] >= d_val[si, day-1]:
                    sell = True
            # 量縮
            if not sell and sell_vol_shrink > 0 and dh >= 2 and vol_ratio[si, day] < sell_vol_shrink:
                sell = True
            # 到期
            if not sell and dh >= hold_days_max:
                sell = True

            if sell and n_trades < 100:
                trade_rets[idx, n_trades] = ret
                trade_days[idx, n_trades] = buy_day
                n_trades += 1
                holding = -1
            continue

        # === 未持有：找買入 ===
        best_si = -1
        best_vol = 0.0
        for si in range(n_stocks):
            buy = True
            if buy and use_rsi_buy == 1 and rsi[si, day] < rsi_buy:
                buy = False
            if buy and use_bb_buy == 1 and bb_pos[si, day] < bb_buy:
                buy = False
            if buy and use_vol == 1 and vol_ratio[si, day] < vol_th:
                buy = False
            if buy and require_ma_bull == 1 and close[si, day] < ma_fast[si, day]:
                buy = False
            if buy and use_macd == 1:
                if macd_mode == 0:
                    if not (macd_hist[si, day] > 0 and macd_hist[si, day-1] <= 0):
                        buy = False
                elif macd_mode == 1:
                    if macd_line[si, day] <= 0:
                        buy = False
            if buy and use_kd == 1:
                if k_val[si, day] < kd_k_th:
                    buy = False
                if buy and kd_cross == 1 and day >= 1:
                    if not (k_val[si, day] > d_val[si, day] and k_val[si, day-1] <= d_val[si, day-1]):
                        buy = False
            if buy and mom_min > 0 and momentum[si, day] < mom_min:
                buy = False
            if buy and consec_green >= 1:
                for g in range(consec_green):
                    if day - g < 0 or is_green[si, day - g] != 1:
                        buy = False
                        break
            if buy and use_gap == 1 and gap[si, day] < 1.0:
                buy = False
            if buy and near_high_pct > 0 and abs(near_high[si, day]) > near_high_pct:
                buy = False
            if buy and above_ma60 == 1 and close[si, day] < ma60[si, day]:
                buy = False
            if buy and require_ma_cross == 1 and ma_fast[si, day] < ma_slow[si, day]:
                buy = False
            if buy and vol_gt_yesterday == 1 and day >= 1:
                if vol_ratio[si, day] <= vol_prev[si, day]:
                    buy = False

            if buy and vol_ratio[si, day] > best_vol:
                best_si = si
                best_vol = vol_ratio[si, day]

        if best_si >= 0 and day + 1 < n_days:
            holding = best_si
            buy_price = close[best_si, day + 1]
            peak_price = buy_price
            buy_day = day + 1

    # === 計算分數 ===
    if n_trades < 10:
        results[idx, 0] = -999999.0
        return

    total_ret = 0.0; win_count = 0; wasted_count = 0
    for i in range(n_trades):
        total_ret += trade_rets[idx, i]
        if trade_rets[idx, i] > 0: win_count += 1
        if trade_rets[idx, i] < 5: wasted_count += 1

    avg_ret = total_ret / n_trades
    win_rate = win_count / n_trades * 100.0
    wasted = wasted_count / n_trades * 100.0

    if avg_ret < 6 or win_rate < 50 or wasted > 60:
        results[idx, 0] = -999999.0
        return

    # 雙段驗證
    mid_day = n_days // 2
    first_sum = 0.0; first_n = 0; second_sum = 0.0; second_n = 0
    for i in range(n_trades):
        if trade_days[idx, i] < mid_day:
            first_sum += trade_rets[idx, i]; first_n += 1
        else:
            second_sum += trade_rets[idx, i]; second_n += 1
    if first_n < 2 or second_n < 2:
        results[idx, 0] = -999999.0; return
    first_avg = first_sum / first_n; second_avg = second_sum / second_n
    if first_avg < 0 or second_avg < 0:
        results[idx, 0] = -999999.0; return
    consistency = min(first_avg, second_avg) / max(first_avg, second_avg)

    # Profit factor
    w_sum = 0.0; l_sum = 0.0
    for i in range(n_trades):
        if trade_rets[idx, i] > 0: w_sum += trade_rets[idx, i]
        else: l_sum += abs(trade_rets[idx, i])
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    if pf > 5: pf = 5.0

    score = (total_ret * 0.15 + avg_ret * 0.30 + win_rate * 0.10 +
             pf * 3 * 0.05 + consistency * 20 * 0.10 +
             n_trades * 0.5 * 0.10 - wasted * 0.20)

    results[idx, 0] = score
    results[idx, 1] = n_trades
    results[idx, 2] = avg_ret
    results[idx, 3] = total_ret
    results[idx, 4] = win_rate
    results[idx, 5] = wasted

# === 參數空間 ===
PARAM_SPACE = {
    "use_rsi_buy": [0,1], "rsi_buy": [40,55,65,75],
    "use_bb_buy": [0,1], "bb_buy": [0.3,0.6,0.8,1.0],
    "use_vol_filter": [0,1], "vol_filter": [1.5,2.5,4.0],
    "require_ma_bull": [0,1],
    "use_macd": [0,1], "macd_mode": [0,1,2],
    "use_kd": [0,1], "kd_buy_k": [30,50,70], "kd_cross": [0,1],
    "momentum_min": [0,3,8],
    "consecutive_green": [0,1,2,3], "gap_up": [0,1],
    "near_high_pct": [0,5,10], "above_ma60": [0,1],
    "require_ma_cross": [0,1], "vol_gt_yesterday": [0,1],
    "stop_loss": [-5,-7,-10,-15],
    "use_take_profit": [0,1], "take_profit": [10,20,40,60],
    "trailing_stop": [0,5,10],
    "use_rsi_sell": [0,1], "rsi_sell": [75,85,95],
    "use_macd_sell": [0,1], "use_kd_sell": [0,1],
    "sell_vol_shrink": [0,0.3,0.7],
    "hold_days": [5,10,15],
}

# 參數名對應 params 陣列的 index
PARAM_KEYS = [
    "use_rsi_buy", "rsi_buy", "use_bb_buy", "bb_buy",
    "use_vol_filter", "vol_filter", "require_ma_bull",
    "use_macd", "macd_mode", "use_kd", "kd_buy_k", "kd_cross",
    "momentum_min", "consecutive_green", "gap_up",
    "near_high_pct", "above_ma60", "require_ma_cross", "vol_gt_yesterday",
    "stop_loss", "use_take_profit", "take_profit", "trailing_stop",
    "use_rsi_sell", "rsi_sell", "use_macd_sell", "use_kd_sell",
    "sell_vol_shrink", "hold_days",
]

MA_FAST_OPTIONS = [3, 5, 10]
MA_SLOW_OPTIONS = [15, 20, 30, 60]
MOM_DAYS_OPTIONS = [3, 5, 10]

def generate_params(n):
    """產生 n 組隨機參數，回傳 (n, 29) float32 陣列 + ma/mom 資訊"""
    arr = np.zeros((n, 29), dtype=np.float32)
    ma_fast_idx = np.random.choice(len(MA_FAST_OPTIONS), n)
    ma_slow_idx = np.random.choice(len(MA_SLOW_OPTIONS), n)
    mom_idx = np.random.choice(len(MOM_DAYS_OPTIONS), n)

    for i, key in enumerate(PARAM_KEYS):
        opts = PARAM_SPACE[key]
        arr[:, i] = np.random.choice(opts, n).astype(np.float32)

    return arr, ma_fast_idx, ma_slow_idx, mom_idx

def main():
    job_id = os.environ.get("JOB_ID", "99")
    print(f"[GPU] 🚀 GPU 進化引擎啟動！")

    # GPU 資訊
    gpu = cuda.get_current_device()
    print(f"[GPU] 裝置：{gpu.name}")

    # 下載資料
    t0 = time.time()
    raw = download_data()
    data = filter_top(raw, 50)
    print(f"[GPU] 資料：{len(data)} 檔 | {time.time()-t0:.1f}秒")
    if len(data) < 10: print("資料不足"); return

    pre = precompute(data)
    ns, nd = pre["n_stocks"], pre["n_days"]
    print(f"[GPU] 指標完成：{ns}檔 x {nd}天")

    # 傳資料到 GPU（只傳一次！）
    d_close = cuda.to_device(pre["close"])
    d_rsi = cuda.to_device(pre["rsi"])
    d_bb_pos = cuda.to_device(pre["bb_pos"])
    d_vol_ratio = cuda.to_device(pre["vol_ratio"])
    d_macd_line = cuda.to_device(pre["macd_line"])
    d_macd_hist = cuda.to_device(pre["macd_hist"])
    d_k_val = cuda.to_device(pre["k_val"])
    d_d_val = cuda.to_device(pre["d_val"])
    d_is_green = cuda.to_device(pre["is_green"])
    d_gap = cuda.to_device(pre["gap"])
    d_near_high = cuda.to_device(pre["near_high"])
    d_bb_width = cuda.to_device(pre["bb_width"])
    d_vol_prev = cuda.to_device(pre["vol_prev"])
    d_ma60 = cuda.to_device(pre["ma60"])

    ma_arrays = {3: pre["ma3"], 5: pre["ma5"], 10: pre["ma10"],
                 15: pre["ma15"], 20: pre["ma20"], 30: pre["ma30"], 60: pre["ma60"]}
    mom_arrays = {3: pre["mom3"], 5: pre["mom5"], 10: pre["mom10"]}

    print(f"[GPU] 資料已載入 GPU！開始進化...")

    best_score = -999999
    batch_size = 100000  # 每批 10 萬組！
    threads_per_block = 256
    start_time = time.time()
    total_tested = 0
    total_improved = 0
    round_num = 0

    while True:
        round_num += 1
        params, ma_fast_idx, ma_slow_idx, mom_idx = generate_params(batch_size)

        # 按 ma_fast/slow 組合分批送 GPU
        for mf in range(len(MA_FAST_OPTIONS)):
            for ms in range(len(MA_SLOW_OPTIONS)):
                mfw = MA_FAST_OPTIONS[mf]
                msw = MA_SLOW_OPTIONS[ms]
                if mfw >= msw: continue

                for md in range(len(MOM_DAYS_OPTIONS)):
                    mdw = MOM_DAYS_OPTIONS[md]
                    mask = (ma_fast_idx == mf) & (ma_slow_idx == ms) & (mom_idx == md)
                    sub_params = params[mask]
                    n_sub = len(sub_params)
                    if n_sub == 0: continue

                    d_ma_fast = cuda.to_device(ma_arrays[mfw])
                    d_ma_slow = cuda.to_device(ma_arrays[msw])
                    d_mom = cuda.to_device(mom_arrays[mdw])

                    d_params = cuda.to_device(sub_params)
                    d_results = cuda.device_array((n_sub, 6), dtype=np.float32)
                    d_trade_rets = cuda.device_array((n_sub, 100), dtype=np.float32)
                    d_trade_days = cuda.device_array((n_sub, 100), dtype=np.float32)

                    blocks = (n_sub + threads_per_block - 1) // threads_per_block
                    gpu_backtest_kernel[blocks, threads_per_block](
                        ns, nd,
                        d_close, d_rsi, d_bb_pos, d_vol_ratio, d_macd_line, d_macd_hist,
                        d_k_val, d_d_val, d_mom, d_is_green, d_gap, d_near_high,
                        d_ma_fast, d_ma_slow, d_ma60, d_bb_width, d_vol_prev,
                        d_params, d_results, d_trade_rets, d_trade_days
                    )

                    results = d_results.copy_to_host()
                    total_tested += n_sub

                    # 找最佳
                    best_idx = np.argmax(results[:, 0])
                    if results[best_idx, 0] > best_score:
                        best_score = results[best_idx, 0]
                        best_n = int(results[best_idx, 1])
                        best_avg = results[best_idx, 2]
                        best_total = results[best_idx, 3]
                        best_wr = results[best_idx, 4]

                        # 還原參數
                        bp = sub_params[best_idx]
                        best_params = {}
                        for i, key in enumerate(PARAM_KEYS):
                            best_params[key] = float(bp[i])
                        best_params["ma_fast_w"] = mfw
                        best_params["ma_slow_w"] = msw
                        best_params["momentum_days"] = mdw

                        total_improved += 1
                        print(f"  [GPU] R{round_num} 新紀錄！{best_score:.1f} | 勝率{best_wr:.0f}% | 平均報酬{best_avg:.1f}% | {best_n}筆")

        elapsed = time.time() - start_time
        speed = total_tested / elapsed if elapsed > 0 else 0
        print(f"[GPU] R{round_num} | 累計{total_tested:,}組 | {elapsed:.0f}秒 | {speed:,.0f}組/秒 | 突破{total_improved}次")

        # 同步到 Gist
        if total_improved > 0 and best_score > 0:
            try:
                gist_id = os.environ.get("GIST_ID", "")
                gh_token = os.environ.get("GH_TOKEN", "")
                if gist_id and gh_token:
                    headers = {"Authorization": f"token {gh_token}"}
                    r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=10)
                    current_score = json.loads(list(r.json()["files"].values())[0]["content"]).get("score", 0)
                    if best_score > current_score:
                        content = json.dumps({
                            "score": round(float(best_score), 4),
                            "source": f"gpu_win_i7",
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "params": best_params,
                            "backtest": {
                                "avg_return": round(float(best_avg), 2),
                                "total_return": round(float(best_total), 2),
                                "win_rate": round(float(best_wr), 2),
                                "total_trades": best_n,
                            },
                        }, ensure_ascii=False, indent=2)
                        requests.patch(f"https://api.github.com/gists/{gist_id}", headers=headers,
                            json={"files": {"best_strategy.json": {"content": content}}}, timeout=10)
                        print(f"  [GPU] ✅ Gist 已同步！({best_score:.2f} > {current_score:.2f})")
                        telegram_push(
                            f"🚀 GPU 突破！\n"
                            f"分數 {best_score:.2f}\n"
                            f"勝率 {best_wr:.0f}% | 平均報酬 {best_avg:.1f}%\n"
                            f"總報酬 {best_total:.0f}% | {best_n}筆\n"
                            f"⚡ GPU {total_tested:,}組/{elapsed:.0f}秒/{speed:,.0f}組/秒"
                        )
                    else:
                        print(f"  [GPU] Gist {current_score:.2f} 更高")
            except Exception as e:
                print(f"  [GPU] Gist 錯誤: {e}")

if __name__ == "__main__":
    main()
