#!/usr/bin/env python3
"""
台股策略進化引擎 v2 - Turbo Edition
使用 NumPy 向量化 + Numba JIT 編譯 + 多核心並行
比舊版快 100-1000 倍
"""

import numpy as np
import numba as nb
import json
import os
import sys
import pickle
import requests
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# === 路徑 ===
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SKILLS_DIR, "cache", "stock_data.pkl")
BEST_FILE = os.path.join(SKILLS_DIR, "best_strategy.json")
BEST_EVER_FILE = os.path.join(SKILLS_DIR, "best_ever.json")
EVOLUTION_LOG = os.path.join(SKILLS_DIR, "evolution_log.json")
HIGH_SCORE_FILE = os.path.join(SKILLS_DIR, "all_time_high_score.json")

# === Telegram ===
TELEGRAM_BOT_TOKEN = "8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI"
TELEGRAM_CHAT_IDS = ["5785839733", "8236911077"]
GIST_ID = "c1bef892d33589baef2142ce250d18c2"
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")

def sync_to_gist(score, output):
    """突破時同步到 Gist，讓所有機器知道新的最高分"""
    try:
        r = requests.get(f"https://api.github.com/gists/{GIST_ID}",
            headers={"Authorization": f"token {GH_TOKEN}"}, timeout=10)
        gist_data = r.json()
        current_score = json.loads(list(gist_data["files"].values())[0]["content"]).get("score", 0)
        if score > current_score:
            gist_content = json.dumps({
                "score": round(score, 4),
                "source": "local_mac",
                "updated_at": datetime.now().isoformat(),
                "params": output["params"],
                "backtest": output["backtest"],
                "trade_details": output.get("trade_details", []),
            }, ensure_ascii=False, indent=2)
            requests.patch(f"https://api.github.com/gists/{GIST_ID}",
                headers={"Authorization": f"token {GH_TOKEN}"},
                json={"files": {"best_strategy.json": {"content": gist_content}}},
                timeout=10)
            print(f"✅ Gist 已同步（{score:.2f} > {current_score:.2f}）", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ Gist 同步失敗: {e}", file=sys.stderr)

def telegram_push(message):
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        except:
            pass

# === 中文名 ===
CN_NAMES = {}
# 先載入可能有英文名的
try:
    sys.path.insert(0, SKILLS_DIR)
    from live_pool import LIVE_STOCKS
    CN_NAMES.update(LIVE_STOCKS)
except: pass
# 再用 tw_scanner 的中文名蓋掉
try:
    from tw_scanner import TW_STOCKS
    CN_NAMES.update(TW_STOCKS)
except: pass
# 最後用硬編碼中文名蓋掉一切（最高優先）
CN_NAMES.update({
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
    "1101.TW": "台泥", "1102.TW": "亞泥", "2880.TW": "華南金", "2885.TW": "元大金",
    "2887.TW": "台新金", "2890.TW": "永豐金", "5880.TW": "合庫金", "2884.TW": "玉山金",
    "2892.TW": "第一金", "1326.TW": "台化", "2345.TW": "智邦", "3017.TW": "奇鋐",
    "6669.TW": "緯穎", "2379.TW": "瑞昱", "3034.TW": "聯詠", "2408.TW": "南亞科",
    "3661.TW": "世芯-KY", "2301.TW": "光寶科", "2395.TW": "研華", "2327.TW": "國巨",
    "3008.TW": "大立光", "2634.TW": "漢翔", "1513.TW": "中興電", "1504.TW": "東元",
    "2049.TW": "上銀", "1476.TW": "儒鴻", "9910.TW": "豐泰", "2207.TW": "和泰車",
    "2368.TW": "金像電", "2449.TW": "京元電子", "2383.TW": "台光電",
    "2367.TW": "燿華", "2312.TW": "金寶", "2399.TW": "映泰", "2355.TW": "敬鵬",
    "2329.TW": "華泰", "1312.TW": "國喬", "2369.TW": "菱生", "6282.TW": "康舒",
    "3049.TW": "和鑫", "4906.TW": "正文", "2506.TW": "太設", "1314.TW": "中石化",
    "2605.TW": "新興", "3189.TW": "景碩", "1402.TW": "遠東新", "9921.TW": "巨大",
    "1590.TW": "亞德客-KY", "2474.TW": "可成", "8454.TW": "富邦媒",
    "2542.TW": "興富發", "5534.TW": "長虹", "2404.TW": "漢唐", "3006.TW": "晶豪科",
    "2801.TW": "彰銀", "2834.TW": "臺企銀", "5876.TW": "上海商銀",
    "1210.TW": "大成", "1227.TW": "佳格", "9933.TW": "中鼎",
    "5274.TWO": "信驊", "3529.TWO": "力旺", "5347.TWO": "世界",
    "6488.TWO": "環球晶", "3105.TWO": "穩懋", "4966.TWO": "譜瑞-KY",
    "4743.TWO": "合一", "6547.TWO": "高端疫苗", "4726.TWO": "永昕",
    "5371.TWO": "中光電", "6244.TWO": "茂迪",
})

def get_name(ticker):
    name = CN_NAMES.get(ticker, "")
    if not name or " " in name or (len(name) > 0 and name[0].isupper()):
        return ticker.replace('.TW', '').replace('.TWO', '')
    return name


# ============================================================
# 1. 預計算所有指標（向量化，只算一次）
# ============================================================
def precompute_all_indicators(data_dict, top_n=50):
    """把所有股票的所有指標一次算好，存成 NumPy 陣列"""

    # 按量排名取前 N
    vol_rank = {}
    for t, h in data_dict.items():
        if len(h) >= 60 and "Volume" in h.columns:
            vol_rank[t] = h["Volume"].tail(20).mean()
    sorted_tickers = sorted(vol_rank, key=vol_rank.get, reverse=True)[:top_n]

    if len(sorted_tickers) < 5:
        return None

    # 找共同長度
    min_len = min(len(data_dict[t]) for t in sorted_tickers)
    min_len = min(min_len, 480)  # 最多 480 天（約 2 年）
    n_stocks = len(sorted_tickers)

    # 建立矩陣 (n_stocks, n_days)
    close = np.zeros((n_stocks, min_len))
    high = np.zeros((n_stocks, min_len))
    low = np.zeros((n_stocks, min_len))
    opn = np.zeros((n_stocks, min_len))
    volume = np.zeros((n_stocks, min_len))
    dates = None

    for si, ticker in enumerate(sorted_tickers):
        h = data_dict[ticker]
        close[si] = h["Close"].values[-min_len:]
        high[si] = h["High"].values[-min_len:]
        low[si] = h["Low"].values[-min_len:]
        opn[si] = h["Open"].values[-min_len:]
        volume[si] = h["Volume"].values[-min_len:]
        if dates is None:
            dates = h.index[-min_len:]

    # 向量化計算指標
    indicators = {}
    indicators["close"] = close
    indicators["volume"] = volume
    indicators["high"] = high
    indicators["low"] = low
    indicators["open"] = opn

    # RSI (14)
    delta = np.diff(close, axis=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # 用 EMA 近似
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        if i == 14:
            avg_gain[:, i] = np.mean(gain[:, :14], axis=1)
            avg_loss[:, i] = np.mean(loss[:, :14], axis=1)
        else:
            avg_gain[:, i] = (avg_gain[:, i-1] * 13 + gain[:, i-1]) / 14
            avg_loss[:, i] = (avg_loss[:, i-1] * 13 + loss[:, i-1]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    indicators["rsi"] = 100 - 100 / (1 + rs)

    # 均線
    for w in [3, 5, 8, 10, 15, 20, 30, 60]:
        ma = np.zeros_like(close)
        for i in range(w, close.shape[1]):
            ma[:, i] = np.mean(close[:, i-w:i], axis=1)
        indicators[f"ma{w}"] = ma

    # 布林通道
    bb_mid = indicators["ma20"]
    bb_std = np.zeros_like(close)
    for i in range(20, close.shape[1]):
        bb_std[:, i] = np.std(close[:, i-20:i], axis=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    indicators["bb_pos"] = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)
    indicators["bb_width"] = np.where(bb_mid > 0, bb_range / bb_mid, 0)

    # 量能
    vol_ma20 = np.zeros_like(volume)
    for i in range(20, volume.shape[1]):
        vol_ma20[:, i] = np.mean(volume[:, i-20:i], axis=1)
    indicators["vol_ratio"] = np.where(vol_ma20 > 0, volume / vol_ma20, 1)

    # OBV
    obv = np.zeros_like(volume)
    obv[:, 0] = volume[:, 0]
    for i in range(1, close.shape[1]):
        obv[:, i] = np.where(close[:, i] > close[:, i-1], obv[:, i-1] + volume[:, i],
                    np.where(close[:, i] < close[:, i-1], obv[:, i-1] - volume[:, i], obv[:, i-1]))
    
    obv_ma = np.zeros_like(obv)
    for i in range(10, obv.shape[1]):
        obv_ma[:, i] = np.mean(obv[:, i-10:i], axis=1)
        
    indicators["obv"] = obv
    indicators["obv_ma"] = obv_ma

    # MACD
    ema12 = np.zeros_like(close)
    ema26 = np.zeros_like(close)
    ema12[:, 0] = close[:, 0]
    ema26[:, 0] = close[:, 0]
    for i in range(1, close.shape[1]):
        ema12[:, i] = ema12[:, i-1] * (1 - 2/13) + close[:, i] * 2/13
        ema26[:, i] = ema26[:, i-1] * (1 - 2/27) + close[:, i] * 2/27
    macd_line = ema12 - ema26
    macd_signal = np.zeros_like(close)
    macd_signal[:, 0] = macd_line[:, 0]
    for i in range(1, close.shape[1]):
        macd_signal[:, i] = macd_signal[:, i-1] * (1 - 2/10) + macd_line[:, i] * 2/10
    indicators["macd_line"] = macd_line
    indicators["macd_hist"] = macd_line - macd_signal

    # KD
    low_9 = np.zeros_like(close)
    high_9 = np.zeros_like(close)
    for i in range(9, close.shape[1]):
        low_9[:, i] = np.min(low[:, i-9:i+1], axis=1)
        high_9[:, i] = np.max(high[:, i-9:i+1], axis=1)
    rsv = np.where((high_9 - low_9) > 0, (close - low_9) / (high_9 - low_9) * 100, 50)
    k_val = np.zeros_like(close)
    d_val = np.zeros_like(close)
    k_val[:, 0] = 50
    d_val[:, 0] = 50
    for i in range(1, close.shape[1]):
        k_val[:, i] = k_val[:, i-1] * 2/3 + rsv[:, i] * 1/3
        d_val[:, i] = d_val[:, i-1] * 2/3 + k_val[:, i] * 1/3
    indicators["k_val"] = k_val
    indicators["d_val"] = d_val

    # 前三日低點
    low3 = np.zeros_like(low)
    for i in range(3, low.shape[1]):
        low3[:, i] = np.min(low[:, i-3:i], axis=1)
    indicators["low3"] = low3

    # ATR (14)
    tr = np.zeros_like(close)
    tr[:, 1:] = np.maximum(high[:, 1:] - low[:, 1:], 
                           np.maximum(np.abs(high[:, 1:] - close[:, :-1]), 
                                      np.abs(low[:, 1:] - close[:, :-1])))
    atr = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        if i == 14:
            atr[:, i] = np.mean(tr[:, 1:15], axis=1)
        else:
            atr[:, i] = (atr[:, i-1] * 13 + tr[:, i]) / 14
    indicators["atr"] = atr

    # 乖離率 (BIAS)
    for w in [5, 10, 20]:
        ma = indicators[f"ma{w}"]
        bias = np.where(ma > 0, (close - ma) / ma * 100, 0)
        indicators[f"bias{w}"] = bias

    # 動量
    for d in [3, 5, 10]:
        mom = np.zeros_like(close)
        mom[:, d:] = (close[:, d:] / close[:, :-d] - 1) * 100
        indicators[f"momentum_{d}"] = mom

    # 連續紅K
    indicators["is_green"] = (close > opn).astype(np.float64)

    # 跳空
    gap = np.zeros_like(close)
    gap[:, 1:] = (opn[:, 1:] / close[:, :-1] - 1) * 100
    indicators["gap"] = gap

    # 近20日新高距離
    high_20 = np.zeros_like(close)
    for i in range(20, close.shape[1]):
        high_20[:, i] = np.max(high[:, i-20:i+1], axis=1)
    indicators["near_high"] = np.where(high_20 > 0, (close / high_20 - 1) * 100, 0)

    # Williams %R (14)
    low_14 = np.zeros_like(close)
    high_14 = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        low_14[:, i] = np.min(low[:, i-14:i+1], axis=1)
        high_14[:, i] = np.max(high[:, i-14:i+1], axis=1)
    indicators["williams_r"] = np.where((high_14 - low_14) > 0, (high_14 - close) / (high_14 - low_14) * -100, -50)

    return {
        "tickers": sorted_tickers,
        "dates": dates,
        "n_stocks": n_stocks,
        "n_days": min_len,
        "indicators": indicators,
    }


# ============================================================
# 2. Numba 編譯的核心回測函數（超快）
# ============================================================
@nb.njit(cache=True)
def simulate_sequential_trading(
    n_stocks, n_days, close, rsi, bb_pos, vol_ratio, macd_line, macd_hist,
    k_val, d_val, momentum, is_green, gap, near_high, williams_r, ma_fast, ma_slow, ma60,
    bb_width, vol_ratio_prev, low3, obv, obv_ma, atr, bias5, bias10, bias20,
    # 買入參數
    use_rsi_buy, rsi_buy, use_bb_buy, bb_buy_th, use_vol, vol_th,
    require_ma_bull, use_macd, macd_mode, use_kd, kd_k_th, kd_cross,
    mom_min, vol_inc_days, bb_width_min, consec_green, use_gap,
    near_high_pct, above_ma60, require_ma_cross, vol_gt_yesterday, use_obv_buy,
    bias_buy_w, bias_buy_th, use_wr_buy, wr_buy_th,
    # 賣出參數
    stop_loss, use_tp, take_profit, trailing_stop,
    use_rsi_sell, rsi_sell_th, use_macd_sell, use_kd_sell,
    sell_below_ma_period, sell_below_low_w, sell_vol_shrink,
    stagnation_days, stagnation_min_ret,
    use_atr_stop, atr_stop_n, bias_sell_w, bias_sell_th,
    hold_days_max,
):
    """
    Numba 編譯的序列交易模擬：一次只持一檔，賣了才買下一檔。
    回傳：(n_trades, total_return, returns_array, trade_info)
    """
    MAX_TRADES = 100
    trade_returns = np.zeros(MAX_TRADES)
    trade_stocks = np.zeros(MAX_TRADES, dtype=nb.int64)
    trade_buy_days = np.zeros(MAX_TRADES, dtype=nb.int64)
    trade_sell_days = np.zeros(MAX_TRADES, dtype=nb.int64)
    trade_hold_days = np.zeros(MAX_TRADES, dtype=nb.int64)
    trade_reasons = np.zeros(MAX_TRADES, dtype=nb.int64)  # 0=到期 1=停利 2=停損 3=RSI 4=移動停利 5=MACD死叉 6=KD死叉 7=跌破均線 8=量縮 9=跌破前低 10=停滯
    n_trades = 0

    holding = -1  # -1 = 空倉
    buy_price = 0.0
    buy_day = 0
    peak_price = 0.0
    atr_stop_val = 0.0

    for day in range(30, n_days - 1):
        # === 有持倉：檢查賣出 ===
        if holding >= 0:
            si = holding
            current = close[si, day]
            days_held = day - buy_day
            ret = (current / buy_price - 1) * 100

            if days_held < 1:
                continue

            if current > peak_price:
                peak_price = current
                if use_atr_stop > 0:
                    atr_stop_val = peak_price - atr_stop_n * atr[si, day]

            sell = False
            reason = 0

            # 1. 停損（永遠開）
            if ret <= stop_loss:
                sell = True
                reason = 2

            # 2. ATR 移動停損
            elif use_atr_stop > 0 and current < atr_stop_val:
                sell = True
                reason = 11

            # 3. 乖離率過大賣出
            elif bias_sell_w > 0:
                cur_bias = 0.0
                if bias_sell_w == 5: cur_bias = bias5[si, day]
                elif bias_sell_w == 10: cur_bias = bias10[si, day]
                elif bias_sell_w == 20: cur_bias = bias20[si, day]
                if cur_bias > bias_sell_th:
                    sell = True
                    reason = 12

            # 4. 停利
            elif use_tp == 1 and ret >= take_profit:
                sell = True
                reason = 1

            # 移動停利
            elif trailing_stop > 0 and peak_price > buy_price:
                dd = (current / peak_price - 1) * 100
                if dd <= -trailing_stop:
                    sell = True
                    reason = 4

            # RSI 超買
            elif use_rsi_sell == 1 and rsi[si, day] >= rsi_sell_th:
                sell = True
                reason = 3

            # MACD 死叉
            elif use_macd_sell == 1 and day >= 1:
                if macd_hist[si, day] < 0 and macd_hist[si, day-1] >= 0:
                    sell = True
                    reason = 5

            # KD 死叉
            elif use_kd_sell == 1 and day >= 1:
                if k_val[si, day] < d_val[si, day] and k_val[si, day-1] >= d_val[si, day-1]:
                    sell = True
                    reason = 6

            # 跌破均線
            elif sell_below_ma_period > 0:
                if current < ma_fast[si, day]:
                    sell = True
                    reason = 7

            # 跌破前三日低點
            elif sell_below_low_w > 0:
                if current < low3[si, day]:
                    sell = True
                    reason = 9

            # 量縮
            elif sell_vol_shrink > 0 and days_held >= 2:
                if vol_ratio[si, day] < sell_vol_shrink:
                    sell = True
                    reason = 8

            # 停滯出場 (Stagnation Exit)
            elif stagnation_days > 0 and days_held >= stagnation_days:
                if ret < stagnation_min_ret:
                    sell = True
                    reason = 10

            # 到期
            elif days_held >= hold_days_max:
                sell = True
                reason = 0

            if sell and n_trades < MAX_TRADES:
                trade_returns[n_trades] = ret
                trade_stocks[n_trades] = si
                trade_buy_days[n_trades] = buy_day
                trade_sell_days[n_trades] = day
                trade_hold_days[n_trades] = days_held
                trade_reasons[n_trades] = reason
                n_trades += 1
                holding = -1
            continue

        # === 空倉：找買入訊號 ===
        best_si = -1
        best_vol = 0.0

        for si in range(n_stocks):
            # RSI
            if use_rsi_buy == 1 and rsi[si, day] < rsi_buy:
                continue

            # 布林
            if use_bb_buy == 1 and bb_pos[si, day] < bb_buy_th:
                continue

            # 量能
            if use_vol == 1 and vol_ratio[si, day] < vol_th:
                continue

            # 均線多頭
            if require_ma_bull == 1 and close[si, day] < ma_fast[si, day]:
                continue

            # MACD
            if use_macd == 1:
                if macd_mode == 0:  # golden_cross
                    if not (macd_hist[si, day] > 0 and macd_hist[si, day-1] <= 0):
                        continue
                elif macd_mode == 1:  # above_zero
                    if macd_line[si, day] <= 0:
                        continue

            # KD
            if use_kd == 1:
                if k_val[si, day] < kd_k_th:
                    continue
                if kd_cross == 1:
                    if not (k_val[si, day] > d_val[si, day] and k_val[si, day-1] <= d_val[si, day-1]):
                        continue

            # 動量
            if mom_min > 0 and momentum[si, day] < mom_min:
                continue

            # 連續放量
            if vol_inc_days >= 2:
                valid = True
                for v in range(vol_inc_days):
                    if day - v < 1 or vol_ratio[si, day - v] < 1.0:
                        valid = False
                        break
                if not valid:
                    continue

            # 布林帶寬
            if bb_width_min > 0 and bb_width[si, day] < bb_width_min:
                continue

            # 連續紅K
            if consec_green >= 1:
                valid = True
                for g in range(consec_green):
                    if day - g < 0 or is_green[si, day - g] != 1:
                        valid = False
                        break
                if not valid:
                    continue

            # 跳空
            if use_gap == 1 and gap[si, day] < 1.0:
                continue

            # 近新高
            if near_high_pct > 0 and abs(near_high[si, day]) > near_high_pct:
                continue

            # 站上MA60
            if above_ma60 == 1 and close[si, day] < ma60[si, day]:
                continue

            # 快線>慢線
            if require_ma_cross == 1 and ma_fast[si, day] < ma_slow[si, day]:
                continue

            # 今量>昨量
            if vol_gt_yesterday == 1 and day >= 1:
                if vol_ratio[si, day] <= vol_ratio_prev[si, day]:
                    continue

            # 乖離率過大不買 (防過熱)
            if bias_buy_w > 0:
                cur_bias = 0.0
                if bias_buy_w == 5: cur_bias = bias5[si, day]
                elif bias_buy_w == 10: cur_bias = bias10[si, day]
                elif bias_buy_w == 20: cur_bias = bias20[si, day]
                if cur_bias > bias_buy_th:
                    continue

            # OBV 趨勢向上
            if use_obv_buy == 1:
                if obv[si, day] <= obv_ma[si, day]:
                    continue

            # Williams %R (強勢區間: -20 to -50)
            if use_wr_buy == 1:
                if williams_r[si, day] < wr_buy_th: # WR 太低 (<-50) 不買
                    continue

            if vol_ratio[si, day] > best_vol:
                best_si = si
                best_vol = vol_ratio[si, day]

        # 買入量能最大的
        if best_si >= 0 and day + 1 < n_days:
            holding = best_si
            buy_price = close[best_si, day + 1]
            buy_day = day + 1
            peak_price = buy_price
            if use_atr_stop > 0:
                atr_stop_val = buy_price - atr_stop_n * atr[best_si, day]

    return n_trades, trade_returns, trade_stocks, trade_buy_days, trade_sell_days, trade_hold_days, trade_reasons


# ============================================================
# 3. 參數空間
# ============================================================
STRATEGY_PARAMS = {
    # 買入：七大核心 (RSI, BB, VOL, MA, Williams %R, MACD, KD)
    "use_rsi_buy": [0, 1], "rsi_buy": [40, 45, 50, 55, 60, 65, 70, 75, 80],
    "use_bb_buy": [0, 1], "bb_buy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "use_vol_filter": [0, 1], "vol_filter": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    "require_ma_bull": [0, 1], "ma_fast_w": [3, 5, 8, 10], "ma_slow_w": [15, 20, 30, 60],
    "use_wr_buy": [0, 1], "wr_buy": [-10, -20, -30, -40, -50, -60],
    "use_macd": [0, 1], "macd_mode": [0, 1, 2],
    "use_kd": [0, 1], "kd_buy_k": [30, 40, 50, 60, 70, 80], "kd_cross": [0, 1],
    
    # 輔助：動能、連續紅K、跳空、突破等
    "momentum_days": [3, 5, 10], "momentum_min": [0, 3, 5, 8],
    "vol_increase_days": [0, 2, 3],
    "bb_width_min": [0, 0.03, 0.05],
    "consecutive_green": [0, 1, 2, 3, 4], "gap_up": [0, 1],
    "near_high_pct": [0, 5, 10, 15], "above_ma60": [0, 1],
    "require_ma_cross": [0, 1], "vol_gt_yesterday": [0, 1],

    # 賣出：停損、停利與其他條件
    "stop_loss": [-5, -7, -10, -12, -15],
    "use_take_profit": [0, 1], "take_profit": [10, 20, 30, 40, 50, 60, 80],
    "trailing_stop": [0, 5, 8, 10, 15],
    "use_rsi_sell": [0, 1], "rsi_sell": [75, 80, 85, 90, 95],
    
    "sell_below_ma_w": [0, 1],
    "sell_below_low_w": [0, 1],
    "sell_vol_shrink": [0, 0.3, 0.5, 0.7, 0.8],
    "stagnation_days": [0, 3, 5, 7], "stagnation_min_ret": [0.5, 1.0, 2.0],
    "use_atr_stop": [0, 1], "atr_stop_n": [1.5, 2.0, 2.5, 3.0, 3.5],
    "hold_days": [3, 5, 7, 10, 12, 15]
}

REASON_NAMES = ["到期", "停利", "停損", "RSI超買", "移動停利", "MACD死叉", "KD死叉", "跌破均線", "量縮", "跌破前低", "停滯", "ATR停損", "乖離過大"]


def run_single_test(args):
    """單一參數組合的回測（給 multiprocessing 用）"""
    params, precomputed = args
    ind = precomputed["indicators"]
    n_stocks = precomputed["n_stocks"]
    n_days = precomputed["n_days"]

    # 選均線
    ma_fast_w = params.get("ma_fast_w", 5)
    ma_slow_w = params.get("ma_slow_w", 20)
    if ma_fast_w >= ma_slow_w:
        return None

    ma_fast = ind.get(f"ma{ma_fast_w}", ind["ma5"])
    ma_slow = ind.get(f"ma{ma_slow_w}", ind["ma20"])
    ma60 = ind.get("ma60", ind["ma20"])

    mom_days = params.get("momentum_days", 5)
    momentum = ind.get(f"momentum_{mom_days}", ind["momentum_5"])

    # 昨天量比
    vol_prev = np.zeros_like(ind["vol_ratio"])
    vol_prev[:, 1:] = ind["vol_ratio"][:, :-1]

    n_trades, returns, stocks, buy_days, sell_days, hold_days, reasons = simulate_sequential_trading(
        n_stocks, n_days,
        ind["close"], ind["rsi"], ind["bb_pos"], ind["vol_ratio"],
        ind["macd_line"], ind["macd_hist"], ind["k_val"], ind["d_val"],
        momentum, ind["is_green"], ind["gap"], ind["near_high"],
        ind["williams_r"], ma_fast, ma_slow, ma60, ind["bb_width"], vol_prev, ind["low3"],
        ind["obv"], ind["obv_ma"], ind["atr"], ind["bias5"], ind["bias10"], ind["bias20"],
        # 買入
        params.get("use_rsi_buy", 1), params.get("rsi_buy", 55),
        params.get("use_bb_buy", 1), params.get("bb_buy", 0.7),
        params.get("use_vol_filter", 1), params.get("vol_filter", 3.0),
        params.get("require_ma_bull", 0), params.get("use_macd", 0),
        params.get("macd_mode", 2), params.get("use_kd", 0),
        params.get("kd_buy_k", 50), params.get("kd_cross", 0),
        params.get("momentum_min", 0), params.get("vol_increase_days", 0),
        params.get("bb_width_min", 0), params.get("consecutive_green", 0),
        params.get("gap_up", 0), params.get("near_high_pct", 0),
        params.get("above_ma60", 0), params.get("require_ma_cross", 0),
        params.get("vol_gt_yesterday", 0), params.get("use_obv_buy", 0),
        params.get("bias_buy_w", 0), params.get("bias_buy_th", 10),
        params.get("use_wr_buy", 0), params.get("wr_buy", -30),
        # 賣出
        params.get("stop_loss", -10), params.get("use_take_profit", 1),
        params.get("take_profit", 20), params.get("trailing_stop", 0),
        params.get("use_rsi_sell", 1), params.get("rsi_sell_th", 90),
        params.get("use_macd_sell", 0), params.get("use_kd_sell", 0),
        params.get("sell_below_ma_w", 0), params.get("sell_below_low_w", 0),
        params.get("sell_vol_shrink", 0),
        params.get("stagnation_days", 0), params.get("stagnation_min_ret", 1.0),
        params.get("use_atr_stop", 0), params.get("atr_stop_n", 2.0),
        params.get("bias_sell_w", 0), params.get("bias_sell_th", 20),
        params.get("hold_days", 10),
    )

    if n_trades < 10:  # 2 年至少要有 10 筆交易才算數
        return None

    rets = returns[:n_trades]
    bds = buy_days[:n_trades]
    avg_ret = np.mean(rets)
    total_ret = np.sum(rets)
    win_rate = np.sum(rets > 0) / n_trades * 100
    avg_hold = np.mean(hold_days[:n_trades])
    max_ret = np.max(rets)
    min_ret = np.min(rets)
    winners = rets[rets > 0]
    losers = rets[rets <= 0]
    pf = abs(np.sum(winners) / np.sum(losers)) if len(losers) > 0 and np.sum(losers) != 0 else 999

    # 硬性門檻
    if avg_ret < 6:
        return None
    if win_rate < 50:
        return None
    if avg_hold > 15 or avg_hold < 1:
        return None

    # === 雙段驗證（防過擬合）===
    # 把交易分成前半段和後半段，兩段都要賺才算數
    n_days = precomputed["n_days"]
    mid_day = n_days // 2
    first_half = rets[bds < mid_day]
    second_half = rets[bds >= mid_day]

    # 兩段都要有交易，且兩段都要正報酬
    if len(first_half) < 2 or len(second_half) < 2:
        return None
    if np.mean(first_half) < 0 or np.mean(second_half) < 0:
        return None

    # 後半段報酬不能比前半段差太多（防止只在某個時期有效）
    consistency = min(np.mean(first_half), np.mean(second_half)) / max(np.mean(first_half), np.mean(second_half))
    # consistency 越接近 1 代表前後段越穩定

    # 白做工懲罰（報酬 < 5% 都算白做工）
    wasted = np.sum(rets < 5) / n_trades * 100

    # 白做工超過 60% 直接淘汰
    if wasted > 60:
        return None

    score = (
        total_ret * 0.15 +              # 總報酬（15%）
        avg_ret * 0.30 +                 # 平均報酬（30%）— 最重要！每筆都要賺
        win_rate * 0.10 +                # 勝率（10%）
        min(pf, 5) * 3 * 0.05 +         # 利潤因子（5%）
        consistency * 20 * 0.10 +        # 前後段穩定性（10%）
        n_trades * 0.5 * 0.10 +          # 交易筆數（10%）— 降權，不鼓勵亂買
        -wasted * 0.20                   # 白做工懲罰（20%）— 加重！
    )

    return {
        "score": score,
        "params": params,
        "n_trades": int(n_trades),
        "avg_return": float(avg_ret),
        "total_return": float(total_ret),
        "win_rate": float(win_rate),
        "max_return": float(max_ret),
        "min_return": float(min_ret),
        "avg_hold_days": float(avg_hold),
        "profit_factor": float(pf),
        "trades_idx": {
            "stocks": stocks[:n_trades].tolist(),
            "buy_days": buy_days[:n_trades].tolist(),
            "sell_days": sell_days[:n_trades].tolist(),
            "returns": rets.tolist(),
            "hold_days": hold_days[:n_trades].tolist(),
            "reasons": reasons[:n_trades].tolist(),
        },
    }


# ============================================================
# 4. 主進化引擎
# ============================================================
def evolve(n_tests=5000):
    """主進化流程"""
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] 🚀 Turbo 進化引擎啟動...", file=sys.stderr)

    # 載入資料
    if not os.path.exists(CACHE_FILE):
        print("快取不存在", file=sys.stderr)
        return
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)

    # 預計算指標（只算一次）
    t0 = datetime.now()
    precomputed = precompute_all_indicators(data, top_n=50)
    if precomputed is None:
        print("資料不足", file=sys.stderr)
        return
    t1 = datetime.now()
    print(f"指標預算完成：{precomputed['n_stocks']}檔 x {precomputed['n_days']}天 | {(t1-t0).total_seconds():.2f}秒", file=sys.stderr)

    # Numba 預熱（第一次會編譯，之後就快了）
    print("Numba 編譯中（首次較慢）...", file=sys.stderr)
    dummy_params = {k: v[0] for k, v in STRATEGY_PARAMS.items()}
    dummy_params["ma_fast_w"] = 5
    dummy_params["ma_slow_w"] = 20
    run_single_test((dummy_params, precomputed))
    t2 = datetime.now()
    print(f"Numba 編譯完成 | {(t2-t1).total_seconds():.2f}秒", file=sys.stderr)

    # 載入歷史最佳分數
    best_score = -999999
    if os.path.exists(BEST_FILE):
        try:
            with open(BEST_FILE, "r") as f:
                best_score = json.load(f).get("score", -999999)
        except: pass
    if os.path.exists(HIGH_SCORE_FILE):
        try:
            with open(HIGH_SCORE_FILE, "r") as f:
                ath = json.load(f).get("score", 0)
                if ath > best_score:
                    best_score = ath
        except: pass
    print(f"歷史最佳：{best_score:.2f}", file=sys.stderr)

    # 產生隨機參數組合
    np.random.seed(int(datetime.now().timestamp()) % 2**31)
    param_sets = []
    for _ in range(n_tests):
        p = {k: np.random.choice(v) for k, v in STRATEGY_PARAMS.items()}
        # 轉原生類型
        p = {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for k, v in p.items()}
        if p.get("ma_fast_w", 5) >= p.get("ma_slow_w", 20):
            continue
        param_sets.append((p, precomputed))

    print(f"開始回測 {len(param_sets)} 組 | {multiprocessing.cpu_count()} 核心並行...", file=sys.stderr)

    # 多核心並行回測
    t3 = datetime.now()
    best_result = None
    improved = 0
    tested = 0

    n_workers = max(1, multiprocessing.cpu_count() - 2)  # 火力全開，留 2 核給系統

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_test, ps): ps for ps in param_sets}
        for future in as_completed(futures):
            tested += 1
            try:
                result = future.result()
                if result and result["score"] > best_score:
                    best_score = result["score"]
                    best_result = result
                    improved += 1
                    print(f"  新紀錄！{result['score']:.1f} | 勝率{result['win_rate']:.0f}% | 平均報酬{result['avg_return']:.1f}% | 總報酬{result['total_return']:.0f}% | {result['n_trades']}筆", file=sys.stderr)
            except:
                pass

    t4 = datetime.now()
    elapsed = (t4 - t3).total_seconds()
    speed = tested / elapsed if elapsed > 0 else 0
    total_elapsed = (t4 - start_time).total_seconds()

    print(f"\n完成 | {tested}組 | {elapsed:.1f}秒 | {speed:.0f}組/秒 | 突破{improved}次", file=sys.stderr)

    # 儲存結果
    if best_result:
        tickers = precomputed["tickers"]
        dates = precomputed["dates"]
        close = precomputed["indicators"]["close"]

        # 組裝交易明細
        ti = best_result["trades_idx"]
        trade_details = []
        for j in range(best_result["n_trades"]):
            si = int(ti["stocks"][j])
            bd = int(ti["buy_days"][j])
            sd = int(ti["sell_days"][j])
            ticker = tickers[si]
            trade_details.append({
                "ticker": ticker,
                "buy_date": str(dates[bd].date()),
                "sell_date": str(dates[sd].date()),
                "buy_price": round(float(close[si, bd]), 2),
                "sell_price": round(float(close[si, sd]), 2),
                "return": round(ti["returns"][j], 2),
                "days": int(ti["hold_days"][j]),
                "reason": REASON_NAMES[int(ti["reasons"][j])],
            })

        # 版本號
        prev_version = 0
        if os.path.exists(BEST_FILE):
            try:
                with open(BEST_FILE, "r") as f:
                    prev_version = json.load(f).get("version", 0)
            except: pass
        new_version = prev_version + 1

        # 策略描述
        p = best_result["params"]
        desc_parts = []
        if p.get("use_rsi_buy", 1): desc_parts.append(f"RSI>={p['rsi_buy']}")
        if p.get("use_bb_buy", 1): desc_parts.append(f"布林>={p['bb_buy']*100:.0f}%")
        if p.get("use_vol_filter", 1): desc_parts.append(f"量>={p['vol_filter']}x")
        if p.get("use_macd", 0): desc_parts.append("MACD")
        if p.get("use_kd", 0): desc_parts.append(f"KD>={p.get('kd_buy_k',50)}")
        if p.get("momentum_min", 0) > 0: desc_parts.append(f"動量>={p['momentum_min']}%")
        if p.get("consecutive_green", 0) > 0: desc_parts.append(f"連{p['consecutive_green']}紅K")

        sell_parts = []
        if p.get("use_take_profit", 1): sell_parts.append(f"停利+{p['take_profit']}%")
        sell_parts.append(f"停損{p['stop_loss']}%")
        if p.get("trailing_stop", 0) > 0: sell_parts.append(f"移動停利{p['trailing_stop']}%")
        if p.get("use_rsi_sell", 0): sell_parts.append(f"RSI>={p['rsi_sell']}")
        if p.get("stagnation_days", 0) > 0: sell_parts.append(f"停滯{p['stagnation_days']}天<{p['stagnation_min_ret']}%")

        strategy_desc = f"買入：{'、'.join(desc_parts)}。賣出：{'、'.join(sell_parts)}。持有{p['hold_days']}天。"

        output = {
            "version": new_version,
            "updated_at": datetime.now().isoformat(),
            "score": round(float(best_score), 4),
            "params": p,
            "backtest": {
                "win_rate": round(best_result["win_rate"], 2),
                "avg_return": round(best_result["avg_return"], 2),
                "total_return": round(best_result["total_return"], 2),
                "max_return": round(best_result["max_return"], 2),
                "min_return": round(best_result["min_return"], 2),
                "avg_hold_days": round(best_result["avg_hold_days"], 2),
                "total_trades": best_result["n_trades"],
                "profit_factor": round(best_result["profit_factor"], 2),
            },
            "trade_summary": {
                "total": len(trade_details),
                "wins": len([t for t in trade_details if t["return"] > 0]),
                "losses": len([t for t in trade_details if t["return"] <= 0]),
            },
            "trade_details": sorted(trade_details, key=lambda x: x["buy_date"]),
            "strategy_description": strategy_desc,
            "engine": f"turbo_v2 | {tested}組 | {elapsed:.1f}秒 | {speed:.0f}組/秒 | {n_workers}核心",
        }

        with open(BEST_FILE, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # 更新永久最高分鎖
        old_ath = 0
        if os.path.exists(HIGH_SCORE_FILE):
            try:
                with open(HIGH_SCORE_FILE, "r") as f:
                    old_ath = json.load(f).get("score", 0)
            except: pass
        if best_score > old_ath:
            with open(HIGH_SCORE_FILE, "w") as f:
                json.dump({"score": round(float(best_score), 4), "time": datetime.now().isoformat()}, f)
            with open(BEST_EVER_FILE, "w") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            # 推播 Telegram
            trade_lines = "\n".join([
                f"  {get_name(t['ticker'])}({t['ticker'].replace('.TW','')}) | {t['buy_date'][5:]}→{t['sell_date'][5:]} | {t['buy_price']}→{t['sell_price']} | {t['return']:+.1f}% | {t['days']}天 | {t['reason']}"
                for t in output["trade_details"]
            ])
            msg = (
                f"🚀 Turbo 策略突破 v{new_version}！\n"
                f"━━━━━━━━━━━━\n"
                f"平均報酬：{best_result['avg_return']:.1f}%\n"
                f"總報酬：{best_result['total_return']:.1f}%\n"
                f"勝率：{best_result['win_rate']:.0f}% | {best_result['n_trades']}筆\n"
                f"持有：{best_result['avg_hold_days']:.1f}天\n"
                f"⚡ {tested}組/{elapsed:.1f}秒/{speed:.0f}組/秒/{n_workers}核\n\n"
                f"策略：{strategy_desc}\n\n"
                f"📋 交易明細：\n{trade_lines}"
            )
            telegram_push(msg)
            sync_to_gist(best_score, output)
        else:
            print(f"分數 {best_score:.2f} 未超越歷史最高 {old_ath:.2f}，不推播", file=sys.stderr)

        # 寫 log
        log_entry = {
            "time": datetime.now().isoformat(),
            "tested": tested,
            "improved": improved,
            "best_score": round(float(best_score), 4),
            "win_rate": round(best_result["win_rate"], 2),
            "avg_return": round(best_result["avg_return"], 2),
            "speed": f"{speed:.0f}組/秒",
            "engine": "turbo_v2",
        }
        log = []
        if os.path.exists(EVOLUTION_LOG):
            try:
                with open(EVOLUTION_LOG, "r") as f:
                    log = json.load(f)
            except: log = []
        log.append(log_entry)
        log = log[-100:]
        with open(EVOLUTION_LOG, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"未突破（歷史最高 {best_score:.2f}）| {tested}組 | {elapsed:.1f}秒 | {speed:.0f}組/秒", file=sys.stderr)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    evolve(n_tests=n)
