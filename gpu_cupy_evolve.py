#!/usr/bin/env python3
"""
GPU 進化引擎 — CuPy 版（RTX 3060 專用）
用 GPU 同時跑上萬組參數的買賣訊號計算
"""
import numpy as np
import cupy as cp
import json, os, sys, time, requests, pickle, base64

# === Telegram / Gist ===
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = os.environ.get("TELEGRAM_CHAT_IDS", "").split(",")
GIST_ID = os.environ.get("GIST_ID", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
DATA_GIST_ID = "a300b9e29372ac76f79eda39a2a86321"
CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")

CN_NAMES = {}
# 從完整名單載入（1958 檔台灣上市櫃股票）
NAMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tw_stock_names.json")
try:
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            CN_NAMES = json.load(f)
except: pass
def get_name(t):
    n = CN_NAMES.get(t, "")
    if not n: return t.replace(".TW","").replace(".TWO","")
    return n
def telegram_push(msg):
    for cid in CHAT_IDS:
        cid = cid.strip()
        if not cid or not BOT_TOKEN: continue
        try: requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json={"chat_id":cid,"text":msg}, timeout=10)
        except: pass

# === 資料載入 ===
import yfinance as yf
# 從完整名單讀取所有股票代號（排除 ETF 00xx 開頭）
TW_TICKERS = []
try:
    nf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tw_stock_names.json")
    with open(nf, "r", encoding="utf-8") as f:
        all_names = json.load(f)
    TW_TICKERS = [t for t in all_names.keys() if not t.startswith("00")]
    CN_NAMES.update(all_names)
except:
    TW_TICKERS = ["2330.TW","2454.TW","2317.TW","2303.TW","2382.TW","3231.TW"]

def download_data():
    if os.path.exists(CACHE_PATH):
        age = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age < 24:
            try:
                with open(CACHE_PATH, "rb") as f: data = pickle.load(f)
                if len(data) >= 10: print(f"[快取] {len(data)} 檔"); return data
            except: pass
    data = {}
    total = len(TW_TICKERS)
    print(f"[下載] {total} 檔股票資料（首次需 10-20 分鐘）...")
    for i, t in enumerate(TW_TICKERS):
        try:
            h = yf.Ticker(t).history(period="2y")
            if len(h) >= 40: data[t] = h
            if i % 10 == 9: time.sleep(1)
            if i % 50 == 49: print(f"  進度 {i+1}/{total} | 成功 {len(data)} 檔")
        except: continue
    print(f"[下載完成] {len(data)} 檔")
    if len(data) < 10:
        try:
            headers = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
            r = requests.get(f"https://api.github.com/gists/{DATA_GIST_ID}", headers=headers, timeout=30)
            finfo = list(r.json()["files"].values())[0]
            raw_url = finfo.get("raw_url", "")
            r2 = requests.get(raw_url, headers=headers, timeout=60) if raw_url else None
            data = pickle.loads(base64.b64decode(r2.text if r2 else finfo["content"]))
            print(f"[Gist] {len(data)} 檔")
        except Exception as e: print(f"[失敗] {e}"); return {}
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

# === GPU 批次交易模擬 ===
# 用 CuPy RawKernel 寫 CUDA C 核心，每個 thread 跑一組參數
CUDA_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void backtest(
    const int n_stocks, const int n_days,
    const float* close, const float* rsi, const float* bb_pos,
    const float* vol_ratio, const float* macd_hist, const float* macd_line,
    const float* k_val, const float* d_val,
    const float* mom3, const float* mom5, const float* mom10,
    const float* is_green, const float* gap, const float* near_high,
    const float* williams_r,
    const float* ma3, const float* ma5, const float* ma10,
    const float* ma15, const float* ma20, const float* ma30, const float* ma60,
    const float* vol_prev,
    const float* squeeze_fire, const float* new_high_60, const float* adx,
    const float* bias, const float* obv_rising, const float* atr_pct,
    const float* params, const int n_params_per_combo,
    float* results, const int n_combos
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_combos) return;

    // 讀參數 — 評分制
    const float* p = params + idx * n_params_per_combo;
    // 買入：每個指標的權重（0-3）和門檻
    int w_rsi = (int)p[0]; float rsi_th = p[1];
    int w_bb = (int)p[2]; float bb_th = p[3];
    int w_vol = (int)p[4]; float vol_th = p[5];
    int w_ma = (int)p[6];
    int w_macd = (int)p[7]; int macd_mode = (int)p[8];
    int w_kd = (int)p[9]; float kd_th = p[10]; int kd_cross = (int)p[11];
    int w_wr = (int)p[12]; float wr_th = p[13];
    int w_mom = (int)p[14]; float mom_th = p[15];
    int w_near_high = (int)p[16]; float near_high_pct = p[17];
    int w_squeeze = (int)p[18]; int w_new_high = (int)p[19];
    int w_adx = (int)p[20]; float adx_th = p[21];
    int consec_green = (int)p[22]; int use_gap = (int)p[23];
    int above_ma60 = (int)p[24]; int vol_gt_yesterday = (int)p[25];
    float buy_threshold = p[26];
    // 賣出
    float stop_loss = p[27]; int use_tp = (int)p[28]; float take_profit = p[29];
    float trailing_stop = p[30];
    int use_rsi_sell = (int)p[31]; float rsi_sell_th = p[32];
    int use_macd_sell = (int)p[33]; int use_kd_sell = (int)p[34];
    float sell_vol_shrink = p[35]; int sell_below_ma = (int)p[36];
    int hold_days_max = (int)p[37];
    // BIAS 乖離率
    int w_bias = (int)p[38]; float bias_max_th = p[39];
    // 停滯出場
    int use_stagnation = (int)p[40]; int stag_days = (int)p[41]; float stag_min_ret = p[42];
    // 保本停損
    int use_breakeven = (int)p[43]; float breakeven_trigger = p[44];
    // OBV 能量潮
    int w_obv = (int)p[45]; int obv_days = (int)p[46];
    // ATR 波動率門檻
    int w_atr = (int)p[47]; float atr_min = p[48];
    // 漸進式最低報酬
    int use_time_decay = (int)p[49]; float ret_per_day = p[50];
    // 鎖利出場
    int use_profit_lock = (int)p[51]; float lock_trigger = p[52]; float lock_floor = p[53];
    // 動量反轉出場
    int use_mom_exit = (int)p[54]; float mom_exit_th = p[55];
    // 多持倉
    int max_pos = (int)p[56]; if (max_pos < 1) max_pos = 1; if (max_pos > 3) max_pos = 3;
    // MA/MOM 選擇
    int ma_fast_idx = (int)p[57];
    int ma_slow_idx = (int)p[58];
    int mom_idx = (int)p[59];

    const float* ma_fast_arr = ma_fast_idx==0 ? ma3 : ma_fast_idx==1 ? ma5 : ma10;
    const float* ma_slow_arr = ma_slow_idx==0 ? ma15 : ma_slow_idx==1 ? ma20 : ma_slow_idx==2 ? ma30 : ma60;
    const float* momentum = mom_idx==0 ? mom3 : mom_idx==1 ? mom5 : mom10;

    // 交易模擬 — 支援多持倉（max_pos 1-3）
    int hold_si[3]; hold_si[0]=-1; hold_si[1]=-1; hold_si[2]=-1;
    float hold_bp[3]; hold_bp[0]=0; hold_bp[1]=0; hold_bp[2]=0;
    float hold_pk[3]; hold_pk[0]=0; hold_pk[1]=0; hold_pk[2]=0;
    int hold_bd[3]; hold_bd[0]=0; hold_bd[1]=0; hold_bd[2]=0;
    int n_holding = 0, n_trades = 0;
    float total_ret = 0, win_count = 0, wasted_count = 0;
    float rets[100];
    int trade_bdays[100];

    for (int day = 30; day < n_days - 1; day++) {
        // === Phase 1: 檢查所有持倉的賣出條件 ===
        bool any_sold = false;
        for (int h = 0; h < max_pos; h++) {
            if (hold_si[h] < 0) continue;
            int si = hold_si[h];
            float cur = close[si * n_days + day];
            int dh = day - hold_bd[h];
            float ret = (cur / hold_bp[h] - 1.0f) * 100.0f;
            if (dh < 1) continue;
            if (cur > hold_pk[h]) hold_pk[h] = cur;
            bool sell = false;

            float effective_stop = stop_loss;
            if (use_breakeven == 1 && (hold_pk[h] / hold_bp[h] - 1.0f) * 100.0f >= breakeven_trigger)
                effective_stop = 0;
            if (ret <= effective_stop) sell = true;
            if (!sell && use_tp == 1 && ret >= take_profit) sell = true;
            if (!sell && trailing_stop > 0 && hold_pk[h] > hold_bp[h]) {
                if ((cur / hold_pk[h] - 1.0f) * 100.0f <= -trailing_stop) sell = true;
            }
            if (!sell && use_rsi_sell == 1 && rsi[si * n_days + day] >= rsi_sell_th) sell = true;
            if (!sell && use_macd_sell == 1 && day >= 1) {
                if (macd_hist[si * n_days + day] < 0 && macd_hist[si * n_days + day - 1] >= 0) sell = true;
            }
            if (!sell && use_kd_sell == 1 && day >= 1) {
                if (k_val[si*n_days+day] < d_val[si*n_days+day] && k_val[si*n_days+day-1] >= d_val[si*n_days+day-1]) sell = true;
            }
            if (!sell && sell_vol_shrink > 0 && dh >= 2 && vol_ratio[si*n_days+day] < sell_vol_shrink) sell = true;
            if (!sell && sell_below_ma > 0) {
                float ma_check = 0;
                if (sell_below_ma == 1) ma_check = ma_fast_arr[si*n_days+day];
                else if (sell_below_ma == 2) ma_check = ma_slow_arr[si*n_days+day];
                else if (sell_below_ma == 3) ma_check = ma60[si*n_days+day];
                if (ma_check > 0 && cur < ma_check) sell = true;
            }
            if (!sell && use_stagnation == 1 && dh >= stag_days && ret < stag_min_ret) sell = true;
            if (!sell && use_time_decay == 1 && dh >= hold_days_max / 2 && ret < (dh - hold_days_max / 2) * ret_per_day) sell = true;
            if (!sell && use_profit_lock == 1) {
                float peak_gain = (hold_pk[h] / hold_bp[h] - 1.0f) * 100.0f;
                if (peak_gain >= lock_trigger && ret < lock_floor) sell = true;
            }
            if (!sell && use_mom_exit == 1 && dh >= 10 && momentum[hold_si[h] * n_days + day] < -mom_exit_th) sell = true;
            if (!sell && dh >= hold_days_max) sell = true;

            if (sell && n_trades < 100) {
                rets[n_trades] = ret;
                trade_bdays[n_trades] = hold_bd[h];
                total_ret += ret;
                if (ret > 0) win_count += 1;
                if (ret < 10) wasted_count += 1;
                n_trades++;
                hold_si[h] = -1;
                n_holding--;
                any_sold = true;
            }
        }

        // === Phase 2: 有空位且今天沒賣出 → 買入最高分 ===
        if (n_holding < max_pos && !any_sold) {
            int best_si = -1;
            float best_buy_score = 0;
            for (int si = 0; si < n_stocks; si++) {
                // 跳過已持有的股票
                bool already = false;
                for (int h = 0; h < max_pos; h++) {
                    if (hold_si[h] == si) { already = true; break; }
                }
                if (already) continue;

                int d = si * n_days + day;
                float sc = 0.0f;

                if (w_rsi > 0 && rsi[d] >= rsi_th) sc += w_rsi;
                if (w_bb > 0 && bb_pos[d] >= bb_th) sc += w_bb;
                if (w_vol > 0 && vol_ratio[d] >= vol_th) sc += w_vol;
                if (w_ma > 0 && close[d] > ma_fast_arr[d]) sc += w_ma;
                if (w_macd > 0) {
                    bool ok = false;
                    if (macd_mode == 0 && day >= 1 && macd_hist[d] > 0 && macd_hist[d-1] <= 0) ok = true;
                    else if (macd_mode == 1 && macd_line[d] > 0) ok = true;
                    else if (macd_mode == 2 && macd_hist[d] > 0) ok = true;
                    if (ok) sc += w_macd;
                }
                if (w_kd > 0) {
                    bool ok = k_val[d] >= kd_th;
                    if (ok && kd_cross == 1 && day >= 1)
                        ok = k_val[d] > d_val[d] && k_val[d-1] <= d_val[d-1];
                    if (ok) sc += w_kd;
                }
                if (w_wr > 0 && williams_r[d] >= wr_th) sc += w_wr;
                if (w_mom > 0 && momentum[d] >= mom_th) sc += w_mom;
                if (w_near_high > 0 && fabsf(near_high[d]) <= near_high_pct) sc += w_near_high;
                if (w_squeeze > 0 && squeeze_fire[d] > 0.5f) sc += w_squeeze;
                if (w_new_high > 0 && new_high_60[d] > 0.5f) sc += w_new_high;
                if (w_adx > 0 && adx[d] >= adx_th) sc += w_adx;
                if (w_bias > 0 && bias[d] >= 0 && bias[d] <= bias_max_th) sc += w_bias;
                if (w_obv > 0 && obv_rising[d] > 0.5f) sc += w_obv;
                // ATR 波動率（過濾低波動金融股）
                if (w_atr > 0 && atr_pct[d] >= atr_min) sc += w_atr;

                if (consec_green >= 1) {
                    bool ok = true;
                    for (int g = 0; g < consec_green; g++) {
                        if (day-g < 0 || is_green[si*n_days+day-g] != 1) { ok = false; break; }
                    }
                    if (ok) sc += 1.0f;
                }
                if (use_gap == 1 && gap[d] >= 1.0f) sc += 1.0f;
                if (above_ma60 == 1 && close[d] >= ma60[d]) sc += 1.0f;
                if (vol_gt_yesterday == 1 && day >= 1 && vol_ratio[d] > vol_prev[d]) sc += 1.0f;

                if (sc >= buy_threshold && sc > best_buy_score) {
                    best_si = si; best_buy_score = sc;
                }
            }
            if (best_si >= 0 && day + 1 < n_days) {
                for (int h = 0; h < max_pos; h++) {
                    if (hold_si[h] < 0) {
                        hold_si[h] = best_si;
                        hold_bp[h] = close[best_si * n_days + day + 1];
                        hold_pk[h] = hold_bp[h];
                        hold_bd[h] = day + 1;
                        n_holding++;
                        break;
                    }
                }
            }
        }
    }

    // 計算分數
    float score = -999999.0f;
    if (n_trades >= 10) {
        float avg_ret = total_ret / n_trades;
        float win_rate = win_count / n_trades * 100.0f;
        float wasted = wasted_count / n_trades * 100.0f;

        if (avg_ret >= 15 && win_rate >= 50 && wasted <= 60) {
            // 雙段驗證
            int mid_day = n_days / 2;
            float f_sum=0, s_sum=0; int f_n=0, s_n=0;
            for (int i=0; i<n_trades; i++) {
                if (trade_bdays[i] < mid_day) { f_sum += rets[i]; f_n++; }
                else { s_sum += rets[i]; s_n++; }
            }
            if (f_n >= 2 && s_n >= 2) {
                float f_avg = f_sum/f_n, s_avg = s_sum/s_n;
                if (f_avg > 0 && s_avg > 0) {
                    float consistency = fminf(f_avg,s_avg) / fmaxf(f_avg,s_avg);
                    float w_sum=0, l_sum=0;
                    for (int i=0; i<n_trades; i++) {
                        if (rets[i]>0) w_sum+=rets[i]; else l_sum+=fabsf(rets[i]);
                    }
                    float pf = l_sum>0 ? w_sum/l_sum : 999.0f;
                    if (pf > 5) pf = 5;
                    score = total_ret*0.10f + avg_ret*0.50f + win_rate*0.10f
                          + pf*3*0.05f + consistency*20*0.10f
                          + n_trades*0.5f*0.10f - wasted*0.40f;
                }
            }
        }
    }

    // 寫結果
    results[idx * 5 + 0] = score;
    results[idx * 5 + 1] = (float)n_trades;
    results[idx * 5 + 2] = n_trades > 0 ? total_ret / n_trades : 0;
    results[idx * 5 + 3] = total_ret;
    results[idx * 5 + 4] = n_trades > 0 ? win_count / n_trades * 100.0f : 0;
}
''', 'backtest')

PARAMS_SPACE = {
    # ====== 評分制買入（權重 0-3 + 門檻）======
    "w_rsi": [0,1,2,3], "rsi_th": [30,35,40,45,50,55,60,63,65,68,70,72,75,80],
    "w_bb": [0,1,2,3], "bb_th": [0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
    "w_vol": [0,1,2,3], "vol_th": [1.2,1.5,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0],
    "w_ma": [0,1,2,3],
    "w_macd": [0,1,2,3], "macd_mode": [0,1,2],
    "w_kd": [0,1,2,3], "kd_th": [20,30,40,50,55,60,65,70,75,80,85], "kd_cross": [0,1],
    "w_wr": [0,1,2,3], "wr_th": [-5,-10,-15,-20,-25,-30,-35,-40,-50],
    "w_mom": [0,1,2,3], "mom_th": [0,2,3,5,8,10,12],
    "w_near_high": [0,1,2], "near_high_pct": [3,5,10,15],
    "w_squeeze": [0,1,2,3], "w_new_high": [0,1,2,3],
    "w_adx": [0,1,2,3], "adx_th": [15,20,25,30,35,40],
    "consecutive_green": [0,1,2,3], "gap_up": [0,1],
    "above_ma60": [0,1], "vol_gt_yesterday": [0,1],
    "buy_threshold": [3,4,5,6,7,8,9,10],
    # ====== 賣出 ======
    "stop_loss": [-3,-5,-7,-10,-12,-15,-20],
    "use_take_profit": [0,1], "take_profit": [20,30,40,50,60,80,100,150],
    "trailing_stop": [0,3,5,7,10,15,20],
    "use_rsi_sell": [0,1], "rsi_sell": [70,75,80,85,90,95],
    "use_macd_sell": [0,1], "use_kd_sell": [0,1],
    "sell_vol_shrink": [0,0.3,0.5,0.7],
    "sell_below_ma": [0,1,2,3],
    "hold_days": [5,7,10,15,20,25,30],
    # ====== BIAS 乖離率 ======
    "w_bias": [0,1,2,3], "bias_max": [3,5,8,10,15,20,30],
    # ====== 停滯出場 ======
    "use_stagnation_exit": [0,1], "stagnation_days": [5,7,10,15], "stagnation_min_ret": [3,5,8,10],
    # ====== 保本停損 ======
    "use_breakeven": [0,1], "breakeven_trigger": [10,15,20,25,30],
    # ====== OBV 能量潮 ======
    "w_obv": [0,1,2,3], "obv_rising_days": [3,5,10],
    # ====== 漸進式最低報酬 ======
    "use_time_decay": [0,1], "ret_per_day": [0.1,0.2,0.3,0.5,0.8,1.0,1.5],
    # ====== ATR 波動率門檻（過濾低波動股）======
    "w_atr": [0,1,2,3], "atr_min": [1.0,1.5,2.0,2.5,3.0,4.0],
    # ====== 鎖利出場 ======
    "use_profit_lock": [0,1], "lock_trigger": [15,20,30,40,50], "lock_floor": [3,5,8,10,15],
    # ====== 動量反轉出場 ======
    "use_mom_exit": [0,1], "mom_exit_th": [0,1,2,3,5],
    # ====== 多持倉 ======
    "max_positions": [1,2],
}

PARAM_ORDER = [
    "w_rsi","rsi_th","w_bb","bb_th",
    "w_vol","vol_th","w_ma",
    "w_macd","macd_mode","w_kd","kd_th","kd_cross",
    "w_wr","wr_th","w_mom","mom_th",
    "w_near_high","near_high_pct",
    "w_squeeze","w_new_high",
    "w_adx","adx_th",
    "consecutive_green","gap_up",
    "above_ma60","vol_gt_yesterday",
    "buy_threshold",
    "stop_loss","use_take_profit","take_profit","trailing_stop",
    "use_rsi_sell","rsi_sell","use_macd_sell","use_kd_sell",
    "sell_vol_shrink","sell_below_ma","hold_days",
    "w_bias","bias_max",
    "use_stagnation_exit","stagnation_days","stagnation_min_ret",
    "use_breakeven","breakeven_trigger",
    "w_obv","obv_rising_days",
    "w_atr","atr_min",
    "use_time_decay","ret_per_day",
    "use_profit_lock","lock_trigger","lock_floor",
    "use_mom_exit","mom_exit_th",
    "max_positions",
]

MA_FAST_OPTS = [3,5,10]
MA_SLOW_OPTS = [15,20,30,60]
MOM_DAYS_OPTS = [3,5,10]

def precompute(data):
    tickers = list(data.keys())
    ml = min(len(data[t]) for t in tickers)
    n = len(tickers)
    close = np.zeros((n,ml),dtype=np.float32)
    high = np.zeros((n,ml),dtype=np.float32)
    low = np.zeros((n,ml),dtype=np.float32)
    opn = np.zeros((n,ml),dtype=np.float32)
    volume = np.zeros((n,ml),dtype=np.float32)
    dates = None
    for si,t in enumerate(tickers):
        h = data[t]
        close[si]=h["Close"].values[-ml:].astype(np.float32)
        high[si]=h["High"].values[-ml:].astype(np.float32)
        low[si]=h["Low"].values[-ml:].astype(np.float32)
        opn[si]=h["Open"].values[-ml:].astype(np.float32)
        volume[si]=h["Volume"].values[-ml:].astype(np.float32)
        if dates is None: dates=h.index[-ml:]

    # RSI
    delta=np.diff(close,axis=1); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    ag=np.zeros_like(close); al=np.zeros_like(close)
    for i in range(14,close.shape[1]):
        if i==14: ag[:,i]=np.mean(gain[:,:14],axis=1); al[:,i]=np.mean(loss[:,:14],axis=1)
        else: ag[:,i]=(ag[:,i-1]*13+gain[:,i-1])/14; al[:,i]=(al[:,i-1]*13+loss[:,i-1])/14
    rs=np.where(al>0,ag/al,100); rsi=(100-100/(1+rs)).astype(np.float32)

    # MA
    ma_d={}
    for w in [3,5,10,15,20,30,60]:
        ma=np.zeros_like(close)
        for i in range(w,close.shape[1]): ma[:,i]=np.mean(close[:,i-w:i],axis=1)
        ma_d[w]=ma.astype(np.float32)

    # BB
    bb_std=np.zeros_like(close)
    for i in range(20,close.shape[1]): bb_std[:,i]=np.std(close[:,i-20:i],axis=1)
    bb_u=ma_d[20]+2*bb_std; bb_l=ma_d[20]-2*bb_std; bb_r=bb_u-bb_l
    bb_pos=np.where(bb_r>0,(close-bb_l)/bb_r,0.5).astype(np.float32)

    # Vol ratio
    vol_ma=np.zeros_like(volume)
    for i in range(20,volume.shape[1]): vol_ma[:,i]=np.mean(volume[:,i-20:i],axis=1)
    vol_ratio=np.where(vol_ma>0,volume/vol_ma,1).astype(np.float32)
    vol_prev=np.zeros_like(vol_ratio); vol_prev[:,1:]=vol_ratio[:,:-1]

    # MACD
    e12=np.zeros_like(close); e26=np.zeros_like(close); e12[:,0]=close[:,0]; e26[:,0]=close[:,0]
    for i in range(1,close.shape[1]):
        e12[:,i]=e12[:,i-1]*(1-2/13)+close[:,i]*2/13
        e26[:,i]=e26[:,i-1]*(1-2/27)+close[:,i]*2/27
    ml_arr=(e12-e26).astype(np.float32)
    ms=np.zeros_like(close); ms[:,0]=ml_arr[:,0]
    for i in range(1,close.shape[1]): ms[:,i]=ms[:,i-1]*(1-2/10)+ml_arr[:,i]*2/10
    mh=(ml_arr-ms).astype(np.float32)

    # KD
    l9=np.zeros_like(close); h9=np.zeros_like(close)
    for i in range(9,close.shape[1]):
        l9[:,i]=np.min(low[:,i-9:i+1],axis=1); h9[:,i]=np.max(high[:,i-9:i+1],axis=1)
    rsv=np.where((h9-l9)>0,(close-l9)/(h9-l9)*100,50)
    kv=np.zeros_like(close); dv=np.zeros_like(close); kv[:,0]=50; dv[:,0]=50
    for i in range(1,close.shape[1]):
        kv[:,i]=kv[:,i-1]*2/3+rsv[:,i]*1/3; dv[:,i]=dv[:,i-1]*2/3+kv[:,i]*1/3

    # Momentum
    mom_d={}
    for d in [3,5,10]:
        m=np.zeros_like(close); m[:,d:]=(close[:,d:]/close[:,:-d]-1)*100
        mom_d[d]=m.astype(np.float32)

    ig=(close>opn).astype(np.float32)
    gp=np.zeros_like(close); gp[:,1:]=(opn[:,1:]/close[:,:-1]-1)*100

    # Williams %R
    l14=np.zeros_like(close); h14=np.zeros_like(close)
    for i in range(14,close.shape[1]):
        l14[:,i]=np.min(low[:,i-14:i+1],axis=1); h14[:,i]=np.max(high[:,i-14:i+1],axis=1)
    wr=np.where((h14-l14)>0,(h14-close)/(h14-l14)*-100,-50).astype(np.float32)

    h20=np.zeros_like(close)
    for i in range(20,close.shape[1]): h20[:,i]=np.max(high[:,i-20:i+1],axis=1)
    nh=np.where(h20>0,(close/h20-1)*100,0).astype(np.float32)

    # ATR (14) — for Keltner Channel
    tr = np.zeros_like(close)
    tr[:, 1:] = np.maximum(high[:, 1:] - low[:, 1:],
        np.maximum(np.abs(high[:, 1:] - close[:, :-1]), np.abs(low[:, 1:] - close[:, :-1])))
    atr = np.zeros_like(close)
    for i in range(1, close.shape[1]):
        if i <= 14: atr[:, i] = np.mean(tr[:, 1:min(i+1,15)], axis=1)
        else: atr[:, i] = (atr[:, i-1] * 13 + tr[:, i]) / 14

    # TTM Squeeze: BB inside Keltner Channel = squeeze ON, release = buy signal
    ema20 = ma_d[20]  # use MA20 as approximation
    kc_upper = ema20 + 1.5 * atr
    kc_lower = ema20 - 1.5 * atr
    bb_upper = ma_d[20] + 2 * bb_std
    bb_lower = ma_d[20] - 2 * bb_std
    squeeze_on = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(np.float32)
    # Squeeze fire: was ON yesterday, OFF today
    squeeze_fire = np.zeros_like(close)
    squeeze_fire[:, 1:] = ((squeeze_on[:, :-1] == 1) & (squeeze_on[:, 1:] == 0)).astype(np.float32)
    # Also require momentum positive (MACD histogram > 0) for quality
    squeeze_fire = (squeeze_fire * (mh > 0)).astype(np.float32)

    # 60-day new high (Taiwan effect: breaking historical high = strong momentum)
    h60 = np.zeros_like(close)
    for i in range(60, close.shape[1]):
        h60[:, i] = np.max(high[:, i-60:i], axis=1)  # previous 60 days high (not including today)
    new_high_60 = (close > h60).astype(np.float32)  # today's close > 60-day high = breakout

    # ADX (14) — 趨勢強度
    tr = np.zeros_like(close)
    tr[:, 1:] = np.maximum(high[:, 1:] - low[:, 1:],
        np.maximum(np.abs(high[:, 1:] - close[:, :-1]), np.abs(low[:, 1:] - close[:, :-1])))
    atr14 = np.zeros_like(close)
    plus_dm = np.zeros_like(close); minus_dm = np.zeros_like(close)
    for i in range(1, close.shape[1]):
        up = high[:, i] - high[:, i-1]
        dn = low[:, i-1] - low[:, i]
        plus_dm[:, i] = np.where((up > dn) & (up > 0), up, 0)
        minus_dm[:, i] = np.where((dn > up) & (dn > 0), dn, 0)
    smooth_plus = np.zeros_like(close); smooth_minus = np.zeros_like(close)
    for i in range(14, close.shape[1]):
        if i == 14:
            atr14[:, i] = np.mean(tr[:, 1:15], axis=1)
            smooth_plus[:, i] = np.mean(plus_dm[:, 1:15], axis=1)
            smooth_minus[:, i] = np.mean(minus_dm[:, 1:15], axis=1)
        else:
            atr14[:, i] = (atr14[:, i-1] * 13 + tr[:, i]) / 14
            smooth_plus[:, i] = (smooth_plus[:, i-1] * 13 + plus_dm[:, i]) / 14
            smooth_minus[:, i] = (smooth_minus[:, i-1] * 13 + minus_dm[:, i]) / 14
    plus_di = np.where(atr14 > 0, smooth_plus / atr14 * 100, 0)
    minus_di = np.where(atr14 > 0, smooth_minus / atr14 * 100, 0)
    dx = np.where((plus_di + minus_di) > 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(close)
    for i in range(28, close.shape[1]):
        if i == 28:
            adx[:, i] = np.mean(dx[:, 14:29], axis=1)
        else:
            adx[:, i] = (adx[:, i-1] * 13 + dx[:, i]) / 14

    # ATR% = ATR / close * 100（波動率，用於過濾低波動股）
    atr_pct = np.where(close > 0, atr / close * 100, 0).astype(np.float32)

    # BIAS 乖離率 = (close - MA20) / MA20 * 100
    bias = np.where(ma_d[20] > 0, (close - ma_d[20]) / ma_d[20] * 100, 0).astype(np.float32)

    # OBV (On Balance Volume) — 量價同步指標
    # OBV = cumsum(volume * sign(close_change))
    price_sign = np.sign(np.diff(close, axis=1))  # +1/-1/0
    vol_signed = np.zeros_like(close)
    vol_signed[:, 1:] = volume[:, 1:] * price_sign
    obv = np.cumsum(vol_signed, axis=1).astype(np.float64)
    # OBV rising = OBV 的短期 MA 在上升（今天 > N天前）
    obv_rising = np.zeros_like(close)
    for d in [3, 5, 10]:
        rising = np.zeros_like(close)
        rising[:, d:] = (obv[:, d:] > obv[:, :-d]).astype(np.float32)
        obv_rising += rising
    # 只要任一週期上升就算（簡化：用最寬鬆標準）
    obv_rising = (obv_rising > 0).astype(np.float32)

    return {"tickers":tickers,"dates":dates,"n_stocks":n,"n_days":ml,
        "close":close,"rsi":rsi,"bb_pos":bb_pos,"vol_ratio":vol_ratio,
        "macd_line":ml_arr,"macd_hist":mh,"k_val":kv.astype(np.float32),
        "d_val":dv.astype(np.float32),"is_green":ig,"gap":gp.astype(np.float32),
        "williams_r":wr,"near_high":nh,"vol_prev":vol_prev.astype(np.float32),
        "squeeze_fire":squeeze_fire,"new_high_60":new_high_60,
        "adx":adx.astype(np.float32),"bias":bias,"obv_rising":obv_rising,"atr_pct":atr_pct,
        "bb_std":bb_std.astype(np.float32),
        "ma_d":ma_d,"mom_d":mom_d,"ma60":ma_d[60]}

REASON_NAMES = ["到期","停利","停損","RSI超買","移動停利","MACD死叉","KD死叉","量縮","跌破均線","停滯出場","漸進停利","鎖利出場","動量反轉"]

def cpu_replay(pre, p):
    """用 CPU 重跑一次最佳參數，拿完整交易明細（股票名、日期、價格）"""
    ns, nd = pre["n_stocks"], pre["n_days"]
    tickers = pre["tickers"]; dates = pre["dates"]; close = pre["close"]
    rsi=pre["rsi"]; bb_pos=pre["bb_pos"]; vol_ratio=pre["vol_ratio"]
    macd_hist=pre["macd_hist"]; macd_line=pre["macd_line"]
    k_val=pre["k_val"]; d_val=pre["d_val"]; williams_r=pre["williams_r"]
    is_green=pre["is_green"]; gap=pre["gap"]; near_high=pre["near_high"]
    vol_prev=pre["vol_prev"]
    squeeze_fire=pre["squeeze_fire"]; new_high_60=pre["new_high_60"]; adx_arr=pre["adx"]; bias_arr=pre["bias"]; obv_rising_arr=pre["obv_rising"]; atr_pct_arr=pre["atr_pct"]
    maf=pre["ma_d"].get(int(p.get("ma_fast_w",5)), pre["ma_d"][5])
    mas=pre["ma_d"].get(int(p.get("ma_slow_w",20)), pre["ma_d"][20])
    ma60=pre["ma60"]
    mom=pre["mom_d"].get(int(p.get("momentum_days",5)), pre["mom_d"][5])

    max_pos=int(p.get("max_positions",1))
    if max_pos<1: max_pos=1
    if max_pos>3: max_pos=3
    hold_si=[-1]*3; hold_bp=[0]*3; hold_pk=[0]*3; hold_bd=[0]*3; n_holding=0; trades=[]
    for day in range(30, nd-1):
        any_sold=False
        for h in range(max_pos):
            if hold_si[h]<0: continue
            si=hold_si[h]; cur=float(close[si,day]); dh=day-hold_bd[h]
            ret=(cur/hold_bp[h]-1)*100
            if dh<1: continue
            if cur>hold_pk[h]: hold_pk[h]=cur
            sell=False; reason=0
            eff_stop=p["stop_loss"]
            if p.get("use_breakeven",0) and (hold_pk[h]/hold_bp[h]-1)*100>=p.get("breakeven_trigger",20): eff_stop=0
            if ret<=eff_stop: sell=True; reason=2
            if not sell and p.get("use_take_profit",1) and ret>=p["take_profit"]: sell=True; reason=1
            if not sell and p.get("trailing_stop",0)>0 and hold_pk[h]>hold_bp[h]:
                if (cur/hold_pk[h]-1)*100<=-p["trailing_stop"]: sell=True; reason=4
            if not sell and p.get("use_rsi_sell",1) and rsi[si,day]>=p.get("rsi_sell",90): sell=True; reason=3
            if not sell and p.get("use_macd_sell",0) and day>=1:
                if macd_hist[si,day]<0 and macd_hist[si,day-1]>=0: sell=True; reason=5
            if not sell and p.get("use_kd_sell",0) and day>=1:
                if k_val[si,day]<d_val[si,day] and k_val[si,day-1]>=d_val[si,day-1]: sell=True; reason=6
            if not sell and p.get("sell_vol_shrink",0)>0 and dh>=2 and vol_ratio[si,day]<p["sell_vol_shrink"]: sell=True; reason=7
            sbm=int(p.get("sell_below_ma",0))
            if not sell and sbm>0:
                mc=0
                if sbm==1: mc=maf[si,day]
                elif sbm==2: mc=mas[si,day]
                elif sbm==3: mc=ma60[si,day]
                if mc>0 and cur<mc: sell=True; reason=8
            if not sell and p.get("use_stagnation_exit",0) and dh>=int(p.get("stagnation_days",10)) and ret<p.get("stagnation_min_ret",5): sell=True; reason=9
            hd_half=int(p["hold_days"])//2
            if not sell and p.get("use_time_decay",0) and dh>=hd_half and ret<(dh-hd_half)*p.get("ret_per_day",0.5): sell=True; reason=10
            if not sell and p.get("use_profit_lock",0):
                pg=(hold_pk[h]/hold_bp[h]-1)*100
                if pg>=p.get("lock_trigger",30) and ret<p.get("lock_floor",10): sell=True; reason=11
            if not sell and p.get("use_mom_exit",0) and dh>=10 and mom[si,day]<-p.get("mom_exit_th",2): sell=True; reason=12
            if not sell and dh>=int(p["hold_days"]): sell=True; reason=0
            if sell:
                trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                    "buy_date":str(dates[hold_bd[h]].date()),"sell_date":str(dates[day].date()),
                    "buy_price":round(hold_bp[h],2),"sell_price":round(cur,2),
                    "return":round(ret,2),"days":dh,"reason":REASON_NAMES[min(reason,len(REASON_NAMES)-1)]})
                hold_si[h]=-1; n_holding-=1; any_sold=True
        if n_holding<max_pos and not any_sold:
            best_si=-1; best_sc=0
            w_rsi=int(p.get("w_rsi",0)); w_bb=int(p.get("w_bb",0)); w_vol=int(p.get("w_vol",0))
            w_ma=int(p.get("w_ma",0)); w_macd=int(p.get("w_macd",0)); w_kd=int(p.get("w_kd",0))
            w_wr=int(p.get("w_wr",0)); w_mom=int(p.get("w_mom",0)); w_nh=int(p.get("w_near_high",0))
            w_sq=int(p.get("w_squeeze",0)); w_newh=int(p.get("w_new_high",0))
            w_adx=int(p.get("w_adx",0)); adx_threshold=p.get("adx_th",25)
            w_bias=int(p.get("w_bias",0)); bias_max_val=p.get("bias_max",15)
            w_obv=int(p.get("w_obv",0))
            w_atr_buy=int(p.get("w_atr",0)); atr_min_val=p.get("atr_min",2.0)
            buy_th=p.get("buy_threshold",5)
            held_set=set(h for h in hold_si if h>=0)
            for si in range(ns):
                if si in held_set: continue
                sc=0.0
                if w_rsi>0 and rsi[si,day]>=p.get("rsi_th",55): sc+=w_rsi
                if w_bb>0 and bb_pos[si,day]>=p.get("bb_th",0.7): sc+=w_bb
                if w_vol>0 and vol_ratio[si,day]>=p.get("vol_th",3): sc+=w_vol
                if w_ma>0 and close[si,day]>maf[si,day]: sc+=w_ma
                if w_macd>0:
                    mm=int(p.get("macd_mode",2)); ok=False
                    if mm==0 and day>=1 and macd_hist[si,day]>0 and macd_hist[si,day-1]<=0: ok=True
                    elif mm==1 and macd_line[si,day]>0: ok=True
                    elif mm==2 and macd_hist[si,day]>0: ok=True
                    if ok: sc+=w_macd
                if w_kd>0:
                    ok=k_val[si,day]>=p.get("kd_th",50)
                    if ok and p.get("kd_cross",0) and day>=1:
                        ok=k_val[si,day]>d_val[si,day] and k_val[si,day-1]<=d_val[si,day-1]
                    if ok: sc+=w_kd
                if w_wr>0 and williams_r[si,day]>=p.get("wr_th",-30): sc+=w_wr
                if w_mom>0 and mom[si,day]>=p.get("mom_th",3): sc+=w_mom
                if w_nh>0 and abs(near_high[si,day])<=p.get("near_high_pct",10): sc+=w_nh
                if w_sq>0 and squeeze_fire[si,day]>0.5: sc+=w_sq
                if w_newh>0 and new_high_60[si,day]>0.5: sc+=w_newh
                if w_adx>0 and adx_arr[si,day]>=adx_threshold: sc+=w_adx
                if w_bias>0 and bias_arr[si,day]>=0 and bias_arr[si,day]<=bias_max_val: sc+=w_bias
                if w_obv>0 and obv_rising_arr[si,day]>0.5: sc+=w_obv
                if w_atr_buy>0 and atr_pct_arr[si,day]>=atr_min_val: sc+=w_atr_buy
                cg=int(p.get("consecutive_green",0))
                if cg>=1:
                    ok=True
                    for g in range(cg):
                        if day-g<0 or is_green[si,day-g]!=1: ok=False; break
                    if ok: sc+=1
                if p.get("gap_up",0) and gap[si,day]>=1.0: sc+=1
                if p.get("above_ma60",0) and close[si,day]>=ma60[si,day]: sc+=1
                if p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]>vol_prev[si,day]: sc+=1
                if sc>=buy_th and sc>best_sc: best_si=si; best_sc=sc
            if best_si>=0 and day+1<nd:
                for h in range(max_pos):
                    if hold_si[h]<0:
                        hold_si[h]=best_si; hold_bp[h]=float(close[best_si,day+1])
                        hold_pk[h]=hold_bp[h]; hold_bd[h]=day+1; n_holding+=1; break
    return sorted(trades, key=lambda x: x["buy_date"])

def main():
    print("[GPU-CuPy] 🚀 RTX 3060 進化引擎啟動！")
    raw = download_data()
    # 過濾掉資料太短的（至少 200 天），避免拖累整體天數
    # 過濾資料太短的，再取成交量前 300 名（平衡速度和覆蓋率）
    valid = {k:v for k,v in raw.items() if len(v) >= 400}
    vol_rank = {}
    for t, h in valid.items():
        if "Volume" in h.columns and len(h) >= 20:
            vol_rank[t] = h["Volume"].tail(20).mean()
    top = sorted(vol_rank, key=vol_rank.get, reverse=True)[:300]
    data = {k: valid[k] for k in top}
    print(f"[過濾] {len(raw)} → {len(valid)} (>=400天) → {len(data)} 檔 (成交量前300)")
    if len(data) < 10: print("資料不足"); return
    pre = precompute(data)
    ns, nd = pre["n_stocks"], pre["n_days"]
    print(f"[GPU] {ns}檔 x {nd}天 | 資料載入 GPU...")

    # 傳資料到 GPU（一次性）
    d_close = cp.asarray(pre["close"])
    d_rsi = cp.asarray(pre["rsi"])
    d_bb = cp.asarray(pre["bb_pos"])
    d_vr = cp.asarray(pre["vol_ratio"])
    d_mh = cp.asarray(pre["macd_hist"])
    d_ml = cp.asarray(pre["macd_line"])
    d_kv = cp.asarray(pre["k_val"])
    d_dv = cp.asarray(pre["d_val"])
    d_ig = cp.asarray(pre["is_green"])
    d_gp = cp.asarray(pre["gap"])
    d_wr = cp.asarray(pre["williams_r"])
    d_nh = cp.asarray(pre["near_high"])
    d_vp = cp.asarray(pre["vol_prev"])
    d_squeeze = cp.asarray(pre["squeeze_fire"])
    d_newhigh = cp.asarray(pre["new_high_60"])
    d_adx = cp.asarray(pre["adx"])
    d_bias = cp.asarray(pre["bias"])
    d_obv_rising = cp.asarray(pre["obv_rising"])
    d_atr_pct = cp.asarray(pre["atr_pct"])
    d_ma60 = cp.asarray(pre["ma60"])

    print("[GPU] 開始進化！每批 500,000 組")
    BATCH = 200000  # 縮小避免 Python 參數生成卡住
    BLOCK = 256
    N_PARAMS = len(PARAM_ORDER)
    best_score = -999999
    best_params = None
    best_avg = 0; best_total = 0; best_wr = 0; best_nt = 0
    total_tested = 0; total_improved = 0; last_synced_improved = 0
    start = time.time()
    rnd = 0
    # Top 5 名人堂（交叉配種用）
    hall_of_fame = []  # [(score, params_dict), ...]
    no_improve_rounds = 0  # 連續沒突破的輪數
    explore_bases = None  # 多起點爬山用的隨機起點
    explore_round = 0

    # 所有 MA/MOM 陣列傳到 GPU（一次性）
    d_ma3 = cp.asarray(pre["ma_d"][3]); d_ma5 = cp.asarray(pre["ma_d"][5])
    d_ma10 = cp.asarray(pre["ma_d"][10]); d_ma15 = cp.asarray(pre["ma_d"][15])
    d_ma20 = cp.asarray(pre["ma_d"][20]); d_ma30 = cp.asarray(pre["ma_d"][30])
    d_mom3 = cp.asarray(pre["mom_d"][3]); d_mom5 = cp.asarray(pre["mom_d"][5])
    d_mom10 = cp.asarray(pre["mom_d"][10])

    # MA/MOM 的對應 index
    MA_FAST_MAP = {3:0, 5:1, 10:2}
    MA_SLOW_MAP = {15:0, 20:1, 30:2, 60:3}
    MOM_MAP = {3:0, 5:1, 10:2}
    N_PARAMS_FULL = N_PARAMS + 3  # 加 ma_fast_idx, ma_slow_idx, mom_idx

    # 嘗試從 Gist 載入最佳策略作為爬山起點
    gist_best_params = None
    try:
        headers = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
        r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=headers, timeout=10)
        gist_data = json.loads(list(r.json()["files"].values())[0]["content"])
        if gist_data.get("score", 0) > 0 and "params" in gist_data:
            gist_best_params = gist_data["params"]
            best_score = gist_data["score"]
            print(f"[GPU] 載入 Gist 最佳策略（{best_score:.2f}）作為爬山起點")
    except: pass

    last_data_date = time.strftime("%Y-%m-%d")

    while True:
        # 每天自動刷新資料（偵測到日期變了就重下載，不用重啟）
        today_str = time.strftime("%Y-%m-%d")
        if today_str != last_data_date:
            print(f"\n[GPU] 🔄 新的一天（{today_str}），自動刷新資料...")
            try:
                if os.path.exists(CACHE_PATH): os.remove(CACHE_PATH)
                raw = download_data()
                valid = {k:v for k,v in raw.items() if len(v) >= 400}
                vol_rank = {}
                for t, h in valid.items():
                    if "Volume" in h.columns and len(h) >= 20:
                        vol_rank[t] = h["Volume"].tail(20).mean()
                top = sorted(vol_rank, key=vol_rank.get, reverse=True)[:300]
                data = {k: valid[k] for k in top}
                print(f"[GPU] 刷新完成：{len(data)} 檔")
                pre = precompute(data)
                ns, nd = pre["n_stocks"], pre["n_days"]
                d_close = cp.asarray(pre["close"]); d_rsi = cp.asarray(pre["rsi"])
                d_bb = cp.asarray(pre["bb_pos"]); d_vr = cp.asarray(pre["vol_ratio"])
                d_mh = cp.asarray(pre["macd_hist"]); d_ml = cp.asarray(pre["macd_line"])
                d_kv = cp.asarray(pre["k_val"]); d_dv = cp.asarray(pre["d_val"])
                d_ig = cp.asarray(pre["is_green"]); d_gp = cp.asarray(pre["gap"])
                d_wr = cp.asarray(pre["williams_r"]); d_nh = cp.asarray(pre["near_high"])
                d_vp = cp.asarray(pre["vol_prev"])
                d_squeeze = cp.asarray(pre["squeeze_fire"]); d_newhigh = cp.asarray(pre["new_high_60"])
                d_adx = cp.asarray(pre["adx"]); d_bias = cp.asarray(pre["bias"])
                d_ma60 = cp.asarray(pre["ma60"])
                d_ma3 = cp.asarray(pre["ma_d"][3]); d_ma5 = cp.asarray(pre["ma_d"][5])
                d_ma10 = cp.asarray(pre["ma_d"][10]); d_ma15 = cp.asarray(pre["ma_d"][15])
                d_ma20 = cp.asarray(pre["ma_d"][20]); d_ma30 = cp.asarray(pre["ma_d"][30])
                d_mom3 = cp.asarray(pre["mom_d"][3]); d_mom5 = cp.asarray(pre["mom_d"][5])
                d_mom10 = cp.asarray(pre["mom_d"][10])
                print(f"[GPU] {ns}檔 x {nd}天 | GPU 陣列已更新，繼續進化！")
                last_data_date = today_str
            except Exception as e:
                print(f"[GPU] 刷新失敗：{e}，沿用舊資料繼續跑")
                last_data_date = today_str  # 避免重複嘗試

        rnd += 1
        params_np = np.zeros((BATCH, N_PARAMS_FULL), dtype=np.float32)
        mutate_rate = min(0.5, 0.15 + no_improve_rounds * 0.02)
        # 多起點爬山：80% 爬山（從隨機起點）+ 20% 隨機，不配種（名人堂還在重建）
        if explore_bases is not None:
            n_random = BATCH // 5
            n_climb = BATCH - n_random
            n_breed = 0
        else:
            # 正常模式：20% 隨機 / 50% 爬山 / 30% 配種
            n_random = BATCH // 5
            n_climb = BATCH // 2
            n_breed = BATCH - n_random - n_climb
        third = n_random  # 相容舊變數名

        # === 全部先用隨機填滿（向量化，超快）===
        for i, key in enumerate(PARAM_ORDER):
            params_np[:, i] = np.random.choice(PARAMS_SPACE[key], BATCH).astype(np.float32)

        # === 爬山微調（向量化）===
        if explore_bases is not None:
            base = explore_bases[explore_round % len(explore_bases)]
        else:
            base = best_params if best_params else gist_best_params
        if base:
            for i, key in enumerate(PARAM_ORDER):
                opts = np.array(PARAMS_SPACE[key], dtype=np.float32)
                base_val = float(base.get(key, opts[0]))
                diffs = np.abs(opts - base_val)
                base_idx = int(np.argmin(diffs))
                keep = np.random.random(n_climb) >= mutate_rate
                lo = max(0, base_idx - 2)
                hi = min(len(opts) - 1, base_idx + 2)
                nearby = np.random.randint(lo, hi + 1, n_climb)
                vals = opts[nearby]
                vals[keep] = opts[base_idx]
                params_np[n_random:n_random+n_climb, i] = vals

        # === 真正的雙親交叉配種（向量化）===
        if len(hall_of_fame) >= 2:
            n_hof = len(hall_of_fame)
            breed_size = n_breed
            # 預建親代參數矩陣
            hof_matrix = np.zeros((n_hof, len(PARAM_ORDER)), dtype=np.float32)
            for h in range(n_hof):
                for i, key in enumerate(PARAM_ORDER):
                    opts = PARAMS_SPACE[key]
                    val = float(hall_of_fame[h][1].get(key, opts[0]))
                    diffs = [abs(float(o) - val) for o in opts]
                    hof_matrix[h, i] = float(opts[diffs.index(min(diffs))])
            # 選兩個不同親代
            parent_a = np.random.randint(n_hof, size=breed_size)
            parent_b = np.random.randint(n_hof, size=breed_size)
            same = parent_a == parent_b
            parent_b[same] = (parent_b[same] + 1) % n_hof
            # 每個參數 50/50 從親代 A 或 B 選
            from_a = np.random.random((breed_size, len(PARAM_ORDER))) < 0.5
            offspring = np.where(from_a, hof_matrix[parent_a], hof_matrix[parent_b])
            # 10% 突變
            mutate_mask = np.random.random((breed_size, len(PARAM_ORDER))) < 0.1
            for i, key in enumerate(PARAM_ORDER):
                opts = np.array(PARAMS_SPACE[key], dtype=np.float32)
                random_vals = opts[np.random.randint(len(opts), size=breed_size)]
                offspring[:, i] = np.where(mutate_mask[:, i], random_vals, offspring[:, i])
                params_np[n_random+n_climb:, i] = offspring[:, i]

        # MA/MOM 選擇（向量化）
        mf_choices = np.random.choice(MA_FAST_OPTS, BATCH)
        ms_choices = np.random.choice(MA_SLOW_OPTS, BATCH)
        md_choices = np.random.choice(MOM_DAYS_OPTS, BATCH)
        if base:
            keep_ma = np.random.random(BATCH) < 0.8
            base_mf = int(base.get("ma_fast_w", 5))
            base_ms = int(base.get("ma_slow_w", 20))
            base_md = int(base.get("momentum_days", 5))
            mf_choices[n_random:] = np.where(keep_ma[n_random:], base_mf, mf_choices[n_random:])
            ms_choices[n_random:] = np.where(keep_ma[n_random:], base_ms, ms_choices[n_random:])
            md_choices[n_random:] = np.where(keep_ma[n_random:], base_md, md_choices[n_random:])
        # 過濾 ma_fast >= ma_slow（向量化）
        bad = mf_choices >= ms_choices
        ms_choices[bad] = max(MA_SLOW_OPTS)
        mf_mapped = np.vectorize(MA_FAST_MAP.get)(mf_choices)
        ms_mapped = np.vectorize(MA_SLOW_MAP.get)(ms_choices)
        md_mapped = np.vectorize(MOM_MAP.get)(md_choices)
        params_np[:, N_PARAMS] = mf_mapped.astype(np.float32)
        params_np[:, N_PARAMS+1] = ms_mapped.astype(np.float32)
        params_np[:, N_PARAMS+2] = md_mapped.astype(np.float32)

        d_params = cp.asarray(params_np)
        d_results = cp.zeros((BATCH, 5), dtype=cp.float32)
        grid = (BATCH + BLOCK - 1) // BLOCK

        CUDA_KERNEL((grid,), (BLOCK,), (
            np.int32(ns), np.int32(nd),
            d_close, d_rsi, d_bb, d_vr, d_mh, d_ml,
            d_kv, d_dv, d_mom3, d_mom5, d_mom10,
            d_ig, d_gp, d_nh, d_wr,
            d_ma3, d_ma5, d_ma10, d_ma15, d_ma20, d_ma30, d_ma60,
            d_vp, d_squeeze, d_newhigh, d_adx, d_bias, d_obv_rising, d_atr_pct,
            d_params, np.int32(N_PARAMS_FULL),
            d_results, np.int32(BATCH)
        ))

        results = d_results.get()
        total_tested += BATCH

        # 收集這批裡分數 > 0 的前 5 名加入名人堂（不用破紀錄也能入）
        top_indices = np.argsort(results[:, 0])[-5:][::-1]
        for ti in top_indices:
            sc = float(results[ti, 0])
            if sc > 0:
                tp = params_np[ti]
                tp_dict = {PARAM_ORDER[i]: float(tp[i]) for i in range(N_PARAMS)}
                tp_dict["ma_fast_w"] = int(mf_choices[ti])
                tp_dict["ma_slow_w"] = int(ms_choices[ti])
                tp_dict["momentum_days"] = int(md_choices[ti])
                hall_of_fame.append((sc, tp_dict))
        hall_of_fame.sort(key=lambda x: -x[0])
        hall_of_fame = hall_of_fame[:5]

        bi = np.argmax(results[:, 0])
        if results[bi, 0] > best_score:
            best_score = float(results[bi, 0])
            best_nt = int(results[bi, 1])
            best_avg = float(results[bi, 2])
            best_total = float(results[bi, 3])
            best_wr = float(results[bi, 4])
            bp = params_np[bi]
            best_params = {PARAM_ORDER[i]: float(bp[i]) for i in range(N_PARAMS)}
            best_params["ma_fast_w"] = int(mf_choices[bi])
            best_params["ma_slow_w"] = int(ms_choices[bi])
            best_params["momentum_days"] = int(md_choices[bi])
            total_improved += 1
            no_improve_rounds = 0
            print(f"  [GPU] 新紀錄！{best_score:.1f} | 勝率{best_wr:.0f}% | 平均{best_avg:.1f}% | {best_nt}筆 | 名人堂Top:{hall_of_fame[0][0]:.1f}")
        else:
            no_improve_rounds += 1
            # 變異率到頂 = 爬山已退化成亂射，啟動多起點爬山
            if mutate_rate >= 0.50:
                hall_of_fame = []
                no_improve_rounds = 0
                # 從最佳策略的「遠親」開始爬（打亂 60% 參數，保留 40% 骨架）
                anchor = best_params if best_params else gist_best_params
                explore_bases = []
                for _eb in range(5):
                    if anchor:
                        rb = {}
                        for key in PARAM_ORDER:
                            if np.random.random() < 0.6:  # 60% 打亂
                                rb[key] = float(np.random.choice(PARAMS_SPACE[key]))
                            else:  # 40% 保留
                                rb[key] = float(anchor.get(key, np.random.choice(PARAMS_SPACE[key])))
                        rb["ma_fast_w"] = anchor.get("ma_fast_w", 5) if np.random.random() < 0.4 else int(np.random.choice(MA_FAST_OPTS))
                        rb["ma_slow_w"] = anchor.get("ma_slow_w", 20) if np.random.random() < 0.4 else int(np.random.choice(MA_SLOW_OPTS))
                        rb["momentum_days"] = anchor.get("momentum_days", 5) if np.random.random() < 0.4 else int(np.random.choice(MOM_DAYS_OPTS))
                    else:
                        rb = {key: float(np.random.choice(PARAMS_SPACE[key])) for key in PARAM_ORDER}
                        rb["ma_fast_w"] = int(np.random.choice(MA_FAST_OPTS))
                        rb["ma_slow_w"] = int(np.random.choice(MA_SLOW_OPTS))
                        rb["momentum_days"] = int(np.random.choice(MOM_DAYS_OPTS))
                    explore_bases.append(rb)
                explore_round = 0
                print(f"  [GPU] 🔄 多起點爬山！5 個遠親起點（60%打亂+40%骨架）各爬 3 輪")

        # 多起點爬山進度
        if explore_bases is not None:
            explore_round += 1
            if explore_round >= 15:  # 5 起點 × 3 輪
                explore_bases = None
                no_improve_rounds = 0  # 歸零，變異率從 0.15 重新開始
                print(f"  [GPU] ✅ 多起點爬山完成，名人堂已種好，變異率歸零，恢復正常模式")

        elapsed = time.time() - start
        speed = total_tested / elapsed
        explore_tag = f" | 探索{explore_round}/15" if explore_bases is not None else ""
        print(f"[GPU] R{rnd} | {total_tested:,}組 | {elapsed:.0f}秒 | {speed:,.0f}組/秒 | 突破{total_improved} | 變異率{mutate_rate:.0%}{explore_tag}")

        # Gist 同步（只在這輪有突破時才檢查）
        if total_improved > last_synced_improved and best_score > 0 and GH_TOKEN and GIST_ID:
            try:
                headers = {"Authorization": f"token {GH_TOKEN}"}
                r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=headers, timeout=10)
                cs = json.loads(list(r.json()["files"].values())[0]["content"]).get("score", 0)
                if best_score > cs + 0.01:
                    # CPU 重跑一次拿完整交易明細
                    trade_details = cpu_replay(pre, best_params)
                    trade_lines = "\n".join([
                        f"  {t['name']}({t['ticker'].replace('.TWO','').replace('.TW','')}) | {t['buy_date'][5:]}→{t['sell_date'][5:]} | {t['return']:+.1f}% | {t['days']}天 | {t['reason']}"
                        for t in trade_details
                    ])
                    content = json.dumps({"score":round(best_score,4),"source":"gpu_rtx3060",
                        "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "params":best_params,"backtest":{
                            "avg_return":round(best_avg,2),"total_return":round(best_total,2),
                            "win_rate":round(best_wr,2),"total_trades":best_nt},
                        "trade_details":trade_details},
                        ensure_ascii=False, indent=2)
                    requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=headers,
                        json={"files":{"best_strategy.json":{"content":content}}}, timeout=10)
                    telegram_push(
                        f"🎮 GPU RTX 3060 突破！\n"
                        f"━━━━━━━━━━━━\n"
                        f"分數：{best_score:.2f} > {cs:.2f}\n"
                        f"平均報酬：{best_avg:.1f}%\n"
                        f"總報酬：{best_total:.0f}%\n"
                        f"勝率：{best_wr:.0f}% | {best_nt}筆\n"
                        f"⚡ {total_tested:,}組/{elapsed:.0f}秒/{speed:,.0f}組/秒\n\n"
                        f"📋 交易明細：\n{trade_lines}"
                    )
                    print(f"  [GPU] ✅ Gist 同步！({best_score:.2f} > {cs:.2f})")
                last_synced_improved = total_improved
            except Exception as e:
                print(f"  [GPU] Gist 錯誤: {e}")

if __name__ == "__main__":
    main()
