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
    "2801.TW": "彰銀", "2834.TW": "臺企銀", "2883.TW": "開發金",
    "2884.TW": "玉山金", "2887.TW": "台新金", "2892.TW": "第一金", "3189.TW": "景碩",
    "2327.TW": "國巨", "1326.TW": "台化", "3008.TW": "大立光", "1402.TW": "遠東新",
    "1590.TW": "亞德客-KY", "5871.TW": "中租-KY", "2395.TW": "研華", "2379.TW": "瑞昱",
    "6239.TW": "力成", "3044.TW": "健鼎", "2474.TW": "可成", "3443.TW": "創意",
    "2345.TW": "智邦", "2449.TW": "京元電子", "1102.TW": "亞泥", "5880.TW": "合庫金",
    "3017.TW": "奇鋐", "6669.TW": "緯穎", "3034.TW": "聯詠", "3661.TW": "世芯-KY",
    "2634.TW": "漢翔", "1513.TW": "中興電", "1504.TW": "東元", "2049.TW": "上銀",
    "1476.TW": "儒鴻", "9910.TW": "豐泰", "2207.TW": "和泰車", "2368.TW": "金像電",
    "2383.TW": "台光電", "1312.TW": "國喬", "1314.TW": "中石化", "2605.TW": "新興",
    "8454.TW": "富邦媒", "2542.TW": "興富發", "2404.TW": "漢唐", "1210.TW": "大成",
    "1227.TW": "佳格", "9933.TW": "中鼎", "5876.TW": "上海商銀", "9921.TW": "巨大",
}
def get_name(t): return CN_NAMES.get(t, t.replace(".TW",""))
def auto_fetch_names(tickers):
    missing = [t for t in tickers if t not in CN_NAMES]
    if not missing: return
    for t in missing:
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName","") or info.get("longName","")
            if name and not name[0].isdigit(): CN_NAMES[t] = name
        except: pass
def telegram_push(msg):
    for cid in CHAT_IDS:
        cid = cid.strip()
        if not cid or not BOT_TOKEN: continue
        try: requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json={"chat_id":cid,"text":msg}, timeout=10)
        except: pass

# === 資料載入 ===
import yfinance as yf
TW_TICKERS = [
    "2330.TW","2454.TW","2317.TW","2303.TW","2382.TW","3231.TW","2353.TW","2357.TW",
    "2881.TW","2882.TW","2891.TW","2886.TW","2412.TW","1301.TW","2603.TW","2609.TW",
    "1216.TW","2002.TW","2308.TW","3711.TW","2409.TW","3481.TW","2356.TW","2324.TW",
    "4938.TW","2337.TW","2344.TW","3037.TW","6770.TW","3576.TW","1802.TW","8039.TW",
    "2485.TW","1711.TW","1717.TW","6505.TW","1303.TW","2406.TW","8150.TW",
    "2615.TW","2618.TW","2610.TW","2912.TW","1101.TW","2880.TW","2885.TW","2890.TW",
    "2801.TW","2834.TW","2883.TW","2884.TW","2887.TW","2892.TW","3189.TW","2301.TW",
    "2408.TW","3008.TW","2345.TW","3443.TW","2474.TW",
]

def download_data():
    if os.path.exists(CACHE_PATH):
        age = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age < 24:
            try:
                with open(CACHE_PATH, "rb") as f: data = pickle.load(f)
                if len(data) >= 10: print(f"[快取] {len(data)} 檔"); return data
            except: pass
    data = {}
    for i, t in enumerate(TW_TICKERS):
        try:
            h = yf.Ticker(t).history(period="2y")
            if len(h) >= 40: data[t] = h
            if i % 5 == 4: time.sleep(1)
        except: continue
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
    const float* params, const int n_params_per_combo,
    float* results, const int n_combos
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_combos) return;

    // 讀參數
    const float* p = params + idx * n_params_per_combo;
    int use_rsi_buy = (int)p[0]; float rsi_buy = p[1];
    int use_bb_buy = (int)p[2]; float bb_buy = p[3];
    int use_vol = (int)p[4]; float vol_th = p[5];
    int require_ma_bull = (int)p[6];
    int use_macd = (int)p[7]; int macd_mode = (int)p[8];
    int use_kd = (int)p[9]; float kd_k_th = p[10]; int kd_cross = (int)p[11];
    int use_wr = (int)p[12]; float wr_th = p[13];
    float mom_min = p[14]; int consec_green = (int)p[15];
    int use_gap = (int)p[16]; float near_high_pct = p[17];
    int above_ma60 = (int)p[18]; int require_ma_cross = (int)p[19];
    int vol_gt_yesterday = (int)p[20];
    float stop_loss = p[21]; int use_tp = (int)p[22]; float take_profit = p[23];
    float trailing_stop = p[24];
    int use_rsi_sell = (int)p[25]; float rsi_sell_th = p[26];
    int use_macd_sell = (int)p[27]; int use_kd_sell = (int)p[28];
    float sell_vol_shrink = p[29]; int sell_below_ma = (int)p[30];
    int hold_days_max = (int)p[31];
    int ma_fast_idx = (int)p[32];  // 0=ma3, 1=ma5, 2=ma10
    int ma_slow_idx = (int)p[33];  // 0=ma15, 1=ma20, 2=ma30, 3=ma60
    int mom_idx = (int)p[34];      // 0=mom3, 1=mom5, 2=mom10

    // 選均線和動量陣列指標
    const float* ma_fast_arr = ma_fast_idx==0 ? ma3 : ma_fast_idx==1 ? ma5 : ma10;
    const float* ma_slow_arr = ma_slow_idx==0 ? ma15 : ma_slow_idx==1 ? ma20 : ma_slow_idx==2 ? ma30 : ma60;
    const float* momentum = mom_idx==0 ? mom3 : mom_idx==1 ? mom5 : mom10;

    // 交易模擬
    int holding = -1;
    float buy_price = 0, peak_price = 0;
    int buy_day = 0, n_trades = 0;
    float total_ret = 0, win_count = 0, wasted_count = 0;
    float rets[100];
    int trade_bdays[100];

    for (int day = 30; day < n_days - 1; day++) {
        if (holding >= 0) {
            int si = holding;
            float cur = close[si * n_days + day];
            int dh = day - buy_day;
            float ret = (cur / buy_price - 1.0f) * 100.0f;
            if (dh < 1) continue;
            if (cur > peak_price) peak_price = cur;
            bool sell = false;

            if (ret <= stop_loss) sell = true;
            if (!sell && use_tp == 1 && ret >= take_profit) sell = true;
            if (!sell && trailing_stop > 0 && peak_price > buy_price) {
                if ((cur / peak_price - 1.0f) * 100.0f <= -trailing_stop) sell = true;
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
            if (!sell && dh >= hold_days_max) sell = true;

            if (sell && n_trades < 100) {
                rets[n_trades] = ret;
                trade_bdays[n_trades] = buy_day;
                total_ret += ret;
                if (ret > 0) win_count += 1;
                if (ret < 5) wasted_count += 1;
                n_trades++;
                holding = -1;
            }
            continue;
        }

        // 找買入
        int best_si = -1;
        float best_vol = 0;
        for (int si = 0; si < n_stocks; si++) {
            int d = si * n_days + day;
            bool buy = true;
            if (buy && use_rsi_buy == 1 && rsi[d] < rsi_buy) buy = false;
            if (buy && use_bb_buy == 1 && bb_pos[d] < bb_buy) buy = false;
            if (buy && use_vol == 1 && vol_ratio[d] < vol_th) buy = false;
            if (buy && require_ma_bull == 1 && close[d] < ma_fast_arr[d]) buy = false;
            if (buy && use_macd == 1) {
                if (macd_mode == 0 && !(macd_hist[d] > 0 && macd_hist[d-1] <= 0)) buy = false;
                else if (macd_mode == 1 && macd_line[d] <= 0) buy = false;
            }
            if (buy && use_kd == 1) {
                if (k_val[d] < kd_k_th) buy = false;
                if (buy && kd_cross == 1 && day >= 1) {
                    if (!(k_val[d] > d_val[d] && k_val[d-1] <= d_val[d-1])) buy = false;
                }
            }
            if (buy && use_wr == 1 && williams_r[d] < wr_th) buy = false;
            if (buy && mom_min > 0 && momentum[d] < mom_min) buy = false;
            if (buy && consec_green >= 1) {
                for (int g = 0; g < consec_green; g++) {
                    if (day-g < 0 || is_green[si*n_days+day-g] != 1) { buy = false; break; }
                }
            }
            if (buy && use_gap == 1 && gap[d] < 1.0f) buy = false;
            if (buy && near_high_pct > 0 && fabsf(near_high[d]) > near_high_pct) buy = false;
            if (buy && above_ma60 == 1 && close[d] < ma60[d]) buy = false;
            if (buy && require_ma_cross == 1 && ma_fast_arr[d] < ma_slow_arr[d]) buy = false;
            if (buy && vol_gt_yesterday == 1 && day >= 1 && vol_ratio[d] <= vol_prev[d]) buy = false;
            if (buy && vol_ratio[d] > best_vol) { best_si = si; best_vol = vol_ratio[d]; }
        }
        if (best_si >= 0 && day + 1 < n_days) {
            holding = best_si;
            buy_price = close[best_si * n_days + day + 1];
            peak_price = buy_price;
            buy_day = day + 1;
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
                          + n_trades*0.5f*0.10f - wasted*0.20f;
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
    "use_rsi_buy": [0,1], "rsi_buy": [35,40,45,50,55,60,65,70,75],
    "use_bb_buy": [0,1], "bb_buy": [0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    "use_vol_filter": [0,1], "vol_filter": [1.5,2.0,2.5,3.0,4.0,5.0],
    "require_ma_bull": [0,1],
    "use_macd": [0,1], "macd_mode": [0,1,2],
    "use_kd": [0,1], "kd_buy_k": [20,30,40,50,60,70,80], "kd_cross": [0,1],
    "use_wr_buy": [0,1], "wr_buy": [-10,-20,-30,-40,-50],
    "momentum_min": [0,3,8],
    "consecutive_green": [0,1,2,3], "gap_up": [0,1],
    "near_high_pct": [0,5,10], "above_ma60": [0,1],
    "require_ma_cross": [0,1], "vol_gt_yesterday": [0,1],
    "stop_loss": [-5,-7,-10,-15],
    "use_take_profit": [0,1], "take_profit": [20,30,40,50,60,80,100],
    "trailing_stop": [0,3,5,7,10],
    "use_rsi_sell": [0,1], "rsi_sell": [75,80,85,90,95],
    "use_macd_sell": [0,1], "use_kd_sell": [0,1],
    "sell_vol_shrink": [0,0.3,0.5,0.7],
    "sell_below_ma": [0,1,2,3],
    "hold_days": [5,7,10,15],
}

PARAM_ORDER = [
    "use_rsi_buy","rsi_buy","use_bb_buy","bb_buy",
    "use_vol_filter","vol_filter","require_ma_bull",
    "use_macd","macd_mode","use_kd","kd_buy_k","kd_cross",
    "use_wr_buy","wr_buy",
    "momentum_min","consecutive_green","gap_up","near_high_pct",
    "above_ma60","require_ma_cross","vol_gt_yesterday",
    "stop_loss","use_take_profit","take_profit","trailing_stop",
    "use_rsi_sell","rsi_sell","use_macd_sell","use_kd_sell",
    "sell_vol_shrink","sell_below_ma","hold_days",
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

    return {"tickers":tickers,"dates":dates,"n_stocks":n,"n_days":ml,
        "close":close,"rsi":rsi,"bb_pos":bb_pos,"vol_ratio":vol_ratio,
        "macd_line":ml_arr,"macd_hist":mh,"k_val":kv.astype(np.float32),
        "d_val":dv.astype(np.float32),"is_green":ig,"gap":gp.astype(np.float32),
        "williams_r":wr,"near_high":nh,"vol_prev":vol_prev.astype(np.float32),
        "ma_d":ma_d,"mom_d":mom_d,"ma60":ma_d[60]}

REASON_NAMES = ["到期","停利","停損","RSI超買","移動停利","MACD死叉","KD死叉","量縮","跌破均線"]

def cpu_replay(pre, p):
    """用 CPU 重跑一次最佳參數，拿完整交易明細（股票名、日期、價格）"""
    ns, nd = pre["n_stocks"], pre["n_days"]
    tickers = pre["tickers"]; dates = pre["dates"]; close = pre["close"]
    rsi=pre["rsi"]; bb_pos=pre["bb_pos"]; vol_ratio=pre["vol_ratio"]
    macd_hist=pre["macd_hist"]; macd_line=pre["macd_line"]
    k_val=pre["k_val"]; d_val=pre["d_val"]; williams_r=pre["williams_r"]
    is_green=pre["is_green"]; gap=pre["gap"]; near_high=pre["near_high"]
    vol_prev=pre["vol_prev"]
    maf=pre["ma_d"].get(int(p.get("ma_fast_w",5)), pre["ma_d"][5])
    mas=pre["ma_d"].get(int(p.get("ma_slow_w",20)), pre["ma_d"][20])
    ma60=pre["ma60"]
    mom=pre["mom_d"].get(int(p.get("momentum_days",5)), pre["mom_d"][5])

    holding=-1; bp=0.0; pk=0.0; bd=0; trades=[]
    for day in range(30, nd-1):
        if holding >= 0:
            si=holding; cur=float(close[si,day]); dh=day-bd
            ret=(cur/bp-1)*100
            if dh<1: continue
            if cur>pk: pk=cur
            sell=False; reason=0
            if ret<=p["stop_loss"]: sell=True; reason=2
            if not sell and p.get("use_take_profit",1) and ret>=p["take_profit"]: sell=True; reason=1
            if not sell and p.get("trailing_stop",0)>0 and pk>bp:
                if (cur/pk-1)*100<=-p["trailing_stop"]: sell=True; reason=4
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
            if not sell and dh>=int(p["hold_days"]): sell=True; reason=0
            if sell:
                trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                    "buy_date":str(dates[bd].date()),"sell_date":str(dates[day].date()),
                    "buy_price":round(bp,2),"sell_price":round(cur,2),
                    "return":round(ret,2),"days":dh,"reason":REASON_NAMES[min(reason,len(REASON_NAMES)-1)]})
                holding=-1
            continue
        best_si=-1; best_v=0
        for si in range(ns):
            buy=True
            if buy and p.get("use_rsi_buy",1) and rsi[si,day]<p.get("rsi_buy",55): buy=False
            if buy and p.get("use_bb_buy",1) and bb_pos[si,day]<p.get("bb_buy",0.7): buy=False
            if buy and p.get("use_vol_filter",1) and vol_ratio[si,day]<p.get("vol_filter",3): buy=False
            if buy and p.get("require_ma_bull",0) and close[si,day]<maf[si,day]: buy=False
            if buy and p.get("use_macd",0):
                mm=int(p.get("macd_mode",2))
                if mm==0 and not (macd_hist[si,day]>0 and macd_hist[si,day-1]<=0): buy=False
                elif mm==1 and macd_line[si,day]<=0: buy=False
            if buy and p.get("use_kd",0):
                if k_val[si,day]<p.get("kd_buy_k",50): buy=False
                if buy and p.get("kd_cross",0) and day>=1:
                    if not (k_val[si,day]>d_val[si,day] and k_val[si,day-1]<=d_val[si,day-1]): buy=False
            if buy and p.get("use_wr_buy",0) and williams_r[si,day]<p.get("wr_buy",-30): buy=False
            if buy and p.get("momentum_min",0)>0 and mom[si,day]<p["momentum_min"]: buy=False
            cg=int(p.get("consecutive_green",0))
            if buy and cg>=1:
                for g in range(cg):
                    if day-g<0 or is_green[si,day-g]!=1: buy=False; break
            if buy and p.get("gap_up",0) and gap[si,day]<1.0: buy=False
            nhp=p.get("near_high_pct",0)
            if buy and nhp>0 and abs(near_high[si,day])>nhp: buy=False
            if buy and p.get("above_ma60",0) and close[si,day]<ma60[si,day]: buy=False
            if buy and p.get("require_ma_cross",0) and maf[si,day]<mas[si,day]: buy=False
            if buy and p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]<=vol_prev[si,day]: buy=False
            if buy and vol_ratio[si,day]>best_v: best_si=si; best_v=vol_ratio[si,day]
        if best_si>=0 and day+1<nd:
            holding=best_si; bp=float(close[best_si,day+1]); pk=bp; bd=day+1
    return sorted(trades, key=lambda x: x["buy_date"])

def main():
    print("[GPU-CuPy] 🚀 RTX 3060 進化引擎啟動！")
    raw = download_data()
    data = filter_top(raw, 50)
    auto_fetch_names(list(data.keys()))
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
    d_ma60 = cp.asarray(pre["ma60"])

    print("[GPU] 開始進化！每批 500,000 組")
    BATCH = 500000
    BLOCK = 256
    N_PARAMS = len(PARAM_ORDER)
    best_score = -999999
    best_params = None
    best_avg = 0; best_total = 0; best_wr = 0; best_nt = 0
    total_tested = 0; total_improved = 0; last_synced_improved = 0
    start = time.time()
    rnd = 0

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

    while True:
        rnd += 1
        # 一次產生全部參數（含 MA/MOM 選擇）
        params_np = np.zeros((BATCH, N_PARAMS_FULL), dtype=np.float32)
        for i, key in enumerate(PARAM_ORDER):
            params_np[:, i] = np.random.choice(PARAMS_SPACE[key], BATCH).astype(np.float32)
        # 隨機選 MA 和 MOM
        mf_choices = np.random.choice(MA_FAST_OPTS, BATCH)
        ms_choices = np.random.choice(MA_SLOW_OPTS, BATCH)
        md_choices = np.random.choice(MOM_DAYS_OPTS, BATCH)
        # 過濾 ma_fast >= ma_slow 的（設 idx 讓 kernel 跳過）
        for j in range(BATCH):
            if mf_choices[j] >= ms_choices[j]:
                ms_choices[j] = max(MA_SLOW_OPTS)  # 強制用 60
            params_np[j, N_PARAMS] = MA_FAST_MAP[mf_choices[j]]
            params_np[j, N_PARAMS+1] = MA_SLOW_MAP[ms_choices[j]]
            params_np[j, N_PARAMS+2] = MOM_MAP[md_choices[j]]

        d_params = cp.asarray(params_np)
        d_results = cp.zeros((BATCH, 5), dtype=cp.float32)
        grid = (BATCH + BLOCK - 1) // BLOCK

        CUDA_KERNEL((grid,), (BLOCK,), (
            np.int32(ns), np.int32(nd),
            d_close, d_rsi, d_bb, d_vr, d_mh, d_ml,
            d_kv, d_dv, d_mom3, d_mom5, d_mom10,
            d_ig, d_gp, d_nh, d_wr,
            d_ma3, d_ma5, d_ma10, d_ma15, d_ma20, d_ma30, d_ma60,
            d_vp,
            d_params, np.int32(N_PARAMS_FULL),
            d_results, np.int32(BATCH)
        ))

        results = d_results.get()
        total_tested += BATCH

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
            print(f"  [GPU] 新紀錄！{best_score:.1f} | 勝率{best_wr:.0f}% | 平均{best_avg:.1f}% | {best_nt}筆")

        elapsed = time.time() - start
        speed = total_tested / elapsed
        print(f"[GPU] R{rnd} | {total_tested:,}組 | {elapsed:.0f}秒 | {speed:,.0f}組/秒 | 突破{total_improved}")

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
                        f"  {t['name']}({t['ticker'].replace('.TW','')}) | {t['buy_date'][5:]}→{t['sell_date'][5:]} | {t['return']:+.1f}% | {t['days']}天 | {t['reason']}"
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
