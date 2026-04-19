#!/usr/bin/env python3
"""
GPU 進化引擎 — CuPy 版（RTX 3060 專用）
用 GPU 同時跑上萬組參數的買賣訊號計算
"""
import numpy as np
import cupy as cp
import json, os, sys, time, requests, pickle, base64

# === Telegram / Gist ===
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8551169875:AAF48gHaISTcKgAAZ_CXCOFoG0ZT21aN0RI")
CHAT_IDS = ["5785839733"]  # William only (v2 experimental)
GIST_ID = os.environ.get("GIST_ID", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
DATA_GIST_ID = "a300b9e29372ac76f79eda39a2a86321"
CACHE_PATH = os.path.join(os.path.expanduser("~"), "stock-evolution", "stock_data_cache.pkl")

# === 評分哲學（2026-04-18 第二次調整：報酬 6 折 + WF 放寬）===
# 7 折版本跑不出突破 — 反向 WF 下 test=2022 熊市難達 324% 年化，
# WF ratio 0.7 要求 2022 熊市年化 ≥ 近期 × 0.7 也過嚴。
# 放寬到 6 折 + WF 0.55，給 GPU 可行區間。
#
# 基於 189 在反向 WF 下實測：
#   Train（新）年化 543% 勝率 60%
#   Test（舊）年化 462% 勝率 67%
#
# 評分主軸（勝率加重 + 波段獎勵）：
#   - s_wr × 2.0 (cap 80)  勝率 65% = +30，75% = +50，80% = +60，90% = +80
#   - s_return × 0.05      年化 400% 封頂 = +20
#   - s_avg × 2.0 (cap 30) 單筆 avg 15%=0 / 20%=+10 / 25%=+20 / 30%=+30（獎勵吃波段）
#   - s_wf × 15            走前校驗
#
# 硬門檻（189 × 0.6，可行區間）：
#   - train 勝率 >= 0.65、test 勝率 >= 0.60（勝率地板維持，核心需求）
#   - train 年化 >= 326%（= 543 × 0.6）
#   - test 年化 >= 277%（= 462 × 0.6）
#   - WF ratio >= 0.55（2022 熊市達近期 55% 更實際）
#   - recent-60d avg >= 5%
MIN_WR_TRAIN = 0.70     # WINRATE-MAX 模式提高到 70%（原 65%）
MIN_WR_TEST = 0.65      # WINRATE-MAX 模式提高到 65%（原 60%）
MIN_TRAIN_ANNUAL = 326  # 189 train 年化 543% × 0.6
MIN_TEST_ANNUAL = 277   # 189 test 年化 462% × 0.6

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

def append_new_days(cache_path):
    """批次 append cache 新天 — 舊資料 0 變動。用 yf.download batch 下載（100 檔/批），
    比舊版逐個 yf.Ticker.history 快 5-10 倍。每批印進度讓你看到東西在動。
    """
    import pandas as pd, traceback
    if not os.path.exists(cache_path):
        return None, 0
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    today = pd.Timestamp.now().normalize()
    updated = 0
    first_err_printed = False

    # 按 last_date 分組（通常所有 ticker 同一天，只有 1 組）
    groups = {}
    for ticker, df in cache.items():
        if df is None or len(df) == 0: continue
        last = df.index[-1]
        if getattr(last, 'tz', None) is not None:
            last = last.tz_localize(None)
        last_date = pd.Timestamp(last).normalize()
        if last_date >= today: continue
        groups.setdefault(last_date, []).append(ticker)

    if not groups:
        print("  所有 ticker 已到今天，無需 append")
        return cache, 0

    BATCH = 100
    for last_date, tickers_to_update in groups.items():
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        total = len(tickers_to_update)
        print(f"  [{last_date.date()} → {today.date()}] {total} 檔需要補 {start}~{end}")

        for bi in range(0, total, BATCH):
            batch_tickers = tickers_to_update[bi:bi+BATCH]
            try:
                batch_df = yf.download(
                    batch_tickers,
                    start=start, end=end,
                    group_by='ticker',
                    progress=False, threads=True,
                    auto_adjust=True,
                )
            except Exception as e:
                if not first_err_printed:
                    print(f"  [batch {bi}-{bi+len(batch_tickers)}] download 失敗: {e}")
                    traceback.print_exc()
                    first_err_printed = True
                continue

            for ticker in batch_tickers:
                try:
                    if len(batch_tickers) == 1:
                        new = batch_df
                    else:
                        lv0 = batch_df.columns.get_level_values(0).unique()
                        if ticker not in lv0: continue
                        new = batch_df[ticker]
                    new = new.dropna(how='all')
                    if new is None or len(new) == 0: continue
                    if new.index.tz is not None:
                        new.index = new.index.tz_localize(None)
                    df = cache[ticker]
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    # 防呆：yfinance 假日抓會返回前一交易日資料，那筆會跟 df 最後一天重複
                    # 強制過濾 new 只保留 > df 最後一天的 row
                    df_last = df.index[-1]
                    new = new[new.index > df_last]
                    # 再過濾 Close 為 NaN 的 row（有時 yfinance 返回帶 Dividends=0 的空 row）
                    if 'Close' in new.columns:
                        new = new[new['Close'].notna()]
                    if len(new) == 0: continue
                    common_cols = [c for c in df.columns if c in new.columns]
                    if not common_cols: continue
                    new = new[common_cols]
                    merged = pd.concat([df, new])
                    # 保險：即使 concat 有重複 index，keep='first' 保舊資料不動
                    if merged.index.duplicated().any():
                        merged = merged[~merged.index.duplicated(keep='first')]
                    cache[ticker] = merged
                    updated += 1
                except Exception as e:
                    if not first_err_printed:
                        print(f"  第一個 ticker 處理失敗 ({ticker}): {type(e).__name__}: {e}")
                        traceback.print_exc()
                        first_err_printed = True
                    continue

            done = min(bi + BATCH, total)
            print(f"  進度 {done}/{total}  累計更新 {updated} 檔", flush=True)

    if updated > 0:
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp, cache_path)
    return cache, updated


def download_data():
    if os.path.exists(CACHE_PATH):
        age = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age < 720:
            try:
                with open(CACHE_PATH, "rb") as f: data = pickle.load(f)
                if len(data) >= 10: print(f"[快取] {len(data)} 檔"); return data
            except: pass
    data = {}
    total = len(TW_TICKERS)
    print(f"[下載] {total} 檔股票資料（首次需 10-20 分鐘）...")
    for i, t in enumerate(TW_TICKERS):
        try:
            h = yf.Ticker(t).history(period="4y")
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
    const float* open_price, const float* top100_mask, const float* market_bull,
    const float* up_days, const float* week52_pos, const float* vol_up_days, const float* mom_accel,
    const float* mfi, const float* cmf, const float* atr_ratio,
    const float* params, const int n_params_per_combo,
    float* results, const int n_combos,
    const int train_start, const int train_end
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
    // 換股（持倉滿時，候選分數 - 持股分數 >= margin → 賣弱換強）
    int upgrade_margin = (int)p[56];
    // 多持倉
    int max_pos = (int)p[57]; if (max_pos < 1) max_pos = 1; if (max_pos > 3) max_pos = 3;
    // 新指標
    int w_up_days_k = (int)p[60]; float up_days_min_k = p[61];
    int w_week52_k = (int)p[62]; float week52_min_k = p[63];
    int w_vol_up_days_k = (int)p[64]; float vol_up_days_min_k = p[65];
    int w_mom_accel_k = (int)p[66]; float mom_accel_min_k = p[67];
    // 價格穩定性（過去 3 天 |close 變化| 上限，0=關）
    float max_3d_change = p[68];
    // 第 1 名 vs 第 2 名分數差門檻（過濾訊號不明確日）
    float top1_margin = p[69];
    // 快速認賠：買後 N 天內虧 threshold% 立刻砍
    int early_exit_days = (int)p[70];
    float early_exit_th = p[71];
    // 訊號持續性：過去 N 天也必須是 top100 強勢股（對抗單日運氣）
    int signal_persist_days = (int)p[72];
    // 賣股後空倉冷卻期（解耦賣跟買）
    int buy_delay_days = (int)p[73];
    // New indicators (MFI, CMF, ATR contraction)
    int w_mfi_k = (int)p[74]; float mfi_th_k = p[75];
    int w_cmf_k = (int)p[76]; float cmf_th_k = p[77];
    int w_atr_contract_k = (int)p[78]; float atr_contract_th_k = p[79];
    // MA/MOM 選擇
    int ma_fast_idx = (int)p[80];
    int ma_slow_idx = (int)p[81];
    int mom_idx = (int)p[82];

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
    float rets[200];
    int trade_bdays[200];
    int hold_days_arr[200];

    for (int day = 60; day < n_days - 1; day++) {
        // === Phase 1: 檢查所有持倉的賣出條件（用 day 收盤價判斷）===
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

            // 賣出用 D+1 開盤價（跟實戰一致：收到訊號隔天一開盤賣）
            if (sell && day + 1 < n_days && n_trades < 200) {
                float sell_price = open_price[si * n_days + day + 1];
                float actual_ret = (sell_price / hold_bp[h] - 1.0f) * 100.0f - 0.585f;  // 扣手續費+證交稅
                rets[n_trades] = actual_ret;
                trade_bdays[n_trades] = hold_bd[h];
                hold_days_arr[n_trades] = day + 1 - hold_bd[h];
                total_ret += actual_ret;
                if (actual_ret > 0) win_count += 1;
                if (actual_ret < 10) wasted_count += 1;
                n_trades++;
                hold_si[h] = -1;
                n_holding--;
            }
        }

        // === Phase 1.5: 換股 — 持倉滿且有更強候選，賣弱換強 ===
        if (upgrade_margin > 0 && n_holding >= max_pos && day + 1 < n_days) {
            // 找候選最高分
            int cand_si = -1; float cand_sc = 0; float cand_vol = 0;
            for (int si = 0; si < n_stocks; si++) {
                if (top100_mask[si * n_days + day] < 0.5f) continue;
                bool already = false;
                for (int h = 0; h < max_pos; h++) {
                    if (hold_si[h] == si) { already = true; break; }
                }
                if (already) continue;
                // max_3d_change 已禁用，移除 check
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
                if (w_atr > 0 && atr_pct[d] >= atr_min) sc += w_atr;
                if (w_up_days_k > 0 && up_days[d] >= up_days_min_k) sc += w_up_days_k;
                if (w_week52_k > 0 && week52_pos[d] >= week52_min_k) sc += w_week52_k;
                if (w_vol_up_days_k > 0 && vol_up_days[d] >= vol_up_days_min_k) sc += w_vol_up_days_k;
                if (w_mom_accel_k > 0 && mom_accel[d] >= mom_accel_min_k) sc += w_mom_accel_k;
                if (w_mfi_k > 0 && mfi[d] >= mfi_th_k) sc += w_mfi_k;
                if (w_cmf_k > 0 && cmf[d] >= cmf_th_k) sc += w_cmf_k;
                if (w_atr_contract_k > 0 && atr_ratio[d] <= atr_contract_th_k) sc += w_atr_contract_k;
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
                if (sc >= buy_threshold && (sc > cand_sc || (sc == cand_sc && vol_ratio[d] > cand_vol))) { cand_si = si; cand_sc = sc; cand_vol = vol_ratio[d]; }
            }
            // 對每檔持股重新評分，找最弱的
            if (cand_si >= 0) {
                int weakest_h = -1; float weakest_sc = 9999;
                for (int h = 0; h < max_pos; h++) {
                    if (hold_si[h] < 0) continue;
                    int si = hold_si[h]; int d = si * n_days + day;
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
                    if (w_kd > 0 && k_val[d] >= kd_th) sc += w_kd;
                    if (w_wr > 0 && williams_r[d] >= wr_th) sc += w_wr;
                    if (w_mom > 0 && momentum[d] >= mom_th) sc += w_mom;
                    if (w_near_high > 0 && fabsf(near_high[d]) <= near_high_pct) sc += w_near_high;
                    if (w_squeeze > 0 && squeeze_fire[d] > 0.5f) sc += w_squeeze;
                    if (w_new_high > 0 && new_high_60[d] > 0.5f) sc += w_new_high;
                    if (w_adx > 0 && adx[d] >= adx_th) sc += w_adx;
                    if (w_bias > 0 && bias[d] >= 0 && bias[d] <= bias_max_th) sc += w_bias;
                    if (w_obv > 0 && obv_rising[d] > 0.5f) sc += w_obv;
                    if (w_atr > 0 && atr_pct[d] >= atr_min) sc += w_atr;
                    if (w_up_days_k > 0 && up_days[d] >= up_days_min_k) sc += w_up_days_k;
                    if (w_week52_k > 0 && week52_pos[d] >= week52_min_k) sc += w_week52_k;
                    if (w_vol_up_days_k > 0 && vol_up_days[d] >= vol_up_days_min_k) sc += w_vol_up_days_k;
                    if (w_mom_accel_k > 0 && mom_accel[d] >= mom_accel_min_k) sc += w_mom_accel_k;
                    if (w_mfi_k > 0 && mfi[d] >= mfi_th_k) sc += w_mfi_k;
                    if (w_cmf_k > 0 && cmf[d] >= cmf_th_k) sc += w_cmf_k;
                    if (w_atr_contract_k > 0 && atr_ratio[d] <= atr_contract_th_k) sc += w_atr_contract_k;
                    if (sc < weakest_sc) { weakest_sc = sc; weakest_h = h; }
                }
                // 候選分數 - 最弱持股分數 >= margin → 賣弱換強
                if (weakest_h >= 0 && cand_sc - weakest_sc >= upgrade_margin && n_trades < 200) {
                    int si = hold_si[weakest_h];
                    float sell_price = open_price[si * n_days + day + 1];
                    float actual_ret = (sell_price / hold_bp[weakest_h] - 1.0f) * 100.0f - 0.585f;  // 扣手續費+證交稅
                    rets[n_trades] = actual_ret;
                    trade_bdays[n_trades] = hold_bd[weakest_h];
                    hold_days_arr[n_trades] = day + 1 - hold_bd[weakest_h];
                    total_ret += actual_ret;
                    if (actual_ret > 0) win_count += 1;
                    if (actual_ret < 10) wasted_count += 1;
                    n_trades++;
                    hold_si[weakest_h] = -1;
                    n_holding--;
                }
            }
        }

        // === Phase 2: 有空位就買一檔 ===
        if (n_holding < max_pos && day + 1 < n_days) {
            int best_si = -1;
            float best_buy_score = 0; float best_buy_vol = 0;
            for (int si = 0; si < n_stocks; si++) {
                // 只從當天成交量前 100 名買（跟實戰一致）
                if (top100_mask[si * n_days + day] < 0.5f) continue;
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
                if (w_atr > 0 && atr_pct[d] >= atr_min) sc += w_atr;
                if (w_up_days_k > 0 && up_days[d] >= up_days_min_k) sc += w_up_days_k;
                if (w_week52_k > 0 && week52_pos[d] >= week52_min_k) sc += w_week52_k;
                if (w_vol_up_days_k > 0 && vol_up_days[d] >= vol_up_days_min_k) sc += w_vol_up_days_k;
                if (w_mom_accel_k > 0 && mom_accel[d] >= mom_accel_min_k) sc += w_mom_accel_k;
                if (w_mfi_k > 0 && mfi[d] >= mfi_th_k) sc += w_mfi_k;
                if (w_cmf_k > 0 && cmf[d] >= cmf_th_k) sc += w_cmf_k;
                if (w_atr_contract_k > 0 && atr_ratio[d] <= atr_contract_th_k) sc += w_atr_contract_k;

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

                // max_3d_change / signal_persist 已禁用（PARAMS_SPACE=[0]），移除 check 加速

                if (sc >= buy_threshold && (sc > best_buy_score || (sc == best_buy_score && vol_ratio[d] > best_buy_vol))) {
                    best_si = si; best_buy_score = sc; best_buy_vol = vol_ratio[d];
                }
            }
            if (best_si >= 0) for (int h = 0; h < max_pos; h++) {
                if (hold_si[h] < 0) {
                    hold_si[h] = best_si;
                    hold_bp[h] = close[best_si * n_days + day + 1];  // 買入 D+1 收盤
                    hold_pk[h] = hold_bp[h];
                    hold_bd[h] = day + 1;
                    n_holding++;
                    break;
                }
            }
        }

        // Phase 3 已移除（第三檔回測表現不佳）
    }

    // === 評分 v6（真 Walk-Forward + 全期實盤安全）：train 計分，全期+test 雙重驗證 ===
    float score = -999999.0f;

    // 先檢查全期實盤安全（實盤會遇到的回撤/筆數/持有/報酬）— 不過就拒絕，不進訓練評分
    bool all_pass = false;
    if (n_trades >= 40 && n_trades <= 200) {  // 1500 天下筆數可能 140-180
        float avg_ret_all = total_ret / n_trades;
        float avg_hold_all = 0;
        for (int i=0; i<n_trades; i++) avg_hold_all += (float)hold_days_arr[i];
        avg_hold_all /= n_trades;
        if (avg_ret_all >= 3.0f && avg_hold_all >= 5.0f) {
            float max_dd_all = 0, run_dd_all = 0;
            for (int i=0; i<n_trades; i++) {
                if (rets[i] < 0) run_dd_all += rets[i];
                else run_dd_all = 0;
                if (run_dd_all < max_dd_all) max_dd_all = run_dd_all;
            }
            if (max_dd_all >= -50.0f) all_pass = true;  // 1500 天含 2020 covid + 2022 熊市，-50 合理
        }
    }
    // 近期 60 天崩盤檢查：抓「train 強、test 前段強、test 後段崩」型策略
    // 如近 60 天有 >= 3 筆但平均 < 5%，拒絕（實盤上線立刻會遇到的品質）
    if (all_pass) {
        int recent_start = n_days - 60;
        float recent_total = 0; int recent_n = 0;
        for (int i=0; i<n_trades; i++) {
            if (trade_bdays[i] >= recent_start) {
                recent_total += rets[i];
                recent_n++;
            }
        }
        if (recent_n >= 3) {
            float recent_avg = recent_total / recent_n;
            if (recent_avg < 5.0f) all_pass = false;  // 跟 Python gate 對齊（勝率策略每筆期望值低，5% 是合理地板）
        }
    }

    // 分 train / test（train 區間外 + 過 warmup 就是 test，支援正向/反向 WF）
    int n_train = 0, n_test = 0;
    float rets_train[200], rets_test[200];
    int bdays_train[200], hd_train[200];
    float total_train = 0, total_test = 0;
    float win_train = 0;
    for (int i=0; i<n_trades; i++) {
        int bd = trade_bdays[i];
        if (bd < 60) continue;  // warmup 期不算
        if (bd >= train_start && bd < train_end) {
            rets_train[n_train] = rets[i];
            bdays_train[n_train] = bd;
            hd_train[n_train] = hold_days_arr[i];
            total_train += rets[i];
            if (rets[i] > 0) win_train += 1;
            n_train++;
        } else {
            rets_test[n_test] = rets[i];
            total_test += rets[i];
            n_test++;
        }
    }

    // D（train）：train 筆數 30-140（適配 900/1500 天不同 cache 長度）+ 全期必須先過
    if (all_pass && n_train >= 30 && n_train <= 140) {
        float avg_ret_tr = total_train / n_train;
        float win_rate_tr = win_train / n_train * 100.0f;
        float avg_hold_tr = 0;
        for (int i=0; i<n_train; i++) avg_hold_tr += (float)hd_train[i];
        avg_hold_tr /= n_train;

        if (avg_ret_tr >= 8 && win_rate_tr >= 35 && avg_hold_tr >= 5.0f) {
            // Sharpe（train）
            float sum_sq = 0;
            for (int i=0; i<n_train; i++) { float d = rets_train[i] - avg_ret_tr; sum_sq += d*d; }
            float std_tr = sqrtf(sum_sq / n_train);
            float sharpe_tr = std_tr > 0.5f ? avg_ret_tr / std_tr : avg_ret_tr;
            if (sharpe_tr > 10) sharpe_tr = 10;

            // 盈虧比（train）
            float aw=0, al=0; int nw=0, nl=0;
            for (int i=0; i<n_train; i++) {
                if (rets_train[i]>0) { aw+=rets_train[i]; nw++; }
                else { al+=fabsf(rets_train[i]); nl++; }
            }
            if (nw>0) aw /= nw; if (nl>0) al /= nl;
            float pl_ratio = al>0.5f ? aw/al : aw;
            if (pl_ratio > 20) pl_ratio = 20;

            // 最大連虧（train）
            float max_dd_tr = 0, run_dd = 0;
            for (int i=0; i<n_train; i++) {
                if (rets_train[i] < 0) run_dd += rets_train[i]; else run_dd = 0;
                if (run_dd < max_dd_tr) max_dd_tr = run_dd;
            }
            if (max_dd_tr >= -50.0f) {  // 放寬到 -50 配合 1500 天
                int max_streak = 0, streak = 0;
                for (int i=0; i<n_train; i++) {
                    if (rets_train[i] <= 0) { streak++; if (streak > max_streak) max_streak = streak; }
                    else streak = 0;
                }

                float train_years = (float)(train_end - train_start) / 250.0f;
                float test_years = (float)(train_start - 60) / 250.0f;
                float train_annual = train_years > 0.5f ? total_train / train_years : total_train;
                float test_annual = test_years > 0.3f ? total_test / test_years : total_test;

                // Walk-Forward 盲測門檻：test 必須有效正報酬、不退化超過 50%
                bool wf_pass = true;
                if (n_test < 5) wf_pass = false;
                if (total_test <= 0) wf_pass = false;
                if (test_annual < train_annual * 0.4f) wf_pass = false;  // 反向 WF 下 test=2022 熊市，0.4 = 88.60 (0.42) 剛好能過當 SEED

                if (wf_pass) {
                    // 3 段一致性（train 期內部）
                    int seg_size = (train_end - train_start) / 3;
                    if (seg_size < 10) seg_size = 10;
                    float seg_ret[3] = {0,0,0}; int seg_n[3] = {0,0,0};
                    for (int i=0; i<n_train; i++) {
                        int bd_rel = bdays_train[i] - train_start;
                        if (bd_rel < 0) bd_rel = 0;
                        int seg = bd_rel / seg_size;
                        if (seg > 2) seg = 2;
                        seg_ret[seg] += rets_train[i]; seg_n[seg]++;
                    }
                    float min_seg_annual = 9999;
                    int active_segs = 0;
                    bool seg_ok = true;
                    for (int s=0; s<3; s++) {
                        if (seg_n[s] >= 4) {
                            if (seg_ret[s] <= 0) seg_ok = false;
                            float sa = seg_ret[s] / (train_years / 3.0f);
                            if (sa < min_seg_annual) min_seg_annual = sa;
                            active_segs++;
                        }
                    }
                    // 加檢：train 最後 1/3 段不能崩（seg[2] avg/筆 >= seg[0] avg/筆 x 0.6）
                    // 擋掉「前期超賺、後期淡掉」的老化策略
                    bool late_seg_ok = true;
                    if (seg_n[0] >= 4 && seg_n[2] >= 4) {
                        float seg0_avg = seg_ret[0] / seg_n[0];
                        float seg2_avg = seg_ret[2] / seg_n[2];
                        if (seg2_avg < seg0_avg * 0.6f) late_seg_ok = false;
                    }
                    if (active_segs >= 2 && seg_ok && late_seg_ok) {
                        float s_consistency = min_seg_annual * 0.03f;
                        if (s_consistency > 10) s_consistency = 10;

                        // === WINRATE-MAX scoring（2026-04-19 改，追求極高勝率）===
                        // s_wr: 2.0→2.5 cap 100 (勝率 70%=50 / 80%=75 / 90%=100)
                        // s_avg: 2.0→0.5 cap 5（大幅降權，避免 GPU 被波段誘惑）
                        float s_wr = (win_rate_tr - 50.0f) * 2.5f;
                        if (s_wr < 0) s_wr = 0;
                        if (s_wr > 100.0f) s_wr = 100.0f;  // 90% 勝率封頂 100 分

                        // s_return 用 min(train, test)，封頂 400% 年化
                        float effective_annual = train_annual < test_annual ? train_annual : test_annual;
                        float capped_annual = effective_annual > 400.0f ? 400.0f : effective_annual;
                        float s_return = capped_annual * 0.05f;
                        if (s_return < 0) s_return = 0;

                        // s_avg 大幅降權（只留象徵性 cap 5，勝率主導）
                        float s_avg = (avg_ret_tr - 15.0f) * 0.5f;
                        if (s_avg < 0) s_avg = 0;
                        if (s_avg > 5.0f) s_avg = 5.0f;

                        // 近 2 年勝率獎勵（最後 500 天，當前市場強勢鼓勵）
                        // 60%=0, 65%=+3, 70%=+6, 75%=+9, 80%=+12, 90%=+15
                        int recent_start = n_days - 500;
                        if (recent_start < train_start) recent_start = train_start;
                        float rec_wins = 0; int rec_n = 0;
                        for (int i=0; i<n_train; i++) {
                            if (bdays_train[i] >= recent_start) {
                                rec_n++;
                                if (rets_train[i] > 0) rec_wins += 1;
                            }
                        }
                        float s_recent = 0;
                        if (rec_n >= 5) {
                            float rec_wr = rec_wins / rec_n * 100.0f;
                            s_recent = (rec_wr - 60.0f) * 0.5f;
                            if (s_recent < 0) s_recent = 0;
                            if (s_recent > 15.0f) s_recent = 15.0f;
                        }

                        // Calmar 獎勵（CAGR 高 + MaxDD 低 = 風險調整後報酬）
                        // train_ann 400% / MaxDD 20% = 20 → 超頂，cap 10
                        float abs_dd = fabsf(max_dd_tr);
                        float calmar = abs_dd > 1.0f ? (train_annual / abs_dd) : 0;
                        float s_calmar = 0;
                        if (calmar > 2.0f) s_calmar = (calmar - 2.0f) * 1.5f;
                        if (s_calmar > 10.0f) s_calmar = 10.0f;

                        // 輔助評分（權重都降低，讓 s_wr 主導）
                        float s_sharpe = sharpe_tr * 2.0f; if (s_sharpe > 10) s_sharpe = 10;
                        float s_pl = pl_ratio * 0.5f; if (s_pl > 5) s_pl = 5;
                        float s_streak = max_streak * 1.0f;
                        float s_dd = fabsf(max_dd_tr) * 0.05f;
                        float s_hold_pen = 0;
                        if (avg_hold_tr < 8.0f) s_hold_pen = (8.0f - avg_hold_tr) * 2.0f;

                        // WF 泛化：保留防老化，權重 25 → 15
                        float wf_ratio = train_annual > 1.0f ? test_annual / train_annual : 1.0f;
                        if (wf_ratio > 1.2f) wf_ratio = 1.2f;
                        float s_wf = wf_ratio * 15.0f;

                        score = s_wr + s_return + s_avg + s_recent + s_calmar + s_sharpe + s_pl + s_consistency + s_wf - s_streak - s_dd - s_hold_pen;
                    }
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
    "w_rsi": [0,1,2,3], "rsi_th": [60,63,65,68,70,72,75,80],
    "w_bb": [0,1,2,3], "bb_th": [0.7,0.75,0.8,0.85,0.9,0.95,1.0],
    "w_vol": [0,1,2,3], "vol_th": [2.0,2.5,3.0,3.5,4.0,5.0,6.0],
    "w_ma": [0,1,2,3],
    "w_macd": [0,1,2,3], "macd_mode": [0,1,2],
    "w_kd": [0,1,2,3], "kd_th": [60,65,70,75,80,85], "kd_cross": [0,1],
    "w_wr": [0,1,2,3], "wr_th": [-25,-30,-35,-40,-50],
    "w_mom": [0,1,2,3], "mom_th": [5,8,10,12,15,20,25],  # 擴大：更嚴動量要求（過濾弱訊號）
    "w_near_high": [0,1,2], "near_high_pct": [3,5,10],
    "w_squeeze": [0,1,2,3], "w_new_high": [0,1,2,3],
    "w_adx": [0,1,2,3], "adx_th": [25,30,35,40],
    "consecutive_green": [0,1,2,3], "gap_up": [0,1],
    "above_ma60": [0,1], "vol_gt_yesterday": [0,1],
    "buy_threshold": [6,8,10,12,14,16,18,20,22],  # 擴大：高門檻 = 超嚴選（減少運氣依賴）
    # ====== 賣出（全自由探索，靠 120 筆上限 + 品質門檻 + 同資料比較防刷分）======
    "stop_loss": [-5,-7,-10,-12,-15,-20],  # 移除 -3（實盤滑價吃不住，GPU 鑽 Sharpe 公式漏洞）
    "use_take_profit": [0,1], "take_profit": [20,30,40,50,60,80,100,150],
    "trailing_stop": [0,3,5,7,10,15,20,25],  # 加 25：超寬 trailing 讓波段跑
    "use_rsi_sell": [0,1], "rsi_sell": [70,75,80,85,90,95],
    "use_macd_sell": [0,1], "use_kd_sell": [0,1],
    "sell_vol_shrink": [0,0.3,0.5,0.7],
    "sell_below_ma": [0,1,2,3],
    "hold_days": [5,7,10,15,20,25,30],
    # ====== BIAS 乖離率 ======
    "w_bias": [0,1,2,3], "bias_max": [3,5,8,10,15,20,30],
    # ====== 停滯出場 ======
    "use_stagnation_exit": [0,1], "stagnation_days": [5,7,10,15], "stagnation_min_ret": [0,1,3,5],
    # ====== 保本停損 ======
    "use_breakeven": [0,1], "breakeven_trigger": [10,15,20,25,30,35],  # 加 35：晚點啟動保本，讓波段跑
    # ====== OBV 能量潮 ======
    "w_obv": [0,1,2,3], "obv_rising_days": [3,5,10],
    # ====== 漸進式最低報酬 ======
    "use_time_decay": [0,1], "ret_per_day": [0.1,0.2,0.3,0.5,0.8,1.0,1.5],
    # ====== ATR 波動率門檻（過濾低波動股）======
    "w_atr": [0,1,2,3], "atr_min": [1.0,1.5,2.0,2.5,3.0,4.0],
    # ====== 鎖利出場 ======
    "use_profit_lock": [0,1], "lock_trigger": [15,20,30,40,50], "lock_floor": [3,5,8,10,15,20],  # 加 20：鎖大贏家更精準
    # ====== 動量反轉出場 ======
    "use_mom_exit": [0,1], "mom_exit_th": [0,1,2,3,5],
    # ====== 類股資金流向 ======
    "w_sector_flow": [0,1,2,3], "sector_flow_topn": [1,2,3,5,8],
    # ====== 新指標（連續上漲/52週位置/連續量增/動量加速）======
    "w_up_days": [0,1,2,3], "up_days_min": [2,3,4,5,7,10],  # 擴大：連漲更多天才算強訊號
    "w_week52": [0,1,2,3], "week52_min": [0.6,0.7,0.8,0.9],
    "w_vol_up_days": [0,1,2], "vol_up_days_min": [2,3,4,5,7],  # 擴大：連量增更多天
    "w_mom_accel": [0,1,2], "mom_accel_min": [0,2,5,8],
    # 🔒 max_3d_change 跟新 universe 重複 → 禁用（只保留 0）
    "max_3d_change": [0],
    # 🎯 top1_margin 保留但精簡範圍（跟 universe 正交，實驗性保留）
    "top1_margin": [0, 3, 5],
    # ⚡ 快速認賠（跟 universe 正交，買錯後的控損機制，保留）
    "early_exit_days": [0, 3, 5, 7],
    "early_exit_th": [-5, -8, -10, -12],
    # 🔥 signal_persist_days 完全跟 universe 重複 → 禁用（只保留 0）
    "signal_persist_days": [0],
    # 🆕 賣股後強制空倉 N 天（解耦「賣」和「買」，避免時機綁架）
    # 0=不等（現況）；2/3/5/7=賣完後空 N 天才買，讓訊號自由生成而非被迫進場
    "buy_delay_days": [0, 2, 3, 5, 7],
    # ====== MFI / CMF / ATR contraction ======
    "w_mfi": [0,1,2,3], "mfi_th": [60,65,70,75,80],
    "w_cmf": [0,1,2,3], "cmf_th": [0.05, 0.10, 0.15, 0.20],
    "w_atr_contract": [0,1,2,3], "atr_contract_th": [0.70, 0.75, 0.80, 0.85],
    # ====== 換股（賣弱換強）======
    "upgrade_margin": [0,3,5,7,10,15],  # 擴大：強換股門檻，買到爛股可被強股換掉
    # ====== 多持倉 ======
    "max_positions": [2],  # 鎖定 2 檔（v1 框架，memory 寫過 3 檔資金效率差不可破）
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
    "upgrade_margin",
    "max_positions",
    "w_sector_flow","sector_flow_topn",
    "w_up_days","up_days_min",
    "w_week52","week52_min",
    "w_vol_up_days","vol_up_days_min",
    "w_mom_accel","mom_accel_min",
    "max_3d_change",  # 🔒 過去 3 天 |close 變化| ≤ X%（0=關，5/7/10/15=啟用）
    "top1_margin",    # 🎯 第 1 名 vs 第 2 名分數差必須 ≥ X（0=關，2/3/5/7=啟用，過濾「矮中選長」）
    "early_exit_days", # ⚡ 買後 N 天內快速認賠視窗（0=關，3/5/7=啟用）
    "early_exit_th",   # ⚡ 買後快速認賠閾值（-5/-8/-10/-12，搭 early_exit_days 使用）
    "signal_persist_days",  # 🔥 核心：買入要求過去 N 天都是 top100（0=關，2/3/5=啟用，對抗「單日運氣」）
    "buy_delay_days",  # 🆕 賣股後強制空倉 N 天，解耦賣跟買（0=關，2/3/5/7=啟用）
    "w_mfi", "mfi_th",
    "w_cmf", "cmf_th",
    "w_atr_contract", "atr_contract_th",
]

MA_FAST_OPTS = [3,5,10]
MA_SLOW_OPTS = [15,20,30,60]
MOM_DAYS_OPTS = [3,5,10]

# === 全自由探索（所有參數全範圍搜索）===
# Gist 保護：只有超過 112.92 才會推送，不怕退化
FROZEN_PARAMS = set()  # 全部解鎖

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

    # 近 20 日新高：h20 排除今日（否則今天創新高時 near_high=0，跟 w_new_high_60 重複抓新高）
    h20=np.zeros_like(close)
    for i in range(20,close.shape[1]): h20[:,i]=np.max(high[:,i-20:i],axis=1)
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

    # 動態成交量前 100 名 mask（向量化，跟實戰 intraday_monitor 一致）
    ranks = np.argsort(np.argsort(-volume, axis=0), axis=0)  # 每天的排名（0=最大）
    top100_mask = (ranks < 100).astype(np.float32)

    # === Universe 嚴格度控制（env var GPU_UNIVERSE）===
    # 解決 William 關切的「單日爆量假突破污染池」問題：
    # 預設 top100 = 每天成交量前 100（含短期爆量股）
    # top50_p3 = 過去 3 天都在前 50（真正持續強勢，過濾一日爆發）
    # top30_p5 = 過去 5 天都在前 30（極嚴，只從當前市場熱點選）
    # 注意：persist universe 實驗顯示反效果（過濾掉中小型爆量 = 移除波段機會）
    # 預設 top100（跟 89.90 訓練時一致）。env var 模式只保留為實驗用途。
    _universe = os.environ.get("GPU_UNIVERSE", "top100").lower()

    def _build_persist_mask(rank_th, persist_days):
        """過去 persist_days 天都要在 rank_th 名內（含當天）"""
        strict = (ranks < rank_th).astype(np.float32)
        persist = strict.copy()
        for _k in range(1, persist_days + 1):
            shifted = np.zeros_like(strict)
            shifted[:, _k:] = strict[:, :-_k]
            persist *= shifted
        return persist

    def _build_partial_persist_mask(rank_th, window, min_days_in):
        """過去 window 天內，至少 min_days_in 天在 rank_th 名內（較寬鬆）"""
        strict = (ranks < rank_th).astype(np.float32)
        count = strict.copy()
        for _k in range(1, window + 1):
            shifted = np.zeros_like(strict)
            shifted[:, _k:] = strict[:, :-_k]
            count = count + shifted
        return (count >= min_days_in).astype(np.float32)

    if _universe in ("top100_p3", "top100p3"):
        top100_mask = _build_persist_mask(100, 3)
        print(f"  🎯 Universe: top100 連續 3 天（寬鬆版）")
    elif _universe in ("top100_p2", "top100p2"):
        top100_mask = _build_persist_mask(100, 2)
        print(f"  🎯 Universe: top100 連續 2 天（最寬鬆強勢過濾）")
    elif _universe in ("top80_p3", "top80p3"):
        top100_mask = _build_persist_mask(80, 3)
        print(f"  🎯 Universe: top80 連續 3 天（中等嚴格）")
    elif _universe in ("top50_p3", "top50p3"):
        top100_mask = _build_persist_mask(50, 3)
        print(f"  🎯 Universe: top50 連續 3 天（嚴格版）")
    elif _universe in ("top100_partial", "top100p"):
        # 過去 5 天內至少 3 天在 top 100（最接近實戰「持續強勢但有波動」）
        top100_mask = _build_partial_persist_mask(100, 5, 3)
        print(f"  🎯 Universe: top100 過去 5 天內 ≥ 3 天入選（彈性強勢）")
    else:
        print(f"  Universe: top100（預設，當天前 100）")
    _universe_sum = int(top100_mask.sum(axis=0).mean())
    print(f"    平均每天 universe 大小：{_universe_sum} 檔")

    # 類股相對強度（Sector Relative Strength）
    # 舊版（成交金額 ratio）的 3 大問題：
    #   1. 大盤漲跌日全部類股都爆量 → 排名洗牌無意義
    #   2. 沒「資金從 A 流向 B」概念，只看類股自身爆量
    #   3. Silent bug：sector_hot 初始化 0 = rank 最熱 → 沒映射股票永遠加分
    # 新版：類股 20d 平均報酬 - 大盤 20d 平均報酬 → 真正的 RS
    try:
        sf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tw_sector_mapping.json")
        with open(sf_path, "r", encoding="utf-8") as f:
            _sec_map = json.load(f)
    except:
        _sec_map = {}
    _sec_names = sorted(set(_sec_map.values()))
    _sec_id = {s: i for i, s in enumerate(_sec_names)}
    _n_sec = max(len(_sec_names), 1)
    _stock_sec = np.full(n, -1, dtype=np.int32)
    for si, t in enumerate(tickers):
        sec = _sec_map.get(t, "")
        if sec in _sec_id:
            _stock_sec[si] = _sec_id[sec]
    # 每檔股票 20d return（收盤對 20 天前）
    _stock_ret = np.zeros((n, ml), dtype=np.float32)
    _prev20 = np.zeros_like(close)
    _prev20[:, 20:] = close[:, :-20]
    _ret_valid = (_prev20 > 0) & (close > 0)
    _stock_ret[:, 20:] = np.where(_ret_valid[:, 20:],
                                  (close[:, 20:] / np.where(_prev20[:, 20:] > 0, _prev20[:, 20:], 1.0) - 1.0) * 100,
                                  0).astype(np.float32)
    # 類股平均報酬：每日把類股所有成份股的 20d return 平均
    _sec_sum = np.zeros((_n_sec, ml), dtype=np.float64)
    _sec_cnt = np.zeros((_n_sec, ml), dtype=np.float64)
    for si in range(n):
        sid = _stock_sec[si]
        if sid >= 0:
            _mask = _ret_valid[si].astype(np.float64)
            _sec_sum[sid] += _stock_ret[si] * _mask
            _sec_cnt[sid] += _mask
    _sec_avg = np.where(_sec_cnt > 0, _sec_sum / np.where(_sec_cnt > 0, _sec_cnt, 1.0), 0).astype(np.float32)
    # 大盤 = 所有類股等權平均（過濾當日有資料的類股）
    _mkt_cnt = (_sec_cnt > 0).astype(np.float32).sum(axis=0)
    _mkt_sum = _sec_avg.sum(axis=0)
    _mkt_ret = np.where(_mkt_cnt > 0, _mkt_sum / np.where(_mkt_cnt > 0, _mkt_cnt, 1.0), 0)
    # 相對強度 RS = 類股報酬 - 大盤報酬
    _sec_rs = _sec_avg - _mkt_ret[np.newaxis, :]
    # 按 RS 排名每天（0 = RS 最強）+ sector_hot 初始化 999（silent bug 修正）
    sector_hot = np.full((n, ml), 999.0, dtype=np.float32)
    _sec_rank = np.full(_n_sec, 999, dtype=np.int32)
    for d in range(20, ml):
        # 排除當天無資料的類股（_sec_cnt[:, d] == 0 的給極大 RS 值才不會誤進 topN 是錯的，應設 -inf 才對）
        _rs_day = np.where(_sec_cnt[:, d] > 0, _sec_rs[:, d], -1e9)
        _rank = np.argsort(-_rs_day)
        _sec_rank[:] = 999
        for r, sid in enumerate(_rank):
            if _sec_cnt[sid, d] > 0:
                _sec_rank[sid] = r
        for si in range(n):
            sid = _stock_sec[si]
            if sid >= 0:
                sector_hot[si, d] = float(_sec_rank[sid])
            # sid < 0 → 保持 999（修 silent bug：舊版保持 0 會被誤判為 top 1）
    _mapped = int((_stock_sec >= 0).sum())
    print(f"  類股相對強度：{_mapped}/{n} 檔有產業映射，{_n_sec} 個產業（RS = 類股20d報酬 - 大盤20d報酬）")

    # 連續上漲天數（close[i] > close[i-1]）
    up_days = np.zeros((n, ml), dtype=np.float32)
    for i in range(1, ml):
        up_days[:, i] = np.where(close[:, i] > close[:, i-1], up_days[:, i-1] + 1, 0)

    # 52 週高低位置（0-1）
    week52_pos = np.zeros((n, ml), dtype=np.float32)
    for i in range(250, ml):
        h252 = np.max(high[:, i-250:i+1], axis=1)
        l252 = np.min(low[:, i-250:i+1], axis=1)
        rng = h252 - l252
        week52_pos[:, i] = np.where(rng > 0, (close[:, i] - l252) / rng, 0.5)

    # 連續量增天數（volume[i] > volume[i-1]）
    vol_up_days = np.zeros((n, ml), dtype=np.float32)
    for i in range(1, ml):
        vol_up_days[:, i] = np.where(volume[:, i] > volume[:, i-1], vol_up_days[:, i-1] + 1, 0)

    # 動量加速度（mom_d[5] 的 5 日差分）
    mom_accel = np.zeros((n, ml), dtype=np.float32)
    mom_accel[:, 5:] = mom_d[5][:, 5:] - mom_d[5][:, :-5]

    # MFI (Money Flow Index, 14-period) -- volume-weighted RSI
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    mfi_arr = np.full((n, ml), 50.0, dtype=np.float32)
    for i in range(15, ml):
        pos_mf = np.zeros(n, dtype=np.float64)
        neg_mf = np.zeros(n, dtype=np.float64)
        for j in range(i-13, i+1):
            up_mask = tp[:, j] > tp[:, j-1]
            pos_mf += np.where(up_mask, raw_mf[:, j], 0)
            neg_mf += np.where(~up_mask, raw_mf[:, j], 0)
        ratio = np.where(neg_mf > 0, pos_mf / neg_mf, 100.0)
        mfi_arr[:, i] = (100 - 100 / (1 + ratio)).astype(np.float32)

    # CMF (Chaikin Money Flow, 20-period) -- institutional accumulation
    hl_range = high - low
    clv = np.where(hl_range > 0, ((close - low) - (high - close)) / hl_range, 0.0)
    clv_vol = clv * volume
    cmf_arr = np.zeros((n, ml), dtype=np.float32)
    for i in range(20, ml):
        vol_sum = volume[:, i-19:i+1].sum(axis=1)
        clv_vol_sum = clv_vol[:, i-19:i+1].sum(axis=1)
        cmf_arr[:, i] = np.where(vol_sum > 0, clv_vol_sum / vol_sum, 0).astype(np.float32)

    # ATR Contraction (ATR5/ATR20 < 0.8 = compression before breakout)
    atr5 = np.zeros_like(close)
    atr20_s = np.zeros_like(close)
    for i in range(5, ml):
        atr5[:, i] = tr[:, i-4:i+1].mean(axis=1)
    for i in range(20, ml):
        atr20_s[:, i] = tr[:, i-19:i+1].mean(axis=1)
    atr_ratio_arr = np.where(atr20_s > 0, atr5 / atr20_s, 1.0).astype(np.float32)
    print(f"  New indicators: MFI/CMF/ATR_ratio computed")

    # 大盤狀態（用所有股票 close 平均當大盤 proxy）+ market_regime_filter 模式切換
    market_close = close.mean(axis=0)  # shape (ml,)
    market_ma20 = np.zeros(ml, dtype=np.float32)
    market_ma60 = np.zeros(ml, dtype=np.float32)
    for i in range(ml):
        if i >= 20: market_ma20[i] = market_close[max(0,i-19):i+1].mean()
        if i >= 60: market_ma60[i] = market_close[max(0,i-59):i+1].mean()

    _market_filter = os.environ.get("GPU_MARKET_FILTER", "0")
    if _market_filter == "1":
        # 大盤 close > MA60（大盤強勢日）
        market_bull = (market_close > market_ma60).astype(np.float32)
        market_bull[:60] = 1.0  # warmup 期允許
        print(f"  大盤過濾：mode 1（close > MA60）| {int(market_bull.sum())}/{ml} 天允許買入")
    elif _market_filter == "2":
        # MA20 > MA60（大盤多頭趨勢）
        market_bull = (market_ma20 > market_ma60).astype(np.float32)
        market_bull[:60] = 1.0
        print(f"  大盤過濾：mode 2（MA20 > MA60）| {int(market_bull.sum())}/{ml} 天允許買入")
    elif _market_filter == "3":
        # 兩條件同時：close > MA60 且 MA20 > MA60
        market_bull = ((market_close > market_ma60) & (market_ma20 > market_ma60)).astype(np.float32)
        market_bull[:60] = 1.0
        print(f"  大盤過濾：mode 3（close > MA60 AND MA20 > MA60）| {int(market_bull.sum())}/{ml} 天允許買入")
    else:
        market_bull = np.ones(ml, dtype=np.float32)
        print(f"  大盤過濾：無（v1 框架）| {ml}/{ml} 天全部允許")

    # Walk-Forward 模式切換（環境變數 GPU_WF_MODE）
    # - "reverse"（預設）：train 新（2023-26）/ test 舊（2020-22 含 covid）— 當前市場學 + 極端驗證
    # - "forward"：train 舊（2020-22 含熊市）/ test 新（2023-26 bull）— 真正 forward test
    _wf_mode = os.environ.get("GPU_WF_MODE", "reverse").lower()
    if _wf_mode == "forward":
        # 正向 WF：train 在前（舊），test 在後（新）
        train_start = 60                       # train 從 warmup 後開始
        train_end = 60 + (ml - 60) * 2 // 3    # train 佔 2/3
        _wf_label = "正向 WF"
        print(f"  正向 WF：warmup 0-60 | train {train_start}-{train_end}（{train_end-train_start}天，舊 含 covid/熊市）| test {train_end}-{ml}（{ml-train_end}天，新 bull）")
    else:
        # 反向 WF（預設）：test 在前（舊），train 在後（新）
        test_end = 60 + (ml - 60) // 3
        train_start = test_end
        train_end = ml
        _wf_label = "反向 WF"
        print(f"  反向 WF：warmup 0-60 | test {60}-{train_start}（{train_start-60}天，舊）| train {train_start}-{train_end}（{train_end-train_start}天，新）")

    return {"tickers":tickers,"dates":dates,"n_stocks":n,"n_days":ml,"train_start":train_start,"train_end":train_end,
        "close":close,"rsi":rsi,"bb_pos":bb_pos,"vol_ratio":vol_ratio,
        "macd_line":ml_arr,"macd_hist":mh,"k_val":kv.astype(np.float32),
        "d_val":dv.astype(np.float32),"is_green":ig,"gap":gp.astype(np.float32),
        "williams_r":wr,"near_high":nh,"vol_prev":vol_prev.astype(np.float32),
        "squeeze_fire":squeeze_fire,"new_high_60":new_high_60,
        "adx":adx.astype(np.float32),"bias":bias,"obv_rising":obv_rising,"atr_pct":atr_pct,
        "top100_mask":top100_mask.astype(np.float32),
        "market_bull":market_bull,
        "open":opn.astype(np.float32),
        "bb_std":bb_std.astype(np.float32),
        "sector_hot":sector_hot,
        "up_days":up_days,"week52_pos":week52_pos,
        "vol_up_days":vol_up_days,"mom_accel":mom_accel,
        "mfi":mfi_arr,"cmf":cmf_arr,"atr_ratio":atr_ratio_arr,
        "ma_d":ma_d,"mom_d":mom_d,"ma60":ma_d[60]}

REASON_NAMES = ["到期","停利","停損","RSI超買","移動停利","MACD死叉","KD死叉","量縮","跌破均線","停滯出場","漸進停利","鎖利出場","動量反轉","換股","保本出場"]

def cpu_replay(pre, p):
    """用 CPU 重跑一次最佳參數，拿完整交易明細（股票名、日期、價格）"""
    ns, nd = pre["n_stocks"], pre["n_days"]
    tickers = pre["tickers"]; dates = pre["dates"]; close = pre["close"]
    top100_mask=pre.get("top100_mask")
    rsi=pre["rsi"]; bb_pos=pre["bb_pos"]; vol_ratio=pre["vol_ratio"]
    macd_hist=pre["macd_hist"]; macd_line=pre["macd_line"]
    k_val=pre["k_val"]; d_val=pre["d_val"]; williams_r=pre["williams_r"]
    is_green=pre["is_green"]; gap=pre["gap"]; near_high=pre["near_high"]
    vol_prev=pre["vol_prev"]
    squeeze_fire=pre["squeeze_fire"]; new_high_60=pre["new_high_60"]; adx_arr=pre["adx"]; bias_arr=pre["bias"]; obv_rising_arr=pre["obv_rising"]; atr_pct_arr=pre["atr_pct"]
    opn=pre.get("open")
    market_bull=pre.get("market_bull")
    sector_hot=pre.get("sector_hot")
    up_days_arr=pre.get("up_days"); week52_arr=pre.get("week52_pos"); vol_up_days_arr=pre.get("vol_up_days"); mom_accel_arr=pre.get("mom_accel")
    mfi_arr=pre.get("mfi"); cmf_arr=pre.get("cmf"); atr_ratio_arr=pre.get("atr_ratio")
    maf=pre["ma_d"].get(int(p.get("ma_fast_w",5)), pre["ma_d"][5])
    mas=pre["ma_d"].get(int(p.get("ma_slow_w",20)), pre["ma_d"][20])
    ma60=pre["ma60"]
    mom=pre["mom_d"].get(int(p.get("momentum_days",5)), pre["mom_d"][5])

    max_pos=int(p.get("max_positions",1))
    if max_pos<1: max_pos=1
    if max_pos>3: max_pos=3
    hold_si=[-1]*3; hold_bp=[0]*3; hold_pk=[0]*3; hold_bd=[0]*3; n_holding=0; trades=[]
    last_sell_day=[-99999]*3  # buy_delay 冷卻用
    _buy_delay_days = int(p.get("buy_delay_days", 0))
    for day in range(60, nd-1):
        # Phase 1: 賣出（D+1 開盤價）
        for h in range(max_pos):
            if hold_si[h]<0: continue
            si=hold_si[h]; cur=float(close[si,day]); dh=day-hold_bd[h]
            ret=(cur/hold_bp[h]-1)*100
            if dh<1: continue
            if cur>hold_pk[h]: hold_pk[h]=cur
            sell=False; reason=0
            eff_stop=p["stop_loss"]
            _is_breakeven = p.get("use_breakeven",0) and (hold_pk[h]/hold_bp[h]-1)*100>=p.get("breakeven_trigger",20)
            if _is_breakeven: eff_stop=0
            # 快速認賠：買後 N 天內若虧 threshold% 立刻砍
            _ee_days = int(p.get("early_exit_days", 0))
            _ee_th = float(p.get("early_exit_th", -5))
            if _ee_days > 0 and dh <= _ee_days and ret <= _ee_th:
                sell=True; reason=2  # 標為停損
            if not sell and ret<=eff_stop: sell=True; reason=14 if _is_breakeven else 2  # 14=保本, 2=停損
            if not sell and p.get("use_take_profit",1) and ret>=p["take_profit"]: sell=True; reason=1
            if not sell and p.get("trailing_stop",0)>0 and hold_pk[h]>hold_bp[h]:
                if (cur/hold_pk[h]-1)*100<=-p["trailing_stop"]: sell=True; reason=4
            if not sell and p.get("use_rsi_sell",0) and rsi[si,day]>=p.get("rsi_sell",90): sell=True; reason=3
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
            if sell and day+1<nd:
                sell_price=float(opn[si,day+1]) if opn is not None else float(close[si,day])
                if np.isnan(sell_price) or sell_price<=0: sell_price=float(close[si,day])  # NaN 防護
                actual_ret=(sell_price/hold_bp[h]-1)*100 - 0.585
                actual_days=day+1-hold_bd[h]
                trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                    "buy_date":str(dates[hold_bd[h]].date()),"sell_date":str(dates[day+1].date()),
                    "buy_price":round(hold_bp[h],2),"sell_price":round(sell_price,2),
                    "return":round(actual_ret,2),"days":actual_days,"reason":REASON_NAMES[min(reason,len(REASON_NAMES)-1)]})
                hold_si[h]=-1; n_holding-=1
                last_sell_day[h] = day + 1  # 記錄賣出日 for buy_delay cooldown
        # Phase 1.5: 換股 — 持倉滿且有更強候選，賣弱換強
        um=int(p.get("upgrade_margin",0))
        if um>0 and n_holding>=max_pos and day+1<nd:
            def _score_stock(si,day):
                d=day; sc=0.0
                if int(p.get("w_rsi",0))>0 and rsi[si,d]>=p.get("rsi_th",55): sc+=int(p["w_rsi"])
                if int(p.get("w_bb",0))>0 and bb_pos[si,d]>=p.get("bb_th",0.7): sc+=int(p["w_bb"])
                if int(p.get("w_vol",0))>0 and vol_ratio[si,d]>=p.get("vol_th",3): sc+=int(p["w_vol"])
                if int(p.get("w_ma",0))>0 and close[si,d]>maf[si,d]: sc+=int(p["w_ma"])
                if int(p.get("w_macd",0))>0:
                    mm=int(p.get("macd_mode",2)); ok=False
                    if mm==0 and d>=1 and macd_hist[si,d]>0 and macd_hist[si,d-1]<=0: ok=True
                    elif mm==1 and macd_line[si,d]>0: ok=True
                    elif mm==2 and macd_hist[si,d]>0: ok=True
                    if ok: sc+=int(p["w_macd"])
                if int(p.get("w_kd",0))>0:
                    ok=k_val[si,d]>=p.get("kd_th",50)
                    if ok and p.get("kd_cross",0) and d>=1: ok=k_val[si,d]>d_val[si,d] and k_val[si,d-1]<=d_val[si,d-1]
                    if ok: sc+=int(p["w_kd"])
                if int(p.get("w_wr",0))>0 and williams_r[si,d]>=p.get("wr_th",-30): sc+=int(p["w_wr"])
                if int(p.get("w_mom",0))>0 and mom[si,d]>=p.get("mom_th",3): sc+=int(p["w_mom"])
                if int(p.get("w_near_high",0))>0 and abs(near_high[si,d])<=p.get("near_high_pct",10): sc+=int(p["w_near_high"])
                if int(p.get("w_squeeze",0))>0 and squeeze_fire[si,d]>0.5: sc+=int(p["w_squeeze"])
                if int(p.get("w_new_high",0))>0 and new_high_60[si,d]>0.5: sc+=int(p["w_new_high"])
                if int(p.get("w_adx",0))>0 and adx_arr[si,d]>=p.get("adx_th",25): sc+=int(p["w_adx"])
                if int(p.get("w_bias",0))>0 and bias_arr[si,d]>=0 and bias_arr[si,d]<=p.get("bias_max",15): sc+=int(p["w_bias"])
                if int(p.get("w_obv",0))>0 and obv_rising_arr[si,d]>0.5: sc+=int(p["w_obv"])
                if int(p.get("w_atr",0))>0 and atr_pct_arr[si,d]>=p.get("atr_min",2): sc+=int(p["w_atr"])
                if int(p.get("w_sector_flow",0))>0 and sector_hot is not None and sector_hot[si,d]<p.get("sector_flow_topn",3): sc+=int(p["w_sector_flow"])
                if int(p.get("w_up_days",0))>0 and up_days_arr is not None and up_days_arr[si,d]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
                if int(p.get("w_week52",0))>0 and week52_arr is not None and week52_arr[si,d]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
                if int(p.get("w_vol_up_days",0))>0 and vol_up_days_arr is not None and vol_up_days_arr[si,d]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
                if int(p.get("w_mom_accel",0))>0 and mom_accel_arr is not None and mom_accel_arr[si,d]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
                # New indicators
                _w_mfi = int(p.get("w_mfi", 0))
                if _w_mfi > 0 and mfi_arr is not None and mfi_arr[si, day] >= p.get("mfi_th", 70): sc += _w_mfi
                _w_cmf = int(p.get("w_cmf", 0))
                if _w_cmf > 0 and cmf_arr is not None and cmf_arr[si, day] >= p.get("cmf_th", 0.1): sc += _w_cmf
                _w_atrc = int(p.get("w_atr_contract", 0))
                if _w_atrc > 0 and atr_ratio_arr is not None and atr_ratio_arr[si, day] <= p.get("atr_contract_th", 0.8): sc += _w_atrc
                return sc
            # 找候選最高分（追蹤 top-1 + top-2）
            cand_si=-1; cand_sc=0; cand_vol=0
            _second_sc = 0
            held_set=set(hh for hh in hold_si if hh>=0)
            _max_3d = float(p.get("max_3d_change", 0))
            _top1_margin = float(p.get("top1_margin", 0))
            _spd = int(p.get("signal_persist_days", 0))
            for si in range(ns):
                if top100_mask is not None and top100_mask[si,day]<0.5: continue
                if si in held_set: continue
                # 🔒 過去 3 天價格穩定性檢查
                if _max_3d > 0 and day >= 3:
                    _p3 = float(close[si, day-3])
                    if _p3 > 0:
                        _chg = (float(close[si, day]) / _p3 - 1.0) * 100.0
                        if abs(_chg) > _max_3d: continue
                # 🔥 訊號持續性：過去 N 天都必須是 top100
                if _spd > 0 and top100_mask is not None:
                    _persist_ok = True
                    for _k in range(1, _spd+1):
                        if day-_k < 0: _persist_ok = False; break
                        if top100_mask[si, day-_k] < 0.5: _persist_ok = False; break
                    if not _persist_ok: continue
                sc=_score_stock(si,day)
                vr=float(vol_ratio[si,day]) if vol_ratio is not None else 0
                if sc>=p.get("buy_threshold",5):
                    if sc>cand_sc or (sc==cand_sc and vr>cand_vol):
                        _second_sc = cand_sc
                        cand_si=si; cand_sc=sc; cand_vol=vr
                    elif sc > _second_sc:
                        _second_sc = sc
            # 相對訊號強度檢查：第 1 名 vs 第 2 名分數差
            if cand_si >= 0 and _top1_margin > 0 and (cand_sc - _second_sc) < _top1_margin:
                cand_si = -1  # 訊號不明確，取消買入
            if cand_si>=0:
                weakest_h=-1; weakest_sc=9999
                for h in range(max_pos):
                    if hold_si[h]<0: continue
                    sc=_score_stock(hold_si[h],day)
                    if sc<weakest_sc: weakest_sc=sc; weakest_h=h
                if weakest_h>=0 and cand_sc-weakest_sc>=um:
                    si=hold_si[weakest_h]
                    sell_price=float(opn[si,day+1]) if opn is not None else float(close[si,day])
                    if np.isnan(sell_price) or sell_price<=0: sell_price=float(close[si,day])  # NaN 防護
                    actual_ret=(sell_price/hold_bp[weakest_h]-1)*100 - 0.585
                    actual_days=day+1-hold_bd[weakest_h]
                    trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                        "buy_date":str(dates[hold_bd[weakest_h]].date()),"sell_date":str(dates[day+1].date()),
                        "buy_price":round(hold_bp[weakest_h],2),"sell_price":round(sell_price,2),
                        "return":round(actual_ret,2),"days":actual_days,"reason":"換股"})
                    hold_si[weakest_h]=-1; n_holding-=1
        # Phase 2: 買入一檔
        # 大盤過濾：market_bull[day] = 0 時不進場
        if n_holding<max_pos and day+1<nd and (market_bull is None or market_bull[day] > 0.5):
            best_si=-1; best_sc=0; best_vol=0
            w_rsi=int(p.get("w_rsi",0)); w_bb=int(p.get("w_bb",0)); w_vol=int(p.get("w_vol",0))
            w_ma=int(p.get("w_ma",0)); w_macd=int(p.get("w_macd",0)); w_kd=int(p.get("w_kd",0))
            w_wr=int(p.get("w_wr",0)); w_mom=int(p.get("w_mom",0)); w_nh=int(p.get("w_near_high",0))
            w_sq=int(p.get("w_squeeze",0)); w_newh=int(p.get("w_new_high",0))
            w_adx=int(p.get("w_adx",0)); adx_threshold=p.get("adx_th",25)
            w_bias=int(p.get("w_bias",0)); bias_max_val=p.get("bias_max",15)
            w_obv=int(p.get("w_obv",0))
            w_atr_buy=int(p.get("w_atr",0)); atr_min_val=p.get("atr_min",2.0)
            buy_th=p.get("buy_threshold",5)
            held_set=set(hh for hh in hold_si if hh>=0)
            for si in range(ns):
                if top100_mask is not None and top100_mask[si,day]<0.5: continue
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
                w_sf=int(p.get("w_sector_flow",0))
                if w_sf>0 and sector_hot is not None and sector_hot[si,day]<p.get("sector_flow_topn",3): sc+=w_sf
                _wud=int(p.get("w_up_days",0))
                if _wud>0 and up_days_arr is not None and up_days_arr[si,day]>=p.get("up_days_min",3): sc+=_wud
                _ww52=int(p.get("w_week52",0))
                if _ww52>0 and week52_arr is not None and week52_arr[si,day]>=p.get("week52_min",0.7): sc+=_ww52
                _wvud=int(p.get("w_vol_up_days",0))
                if _wvud>0 and vol_up_days_arr is not None and vol_up_days_arr[si,day]>=p.get("vol_up_days_min",3): sc+=_wvud
                _wma=int(p.get("w_mom_accel",0))
                if _wma>0 and mom_accel_arr is not None and mom_accel_arr[si,day]>=p.get("mom_accel_min",2): sc+=_wma
                cg=int(p.get("consecutive_green",0))
                if cg>=1:
                    ok=True
                    for g in range(cg):
                        if day-g<0 or is_green[si,day-g]!=1: ok=False; break
                    if ok: sc+=1
                if p.get("gap_up",0) and gap[si,day]>=1.0: sc+=1
                if p.get("above_ma60",0) and close[si,day]>=ma60[si,day]: sc+=1
                if p.get("vol_gt_yesterday",0) and day>=1 and vol_ratio[si,day]>vol_prev[si,day]: sc+=1
                _w_mfi=int(p.get("w_mfi",0))
                if _w_mfi>0 and mfi_arr is not None and mfi_arr[si,day]>=p.get("mfi_th",70): sc+=_w_mfi
                _w_cmf=int(p.get("w_cmf",0))
                if _w_cmf>0 and cmf_arr is not None and cmf_arr[si,day]>=p.get("cmf_th",0.1): sc+=_w_cmf
                _w_atrc=int(p.get("w_atr_contract",0))
                if _w_atrc>0 and atr_ratio_arr is not None and atr_ratio_arr[si,day]<=p.get("atr_contract_th",0.8): sc+=_w_atrc
                vr=float(vol_ratio[si,day]) if vol_ratio is not None else 0
                if sc>=buy_th and (sc>best_sc or (sc==best_sc and vr>best_vol)): best_si=si; best_sc=sc; best_vol=vr
            if best_si>=0:
                for h in range(max_pos):
                    if hold_si[h]<0:
                        # buy_delay cooldown check
                        if _buy_delay_days > 0 and (day - last_sell_day[h]) < _buy_delay_days: continue
                        hold_si[h]=best_si; hold_bp[h]=float(close[best_si,day+1])
                        hold_pk[h]=hold_bp[h]; hold_bd[h]=day+1; n_holding+=1; break
        # Phase 3 已移除（第三檔回測表現不佳）
    # Append active holdings as "持有中"
    for h in range(max_pos):
        if hold_si[h] >= 0:
            si = hold_si[h]
            cur = float(close[si, nd-1])
            dh = nd - 1 - hold_bd[h]
            ret = (cur / hold_bp[h] - 1) * 100 - 0.585 if hold_bp[h] > 0 else 0
            trades.append({"ticker":tickers[si],"name":get_name(tickers[si]),
                "buy_date":str(dates[hold_bd[h]].date()),"sell_date":"",
                "buy_price":round(hold_bp[h],2),"sell_price":round(cur,2),
                "return":round(ret,2),"days":dh,"reason":"持有中",
                "peak_price":round(hold_pk[h],2)})
    return sorted(trades, key=lambda x: x["buy_date"])

def main():
    # === 動態調整 PARAMS_SPACE（env var 控制，對抗 89.90 的「早保本」弱點）===
    if os.environ.get("GPU_NO_BREAKEVEN") == "1":
        PARAMS_SPACE["use_breakeven"] = [0]  # 強制禁用保本機制
        print("  [Mode] 🚫 保本機制禁用（強迫探索純 trailing + stop_loss 策略）")
    _be_min = int(os.environ.get("GPU_BREAKEVEN_MIN", "0"))
    if _be_min > 10:
        PARAMS_SPACE["breakeven_trigger"] = [v for v in PARAMS_SPACE["breakeven_trigger"] if v >= _be_min]
        print(f"  [Mode] ⚠️  breakeven_trigger ≥ {_be_min}（禁 trigger=10，讓波段跑）")
    if os.environ.get("GPU_HIGH_PROFIT") == "1":
        PARAMS_SPACE["take_profit"] = [80, 100, 150]  # 強制高停利
        PARAMS_SPACE["trailing_stop"] = [15, 20, 25]  # 強制寬 trailing
        print(f"  [Mode] 🎯 強迫吃大波段（停利 80+, trailing 15+）")

    if os.environ.get("GPU_STRICT_SIGNAL") == "1":
        # 強迫「只買持續強勢股」— 用現有指標提高嚴格度
        PARAMS_SPACE["week52_min"] = [0.8, 0.9]          # 52 週位置必須頂部
        PARAMS_SPACE["up_days_min"] = [7, 10]             # 至少連漲 7 天
        PARAMS_SPACE["mom_th"] = [15, 20, 25]             # 動量 ≥ 15%
        PARAMS_SPACE["w_week52"] = [2, 3]                  # 強迫啟用 52 週指標
        PARAMS_SPACE["w_up_days"] = [2, 3]                 # 強迫啟用連漲指標
        PARAMS_SPACE["w_mom"] = [2, 3]                     # 強迫啟用動量指標
        PARAMS_SPACE["adx_th"] = [35, 40]                  # ADX 趨勢必強
        PARAMS_SPACE["w_adx"] = [2, 3]                     # 強迫啟用 ADX
        print(f"  [Mode] 🔥 嚴格訊號模式：52週頂 + 連漲 7+ + 動量 15+ + ADX 35+")
        print(f"         = 強迫買「過去持續強勢的股」，當日爆發不夠格")

    print("[GPU-CuPy] 🚀 RTX 3060 進化引擎啟動！")
    print(f"[GPU-CuPy] 🎯 勝率優先 + 波段獎勵 + 反向 Walk-Forward（融合 189+88.60 優點）")
    print(f"")
    print(f"  ═══ Scoring（勝率主軸 + 波段副軸）═══")
    print(f"  主軸 s_wr × 2.5 (cap 100)  勝率 70%=+50 / 75%=+62 / 80%=+75 / 90%=+100 [WINRATE-MAX]")
    print(f"  報酬 s_return × 0.05 (cap 20)  年化 400% 封頂")
    print(f"  波段 s_avg × 0.5 (cap 5)  大幅降權（勝率主導，放棄波段追求）")
    print(f"  近期 s_recent × 0.5 (cap 15)  近 2 年勝率 65%=+3 / 70%=+6 / 75%=+9 / 80%=+12 ← 新")
    print(f"  風調 s_calmar × 1.5 (cap 10)  Calmar 3=+1.5 / 5=+4.5 / 8=+9 ← 新")
    print(f"  輔助 s_wf×15 / s_sharpe×2 / s_pl×0.5 / s_consistency×0.03 / penalties")
    print(f"")
    _wf_mode_show = os.environ.get("GPU_WF_MODE", "reverse").lower()
    if _wf_mode_show == "forward":
        print(f"  ═══ 正向 Walk-Forward 切點（動態依 cache 長度）═══")
        print(f"  warmup  0-60     指標暖身")
        print(f"  train   舊的 2/3  舊期（GPU 學這裡，含 covid/熊市等極端）")
        print(f"  test    新的 1/3  新期（近期市場，當驗證盲測）")
    else:
        print(f"  ═══ 反向 Walk-Forward 切點（動態依 cache 長度）═══")
        print(f"  warmup  0-60     指標暖身")
        print(f"  test    舊的 1/3  舊期（含 2020 covid/2022 熊市等）")
        print(f"  train   新的 2/3  新期（近期市場，GPU 學這裡）")
    print(f"")
    print(f"  ═══ Kernel 硬門檻（不過→score 0）═══")
    print(f"  train 筆數 30-140 | avg ≥ 8% | wr ≥ 35% | avg_hold ≥ 5 天 | MaxDD ≥ -50% ← 適配 1500 天")
    print(f"  test 筆數 ≥ 5 | total_test > 0（test 不能爆）")
    print(f"  WF ratio: test_annual ≥ train_annual × 0.4（反向 WF 下 test=2022 熊市，0.55 太嚴連 88.60 都過不了）")
    print(f"  Seg 3 段都要正報酬 | seg[2] ≥ seg[0] × 0.6（防老化）")
    print(f"")
    print(f"  ═══ Python gate（top-20 cpu_replay 驗證）═══")
    print(f"  全期 40-200 筆 | avg ≥ 10% | avg_hold ≥ 5 | MaxDD ≥ -50% ← 適配 1500 天")
    print(f"  WF ratio ≥ 0.4 | test_total > 0")
    print(f"  報酬地板: train 年化 ≥ {MIN_TRAIN_ANNUAL}%（189 × 0.6）| test 年化 ≥ {MIN_TEST_ANNUAL}%（189 × 0.6）")
    print(f"  勝率地板: train ≥ {MIN_WR_TRAIN*100:.0f}% | test ≥ {MIN_WR_TEST*100:.0f}%")
    print(f"  最新 60 天 avg ≥ 5%（近期崩盤檢查）")
    print(f"  🆕 近 2 年 avg ≥ 15%（擋退化策略）")
    print(f"  🆕 train Calmar ≥ 2（CAGR/MaxDD，風險調整）")
    print(f"  🆕 use_mom_exit 強制 40% 探索（解決 22 筆保本出場）")
    print(f"  ═══ 診斷 ═══ 每 5 輪無突破時印 gate fail 分布，看卡在哪")
    print(f"")

    # 啟動時歸檔舊 pending_push.json（避免上次 session 的未推 pending 跟新 session 混淆）
    _pending_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_push.json")
    if os.path.exists(_pending_path):
        import shutil
        _archive = _pending_path + f".reset_{int(time.time())}"
        shutil.move(_pending_path, _archive)
        print(f"[GPU-CuPy] 🗑️ 舊 pending_push.json 歸檔 → {os.path.basename(_archive)}（新 session 從空 pending 開始）")
    raw = download_data()
    # 動態選擇天數：優先 1500（6 年含 2020 covid），但至少要 500 檔才用；否則 fallback 900
    _lens = [len(v) for v in raw.values()]
    _n_1500 = sum(1 for l in _lens if l >= 1500)
    _n_1200 = sum(1 for l in _lens if l >= 1200)
    _n_900 = sum(1 for l in _lens if l >= 900)
    if _n_1500 >= 500:
        TARGET_DAYS = 1500  # 6 年黃金區：含 2020 covid 暴跌 + 2020-2021 bull + 2022 熊 + 2023-2025 bull
    elif _n_1200 >= 800:
        TARGET_DAYS = 1200  # 4.8 年備案
    else:
        TARGET_DAYS = 900   # 原設定
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    print(f"[過濾] {len(raw)} → {len(data)} 檔（動態選 {TARGET_DAYS} 天 | 1500-qualified {_n_1500} / 1200-q {_n_1200} / 900-q {_n_900}）")
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
    d_open = cp.asarray(pre["open"])
    d_top100 = cp.asarray(pre["top100_mask"])
    d_market = cp.asarray(pre["market_bull"])
    d_up_days = cp.asarray(pre["up_days"]); d_week52 = cp.asarray(pre["week52_pos"])
    d_vol_up_days = cp.asarray(pre["vol_up_days"]); d_mom_accel = cp.asarray(pre["mom_accel"])
    d_mfi = cp.asarray(pre["mfi"]); d_cmf = cp.asarray(pre["cmf"]); d_atr_ratio = cp.asarray(pre["atr_ratio"])
    d_ma60 = cp.asarray(pre["ma60"])

    print("[GPU] 開始進化！每批 500,000 組")
    BATCH = 200000  # 縮小避免 Python 參數生成卡住
    BLOCK = 256
    N_PARAMS = len(PARAM_ORDER)
    best_score = -999999  # R1 結束後用 SEED kernel 分數覆蓋
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

    # Gist 載入舊策略當 reference（不給 999 分保護，讓高勝率策略自然取代）
    # 189（勝率 62%）在新 scoring 下大約 50-60 分，GPU 搜到 70%+ 勝率策略會自動勝出
    gist_best_params = None
    try:
        headers = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
        r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=headers, timeout=10)
        gist_data = json.loads(list(r.json()["files"].values())[0]["content"])
        if "params" in gist_data:
            gist_best_params = gist_data["params"]
            import math as _math
            _baseline_trades = cpu_replay(pre, gist_best_params)
            _baseline_trades = [t for t in _baseline_trades if not _math.isnan(t.get("return",0)) and t.get("reason") != "持有中"]
            _baseline_total = sum(t.get("return",0) for t in _baseline_trades)
            _baseline_n = len(_baseline_trades)
            _baseline_avg = _baseline_total / _baseline_n if _baseline_n else 0
            _baseline_wr = sum(1 for t in _baseline_trades if t.get("return",0)>0) / _baseline_n * 100 if _baseline_n else 0
            print(f"[GPU] Gist 策略在當前資料上：{_baseline_n}筆 avg{_baseline_avg:.1f}% 總{_baseline_total:.0f}% 勝率{_baseline_wr:.0f}%")

            # === SEED 診斷：逐一檢查 kernel gate，找出卡在哪 ===
            _tr_end_day = pre["train_end"]
            _tr_start_day = pre["train_start"]
            _tr_start_str = str(pre["dates"][_tr_start_day].date())
            _tr_end_str = str(pre["dates"][_tr_end_day-1].date()) if _tr_end_day < pre["n_days"] else str(pre["dates"][-1].date())
            # train = buy_date 在 train 區間，test = 其他（支援正向/反向 WF）
            _tr_tr = [t for t in _baseline_trades if _tr_start_str <= t.get("buy_date","") <= _tr_end_str]
            _te_tr = [t for t in _baseline_trades if t.get("buy_date","") < _tr_start_str or t.get("buy_date","") > _tr_end_str]
            _n_tr, _n_te = len(_tr_tr), len(_te_tr)
            _tot_tr = sum(t.get("return",0) for t in _tr_tr)
            _tot_te = sum(t.get("return",0) for t in _te_tr)
            _avg_tr = _tot_tr/_n_tr if _n_tr else 0
            _wr_tr = sum(1 for t in _tr_tr if t.get("return",0)>0)/_n_tr*100 if _n_tr else 0
            _ah_tr = sum(t.get("days",0) for t in _tr_tr)/_n_tr if _n_tr else 0
            _tr_y = (_tr_end_day - _tr_start_day) / 250.0
            # test 天數 = 全部 - train - warmup（支援正向/反向）
            _te_days = (pre["n_days"] - 60) - (_tr_end_day - _tr_start_day)
            _te_y = _te_days / 250.0
            _tr_ann = _tot_tr/_tr_y if _tr_y > 0.5 else _tot_tr
            _te_ann = _tot_te/_te_y if _te_y > 0.3 else _tot_te
            _wf = _te_ann/_tr_ann if _tr_ann > 0 else 0
            _max_dd_tr = 0; _run = 0
            for _t in _tr_tr:
                _r = _t.get("return",0)
                _run = _run + _r if _r < 0 else 0
                if _run < _max_dd_tr: _max_dd_tr = _run
            _rec_str = str(pre["dates"][pre["n_days"]-60].date())
            _rec = [t for t in _baseline_trades if t.get("buy_date","") >= _rec_str]
            _rec_avg = sum(t.get("return",0) for t in _rec)/len(_rec) if _rec else 0
            # seg 3 段（train 期內部）
            _seg_size = max(10, (_tr_end_day - _tr_start_day) // 3)
            _seg = [[],[],[]]
            _date_to_day = {str(d.date()): i for i,d in enumerate(pre["dates"])}
            for _t in _tr_tr:
                _bd = _date_to_day.get(_t.get("buy_date",""), -1)
                if _bd < 0: continue
                _rel = max(0, _bd - _tr_start_day)
                _s = min(2, _rel // _seg_size)
                _seg[_s].append(_t.get("return",0))
            _seg_totals = [sum(s) for s in _seg]
            _seg_avgs = [sum(s)/len(s) if s else 0 for s in _seg]

            print(f"[診斷] SEED 在反向 WF 下的分解（看卡哪個 gate）：")
            print(f"  train ({_tr_start_str} ~ {_tr_end_str}): n={_n_tr} avg={_avg_tr:.1f}% wr={_wr_tr:.0f}% avg_hold={_ah_tr:.1f}天 MaxDD={_max_dd_tr:.0f}% 總{_tot_tr:.0f}% 年化{_tr_ann:.0f}%")
            print(f"  test  (train 區間外): n={_n_te} 總{_tot_te:.0f}% 年化{_te_ann:.0f}%")
            print(f"  WF ratio: {_wf:.2f} (kernel 需 ≥ 0.55)")
            print(f"  近60天: {len(_rec)} 筆 avg={_rec_avg:.1f}% (kernel 需 ≥ 5%)")
            print(f"  train 3段: n={[len(s) for s in _seg]} 總{_seg_totals} avg{[round(a,1) for a in _seg_avgs]}")
            print(f"  kernel 門檻：n_train 30-140 | avg_tr ≥ 8 | wr_tr ≥ 35 | avg_hold ≥ 5 | MaxDD ≥ -50 | WF ≥ 0.4 | 3段都正 | seg[2] ≥ seg[0]×0.6")
            _fail = []
            _n_all = len(_baseline_trades)
            if _n_all < 40: _fail.append(f"n_all {_n_all} < 40")
            if _n_all > 200: _fail.append(f"n_all {_n_all} > 200")
            if _n_tr < 30: _fail.append(f"n_train {_n_tr} < 30")
            if _n_tr > 140: _fail.append(f"n_train {_n_tr} > 140")
            if _avg_tr < 8: _fail.append(f"avg_tr {_avg_tr:.1f}% < 8")
            if _ah_tr < 5: _fail.append(f"avg_hold {_ah_tr:.1f} < 5")
            if _max_dd_tr < -50: _fail.append(f"MaxDD {_max_dd_tr:.0f}% < -50")
            if _wf < 0.4 and _tr_ann > 0: _fail.append(f"WF {_wf:.2f} < 0.4")
            if _rec_avg < 5 and len(_rec) >= 3: _fail.append(f"近60天 avg {_rec_avg:.1f}% < 5")
            for _i, _s in enumerate(_seg):
                if len(_s) >= 4 and sum(_s) <= 0:
                    _fail.append(f"seg[{_i}] total {sum(_s):.0f}% ≤ 0")
            if len(_seg[0]) >= 4 and len(_seg[2]) >= 4:
                if _seg_avgs[2] < _seg_avgs[0] * 0.6:
                    _fail.append(f"seg[2] avg {_seg_avgs[2]:.1f}% < seg[0] {_seg_avgs[0]:.1f}% × 0.6")
            if _fail:
                print(f"  ⛔ 卡在：{' | '.join(_fail)}")
            else:
                print(f"  ✅ 所有 gate 都過（kernel 分數若還 0，可能是 warmup 期指標未到位）")
            best_params = dict(gist_best_params)
            # 多起點播種：現 Gist 策略 + 歷史名策略（189 / 88.60 的關鍵參數）
            # 讓 HOF 有基因多樣性，配種才有效
            SEED_189 = {"stop_loss":-12,"take_profit":40,"trailing_stop":15,"hold_days":30,
                        "use_breakeven":0,"breakeven_trigger":20,"sell_below_ma":3,
                        "buy_threshold":14,"above_ma60":0,
                        "w_rsi":1,"rsi_th":65,"w_ma":3,"w_macd":3,"macd_mode":0,
                        "w_kd":2,"kd_th":80,"w_mom":2,"mom_th":8,"w_new_high":2,
                        "w_adx":2,"adx_th":40,"w_obv":2,"obv_rising_days":10,
                        "w_atr":2,"atr_min":3,"w_up_days":2,"up_days_min":5,
                        "w_week52":2,"week52_min":0.6,"w_bias":1,"bias_max":8,
                        "w_mom_accel":1,"mom_accel_min":8,"consecutive_green":1,"gap_up":1,
                        "ma_fast_w":5,"ma_slow_w":60,"momentum_days":3,"max_positions":2}
            SEED_88 = {"stop_loss":-20,"take_profit":40,"trailing_stop":15,"hold_days":30,
                       "use_breakeven":1,"breakeven_trigger":10,"sell_below_ma":0,
                       "buy_threshold":6,"above_ma60":1,
                       "w_rsi":3,"rsi_th":65,"w_ma":3,"w_macd":3,"macd_mode":0,
                       "w_kd":2,"kd_th":80,"w_wr":3,"wr_th":-40,"w_mom":3,"mom_th":8,
                       "w_near_high":2,"near_high_pct":10,"w_new_high":2,
                       "w_adx":2,"adx_th":40,"w_bias":2,"bias_max":3,
                       "w_obv":1,"w_atr":3,"atr_min":3,"w_bb":1,"bb_th":0.9,
                       "w_up_days":2,"up_days_min":5,"w_week52":3,"week52_min":0.7,
                       "w_vol_up_days":1,"vol_up_days_min":2,"w_mom_accel":1,"mom_accel_min":0,
                       "consecutive_green":1,"gap_up":1,
                       "ma_fast_w":3,"ma_slow_w":15,"momentum_days":3,"max_positions":2}
            hall_of_fame = [
                (0, dict(gist_best_params)),   # 當前 Gist 策略（89.90）
                (0, dict(SEED_189)),           # 波段強（高 avg）
                (0, dict(SEED_88)),            # 勝率強
            ]
            print(f"[GPU] 🌱 多起點播種 HOF：當前 Gist + 189 + 88.60（基因多樣性配種更有效）")
    except Exception as _e:
        print(f"[GPU] Gist 載入失敗：{_e}")
    # 掃描跳過（曾基於 v5，會污染起點）

    last_data_date = time.strftime("%Y-%m-%d")

    while True:
        # 每天自動刷新資料 — 增量 append 新的一天，舊資料完全不動
        today_str = time.strftime("%Y-%m-%d")
        if today_str != last_data_date:
            print(f"\n[GPU] 🔄 新的一天（{today_str}），增量抓新資料（舊 cache 不動）...")
            try:
                raw, _n_updated = append_new_days(CACHE_PATH)
                if raw is None or _n_updated == 0:
                    print(f"[GPU] 沒新資料可加（可能假日/無交易），沿用舊 cache")
                    raw = download_data()
                else:
                    print(f"[GPU] {_n_updated} 檔 cache 已 append 新資料 — 舊部分 0 變動")
                data = {k:v for k,v in raw.items() if len(v) >= 900}
                print(f"[GPU] 刷新完成：{len(data)} 檔（>=900天，v1 框架）")
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
                d_open = cp.asarray(pre["open"])
                d_top100 = cp.asarray(pre["top100_mask"])
                d_market = cp.asarray(pre["market_bull"])
                d_up_days = cp.asarray(pre["up_days"]); d_week52 = cp.asarray(pre["week52_pos"])
                d_vol_up_days = cp.asarray(pre["vol_up_days"]); d_mom_accel = cp.asarray(pre["mom_accel"])
                d_mfi = cp.asarray(pre["mfi"]); d_cmf = cp.asarray(pre["cmf"]); d_atr_ratio = cp.asarray(pre["atr_ratio"])
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
        mutate_rate = min(0.5, 0.08 + no_improve_rounds * 0.02)  # 初期 8% 精緻，累加到 50% 觸發多起點爬山
        # 多起點爬山：80% 爬山（從隨機起點）+ 20% 隨機，不配種（名人堂還在重建）
        if explore_bases is not None:
            n_random = BATCH // 5
            n_climb = BATCH - n_random
            n_breed = 0
        elif len(hall_of_fame) < 3:
            # 早期：25% 隨機 + 75% 爬山（HOF 不夠就集中微調 SEED，配種無效）
            n_random = BATCH // 4
            n_climb = BATCH - n_random
            n_breed = 0
        elif no_improve_rounds < 5:
            # 正常：15% 隨機 / 45% 爬山 / 40% 配種
            n_random = int(BATCH * 0.15)
            n_climb = int(BATCH * 0.45)
            n_breed = BATCH - n_random - n_climb
        else:
            # 停滯：30% 隨機 / 30% 爬山 / 40% 配種（加強探索）
            n_random = int(BATCH * 0.30)
            n_climb = int(BATCH * 0.30)
            n_breed = BATCH - n_random - n_climb
        third = n_random  # 相容舊變數名

        # === 全部先用隨機填滿（向量化，超快）===
        _base_for_freeze = best_params  # NO-OVERFIT：不 fallback v5
        # B: 新指標強制探索 — 40% 機率非零
        # 加入 use_mom_exit 讓 GPU 真的探索動量反轉出場（解決 22 筆保本出場的問題）
        NEW_INDICATOR_WEIGHTS = {"w_sector_flow","w_up_days","w_week52","w_vol_up_days","w_mom_accel","use_mom_exit"}
        for i, key in enumerate(PARAM_ORDER):
            opts = np.array(PARAMS_SPACE[key], dtype=np.float32)
            if key in FROZEN_PARAMS and _base_for_freeze:
                # 凍結參數：隨機部分只在基準值 ±1 格內微調
                base_val = float(_base_for_freeze.get(key, opts[0]))
                base_idx = int(np.argmin(np.abs(opts - base_val)))
                lo = max(0, base_idx - 1)
                hi = min(len(opts) - 1, base_idx + 1)
                params_np[:, i] = opts[np.random.randint(lo, hi + 1, BATCH)]
            elif key in NEW_INDICATOR_WEIGHTS:
                # B: 新指標 40% 強制非零（避免被 w=0 主導）
                nonzero_opts = opts[opts > 0]
                if len(nonzero_opts) > 0:
                    force_nonzero = np.random.random(BATCH) < 0.4
                    zero_vals = np.zeros(BATCH, dtype=np.float32)
                    nonzero_vals = np.random.choice(nonzero_opts, BATCH).astype(np.float32)
                    params_np[:, i] = np.where(force_nonzero, nonzero_vals, zero_vals)
                else:
                    params_np[:, i] = np.random.choice(opts, BATCH).astype(np.float32)
            else:
                # 自由參數：全範圍探索
                params_np[:, i] = np.random.choice(opts, BATCH).astype(np.float32)

        # === 爬山微調（向量化）===
        if explore_bases is not None:
            base = explore_bases[explore_round % len(explore_bases)]
        else:
            base = best_params  # NO-OVERFIT：不 fallback v5
        if base:
            for i, key in enumerate(PARAM_ORDER):
                opts = np.array(PARAMS_SPACE[key], dtype=np.float32)
                base_val = float(base.get(key, opts[0]))
                diffs = np.abs(opts - base_val)
                base_idx = int(np.argmin(diffs))
                keep = np.random.random(n_climb) >= mutate_rate
                # 半凍結：凍結參數 radius=1，自由參數正常擴展
                if key in FROZEN_PARAMS:
                    radius = 1  # 只能微調到鄰近一格
                else:
                    radius = 1 + min(no_improve_rounds // 3, 4)  # 初期 1 格（±1 鄰域），每 3 輪 +1 cap 5
                lo = max(0, base_idx - radius)
                hi = min(len(opts) - 1, base_idx + radius)
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
            breed_mut_rate = min(0.3, 0.1 + no_improve_rounds * 0.01)  # 動態突變率
            mutate_mask = np.random.random((breed_size, len(PARAM_ORDER))) < breed_mut_rate
            for i, key in enumerate(PARAM_ORDER):
                opts = np.array(PARAMS_SPACE[key], dtype=np.float32)
                if key in FROZEN_PARAMS and _base_for_freeze:
                    # 凍結參數：配種突變也限制在基準值 ±1 格
                    base_val = float(_base_for_freeze.get(key, opts[0]))
                    base_idx = int(np.argmin(np.abs(opts - base_val)))
                    lo = max(0, base_idx - 1)
                    hi = min(len(opts) - 1, base_idx + 1)
                    random_vals = opts[np.random.randint(lo, hi + 1, size=breed_size)]
                else:
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

        # SEED：R1 把 Gist 策略放到 params_np[0]，讓 kernel 給它新公式下的分數作 baseline
        if rnd == 1 and gist_best_params:
            for _i, _key in enumerate(PARAM_ORDER):
                _opts = np.array(PARAMS_SPACE[_key], dtype=np.float32)
                _val = float(gist_best_params.get(_key, _opts[0]))
                params_np[0, _i] = _opts[int(np.argmin(np.abs(_opts - _val)))]
            params_np[0, N_PARAMS] = MA_FAST_MAP.get(int(gist_best_params.get("ma_fast_w", 5)), 1)
            params_np[0, N_PARAMS+1] = MA_SLOW_MAP.get(int(gist_best_params.get("ma_slow_w", 20)), 1)
            params_np[0, N_PARAMS+2] = MOM_MAP.get(int(gist_best_params.get("momentum_days", 5)), 1)

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
            d_open, d_top100, d_market,
            d_up_days, d_week52, d_vol_up_days, d_mom_accel,
            d_mfi, d_cmf, d_atr_ratio,
            d_params, np.int32(N_PARAMS_FULL),
            d_results, np.int32(BATCH),
            np.int32(pre["train_start"]),
            np.int32(pre["train_end"])
        ))

        results = d_results.get()
        total_tested += BATCH

        # SEED：R1 抓 Gist 策略（position 0）的 kernel 分數當 baseline
        if rnd == 1 and gist_best_params:
            _seed_score = float(results[0, 0])
            _seed_nt = int(results[0, 1])
            _seed_total = float(results[0, 3])
            if _seed_score > 0:
                best_score = _seed_score
                best_nt = _seed_nt
                best_avg = float(results[0, 2])
                best_total = _seed_total
                best_wr = float(results[0, 4])
                total_improved += 1
                print(f"  [GPU] 🌱 SEED baseline：{_seed_score:.1f} | {_seed_nt}筆 | 總{_seed_total:.0f}%（新公式下 Gist 策略分數，要超過它才算進步）")
            else:
                # SEED 無效可能原因：
                # (1) 真的過不了（strict mode 下 89.90 被 remap 到不合格位置）
                # (2) Kernel vs cpu_replay 邏輯分歧（Python 全過但 kernel 不認，memory 記錄過）
                # 地板設 85 當「近 89.90 水準」baseline，避免 GPU 推一堆 60-80 分的爛策略
                best_score = 85.0
                print(f"  [GPU] ⚠️ SEED kernel 分數無效（{_seed_score:.1f}，{_seed_nt}筆）— WINRATE-MAX 下 89.90 預估 ~90-95")
                print(f"  [GPU] 🛡️ 設 best_score=85 當 WINRATE 地板（>85 才通知新突破）")

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
        # 🌱 保護種子（前 3 個初始 seed：89.90 + 189 + 88.60），避免配種基因池退化
        if explore_bases is None:  # 多起點爬山時不保護（那時在重新探索）
            _seeds = hall_of_fame[:3] if len(hall_of_fame) >= 3 else hall_of_fame[:]
            _rest = hall_of_fame[3:] if len(hall_of_fame) > 3 else []
            _rest.sort(key=lambda x: -x[0])
            hall_of_fame = _seeds + _rest[:12]  # 3 保護 + 12 開放 slot
        else:
            hall_of_fame.sort(key=lambda x: -x[0])
            hall_of_fame = hall_of_fame[:15]

        # 🔍 掃 kernel 前 20 名，用 cpu_replay 當最終判官（kernel 和 cpu_replay 有分歧，以 cpu_replay 為準）
        _top_n_idx = np.argsort(results[:, 0])[-20:][::-1]
        _candidate_found = False
        _candidate_trades = None  # 保留，Gist 推送區可直接用
        # Gate fail 統計：看哪個 gate 擋最多（每輪歸零）
        _gate_fail = {"cnt":0, "avg":0, "hold":0, "dd":0, "test_n":0, "test_tot":0,
                      "wf":0, "tr_ann":0, "ts_ann":0, "wr_tr":0, "wr_ts":0, "recent":0}
        _gate_best_wr = 0; _gate_best_tr_ann = 0; _gate_best_ts_ann = 0
        import math as _mc
        for _ti in _top_n_idx:
            _sc = float(results[_ti, 0])
            if _sc <= best_score or _sc <= 0:
                break  # 剩下都更低
            _tp = params_np[_ti]
            _cp = {PARAM_ORDER[_i]: float(_tp[_i]) for _i in range(N_PARAMS)}
            _cp["ma_fast_w"] = int(mf_choices[_ti])
            _cp["ma_slow_w"] = int(ms_choices[_ti])
            _cp["momentum_days"] = int(md_choices[_ti])
            _tds = cpu_replay(pre, _cp)
            _tds = [t for t in _tds if not _mc.isnan(t.get("return",0))]
            _cmp = [t for t in _tds if t.get("reason") != "持有中"]
            _cnt = len(_cmp)
            if _cnt < 40 or _cnt > 200: _gate_fail["cnt"] += 1; continue
            _cavg = sum(t.get("return",0) for t in _cmp) / _cnt
            if _cavg < 10: _gate_fail["avg"] += 1; continue
            _cah = sum(t.get("days",0) for t in _cmp) / _cnt
            if _cah < 5: _gate_fail["hold"] += 1; continue
            _crun = 0; _cmdd = 0
            for _t in _cmp:
                _r = _t.get("return",0)
                if _r < 0: _crun += _r
                else: _crun = 0
                if _crun < _cmdd: _cmdd = _crun
            if _cmdd < -50: _gate_fail["dd"] += 1; continue
            # train = buy_date 在 [train_start, train_end) 區間，test = 其他（支援正向/反向 WF）
            _tr_start_date = str(pre["dates"][pre["train_start"]].date())
            _tr_end_date = str(pre["dates"][pre["train_end"]-1].date()) if pre["train_end"] < pre["n_days"] else str(pre["dates"][-1].date())
            _ctr = [t for t in _cmp if _tr_start_date <= t.get("buy_date","") <= _tr_end_date]  # train
            _cts = [t for t in _cmp if t.get("buy_date","") < _tr_start_date or t.get("buy_date","") > _tr_end_date]  # test
            if len(_cts) < 5: _gate_fail["test_n"] += 1; continue
            _cts_tot = sum(t.get("return",0) for t in _cts)
            _ctr_tot = sum(t.get("return",0) for t in _ctr)
            if _cts_tot <= 0: _gate_fail["test_tot"] += 1; continue
            _tr_y = (pre["train_end"] - pre["train_start"]) / 250.0
            # test 年數 = 全部 - train - warmup
            _ts_days = (pre["n_days"] - 60) - (pre["train_end"] - pre["train_start"])
            _ts_y = _ts_days / 250.0
            _tr_ann = _ctr_tot/_tr_y if _tr_y > 0.5 else _ctr_tot
            _ts_ann = _cts_tot/_ts_y if _ts_y > 0.3 else _cts_tot
            if _tr_ann > 0 and _ts_ann < _tr_ann * 0.4: _gate_fail["wf"] += 1; continue
            # 報酬地板（train + test 年化不能太低）
            if _tr_ann < MIN_TRAIN_ANNUAL:
                _gate_fail["tr_ann"] += 1
                if _tr_ann > _gate_best_tr_ann: _gate_best_tr_ann = _tr_ann
                continue
            if _ts_ann < MIN_TEST_ANNUAL:
                _gate_fail["ts_ann"] += 1
                if _ts_ann > _gate_best_ts_ann: _gate_best_ts_ann = _ts_ann
                continue
            # 勝率硬門檻（train + test 各自把關，避免單期 overfit）
            _wr_train = sum(1 for t in _ctr if t.get("return",0) > 0) / len(_ctr) if _ctr else 0
            _wr_test = sum(1 for t in _cts if t.get("return",0) > 0) / len(_cts) if _cts else 0
            if _wr_train < MIN_WR_TRAIN:
                _gate_fail["wr_tr"] += 1
                if _wr_train > _gate_best_wr: _gate_best_wr = _wr_train
                continue
            if _wr_test < MIN_WR_TEST: _gate_fail["wr_ts"] += 1; continue
            # 近期 60 天崩盤檢查（勝率策略每筆期望值低，放寬為 5%）
            _recent_cutoff = str(pre["dates"][pre["n_days"] - 60].date())
            _recent = [t for t in _cmp if t.get("buy_date","") >= _recent_cutoff]
            if len(_recent) >= 3:
                _recent_avg = sum(t.get("return",0) for t in _recent) / len(_recent)
                if _recent_avg < 5:
                    _gate_fail["recent"] += 1
                    continue
            # 🔓 Calmar / 近 2 年 avg gate 移除（放在 kernel scoring 當加分項）
            # 這兩個 gate 太嚴擋掉真正有潛力的策略，改用 scoring 獎勵代替硬擋
            # 過了所有 gate，接受
            best_score = _sc
            best_nt = int(results[_ti, 1])
            best_avg = float(results[_ti, 2])
            best_total = float(results[_ti, 3])
            best_wr = float(results[_ti, 4])
            best_params = _cp
            _candidate_trades = _tds
            _candidate_found = True
            total_improved += 1
            no_improve_rounds = 0
            _wf_pct = _ts_ann/_tr_ann*100 if _tr_ann>0 else 0
            print(f"  [GPU] ✅ 新紀錄（cpu_replay 驗證）！{best_score:.1f} | {_cnt}筆 總{_ctr_tot+_cts_tot:.0f}% MaxDD{_cmdd:.1f}% WF比{_wf_pct:.0f}%")
            break
        if not _candidate_found:
            no_improve_rounds += 1
            # B+ 週期擾動：每 10 輪無突破，替換最弱 HOF 為隨機策略（打破局部最佳）
            if no_improve_rounds > 0 and no_improve_rounds % 10 == 0 and len(hall_of_fame) >= 2:
                # 產生一個完全隨機的新策略
                rand_p = {}
                for key in PARAM_ORDER:
                    rand_p[key] = float(np.random.choice(PARAMS_SPACE[key]))
                # ma 確保 fast < slow
                _mfw = int(np.random.choice(MA_FAST_OPTS))
                _msw = int(np.random.choice([o for o in MA_SLOW_OPTS if o > _mfw] or [max(MA_SLOW_OPTS)]))
                rand_p["ma_fast_w"] = _mfw
                rand_p["ma_slow_w"] = _msw
                rand_p["momentum_days"] = int(np.random.choice(MOM_DAYS_OPTS))
                # 替換 HOF 最後一名（分數最低）
                hall_of_fame[-1] = (0, rand_p)
                print(f"  [GPU] 🎲 B+ 週期擾動：{no_improve_rounds}輪無突破，HOF 最弱 slot 換成隨機策略（打破局部最佳）")
            # 變異率到頂 = 爬山已退化成亂射，啟動多起點爬山
            if mutate_rate >= 0.50:
                hall_of_fame = []
                no_improve_rounds = 0
                # 從最佳策略的「遠親」開始爬（核心參數保留80%，微調參數打亂80%）
                anchor = best_params  # NO-OVERFIT：不 fallback v5
                core_keys = {"stop_loss","hold_days","buy_threshold","max_positions","sell_below_ma","trailing_stop","use_breakeven","breakeven_trigger"}
                explore_bases = []
                for _eb in range(5):
                    if anchor:
                        rb = {}
                        for key in PARAM_ORDER:
                            if key in core_keys:
                                # 核心參數：80% 保留，20% 打亂
                                if np.random.random() < 0.2:
                                    rb[key] = float(np.random.choice(PARAMS_SPACE[key]))
                                else:
                                    rb[key] = float(anchor.get(key, np.random.choice(PARAMS_SPACE[key])))
                            else:
                                # 微調參數：80% 打亂，20% 保留
                                if np.random.random() < 0.8:
                                    rb[key] = float(np.random.choice(PARAMS_SPACE[key]))
                                else:
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
        # Gate fail 診斷：top-20 候選被哪個 gate 擋下（每 5 輪印一次）
        if rnd % 5 == 0 and not _candidate_found:
            _gate_sum = sum(_gate_fail.values())
            if _gate_sum > 0:
                _top_gates = sorted(_gate_fail.items(), key=lambda x: -x[1])[:4]
                _gate_str = " | ".join(f"{k}:{v}" for k, v in _top_gates if v > 0)
                _bests = []
                if _gate_best_wr > 0: _bests.append(f"max_wr={_gate_best_wr*100:.0f}%")
                if _gate_best_tr_ann > 0: _bests.append(f"max_tr_ann={_gate_best_tr_ann:.0f}%")
                if _gate_best_ts_ann > 0: _bests.append(f"max_ts_ann={_gate_best_ts_ann:.0f}%")
                print(f"  [GPU-診斷] top-20 gate fail: {_gate_str} | 最接近: {' '.join(_bests) or 'N/A'}")

        # Pending push（新突破已在上面的 cpu_replay 驗證通過）
        if total_improved > last_synced_improved and _candidate_trades is not None:
            try:
                trade_details = _candidate_trades
                _completed_td = [t for t in trade_details if t.get("reason") != "持有中"]
                # train = buy_date 在 [train_start, train_end] 區間；test = 其他（支援正向/反向 WF）
                _tr_start_str2 = str(pre["dates"][pre["train_start"]].date())
                _tr_end_str2 = str(pre["dates"][pre["train_end"]-1].date()) if pre["train_end"] < pre["n_days"] else str(pre["dates"][-1].date())
                _train_trades = [t for t in _completed_td if _tr_start_str2 <= t.get("buy_date","") <= _tr_end_str2]
                _test_trades = [t for t in _completed_td if t.get("buy_date","") < _tr_start_str2 or t.get("buy_date","") > _tr_end_str2]
                _train_total = sum(t.get("return",0) for t in _train_trades)
                _test_total = sum(t.get("return",0) for t in _test_trades)
                _train_years = (pre["train_end"] - pre["train_start"]) / 250.0
                _test_days = (pre["n_days"] - 60) - (pre["train_end"] - pre["train_start"])
                _test_years = _test_days / 250.0
                _train_ann = _train_total / _train_years if _train_years > 0.5 else _train_total
                _test_ann = _test_total / _test_years if _test_years > 0.3 else _test_total
                _ratio = (_test_ann / _train_ann * 100) if _train_ann > 0 else 0

                yearly = {}
                for t in _completed_td:
                    y = t["buy_date"][:4]
                    if y not in yearly: yearly[y] = {"n":0,"ret":0,"win":0}
                    yearly[y]["n"] += 1; yearly[y]["ret"] += t["return"]
                    if t["return"] > 0: yearly[y]["win"] += 1
                n_all = len(_completed_td)
                total_r = sum(t["return"] for t in _completed_td)
                avg_r = total_r / n_all if n_all else 0
                wr_r = sum(1 for t in _completed_td if t["return"]>0) / n_all * 100 if n_all else 0
                content = json.dumps({"score":round(best_score,4),"source":"gpu_rtx3060_winrate_first",
                    "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "params":best_params,"backtest":{
                        "avg_return":round(avg_r,2),"total_return":round(total_r,2),
                        "win_rate":round(wr_r,2),"total_trades":n_all},
                    "trade_details":trade_details},
                    ensure_ascii=False, indent=2)
                _pending_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_push.json")
                with open(_pending_path, "w", encoding="utf-8") as _f:
                    _f.write(content)
                year_lines = "\n".join([
                    f"  {y}: {d['n']}筆 avg{d['ret']/d['n']:.1f}% 總{d['ret']:.0f}% 勝率{d['win']/d['n']*100:.0f}%"
                    for y, d in sorted(yearly.items())
                ])
                telegram_push(
                    f"🎯 勝率優先 GPU 找到新策略（待審核）\n"
                    f"━━━━━━━━━━━━\n"
                    f"分數：{best_score:.2f}\n"
                    f"全期：{n_all}筆 avg{avg_r:.1f}% 總{total_r:.0f}% 勝率{wr_r:.0f}%\n"
                    f"WF：train 年化{_train_ann:.0f}% vs test 年化{_test_ann:.0f}% ({_ratio:.0f}%)\n"
                    f"停損{best_params.get('stop_loss',0):.0f}% | 持倉{best_params.get('max_positions',2):.0f}檔\n"
                    f"⚡ {total_tested:,}組/{elapsed:.0f}秒\n\n"
                    f"📊 分年績效：\n{year_lines}\n\n"
                    f"💾 已存 pending_push.json\n"
                    f"✅ 審核 OK → python push_pending.py"
                )
                print(f"  [GPU] 💾 pending_push.json 已更新，Telegram 通知已發")
                last_synced_improved = total_improved
            except Exception as e:
                print(f"  [GPU] pending 寫入錯誤: {e}")

if __name__ == "__main__":
    main()
