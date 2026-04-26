r"""
重建 Tab 3 backtest_results.json — 用 cpu_replay 真實結果蓋掉失真版

問題：
  Tab 3 backtest_results.json 是 daily_scan 80 天 cold start 算的失真版
  跟 cpu_replay (1500 天完整) 結果不一致

修法：
  1. 用 download_data + 1500 天 cache 跑 cpu_replay
  2. 重建 trades + stats
  3. 推 Data Gist 蓋掉 backtest_results.json

讓 Tab 3 顯示真實正確軌跡（之後 daily_scan 再從這基礎累積）
"""
import os, sys, json, types, urllib.request, base64, time, ssl
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "stock-evolution"))
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from gpu_cupy_evolve import precompute, cpu_replay, download_data


# ───────────────────────────────────────────────────────────────────
# 雙價系統：抓 TWSE/TPEx 官方 unadjusted close（給 Tab 3 顯示用）
# 89.905 cpu_replay 仍用 cache adjusted close（不動）
# ───────────────────────────────────────────────────────────────────
def fetch_official_close():
    """抓 TWSE STOCK_DAY_ALL + TPEx daily close

    回 ({ticker_no_suffix: {'close': float, 'open': float, 'high': float, 'low': float}}, date_str)
    """
    ctx = ssl._create_unverified_context()
    out = {}

    # TWSE 上市
    try:
        r = urllib.request.urlopen(
            "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json",
            context=ctx, timeout=15)
        d = json.loads(r.read())
        twse_date = d.get("date")
        for row in d.get("data", []):
            try:
                tk = row[0].strip()
                close_str = row[7].replace(",", "")
                if close_str in ("--", "", "---"): continue
                close_v = float(close_str)
                if close_v <= 0: continue
                out[tk] = {
                    "close": close_v,
                    "open": float(row[4].replace(",", "")) if row[4] not in ("--","","---") else close_v,
                    "high": float(row[5].replace(",", "")) if row[5] not in ("--","","---") else close_v,
                    "low": float(row[6].replace(",", "")) if row[6] not in ("--","","---") else close_v,
                }
            except: continue
    except Exception as e:
        print(f"  ⚠️ TWSE fetch fail: {e}")
        twse_date = None

    # TPEx 上櫃
    try:
        r = urllib.request.urlopen(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            context=ctx, timeout=15)
        d = json.loads(r.read())
        for row in d:
            try:
                tk = row.get("SecuritiesCompanyCode", "").strip()
                close_str = row.get("Close", "").strip()
                if not tk or close_str in ("--", "", "---"): continue
                close_v = float(close_str)
                if close_v <= 0: continue
                if tk in out: continue  # 不覆蓋 TWSE
                out[tk] = {
                    "close": close_v,
                    "open": float(row.get("Open", close_str)) if row.get("Open","--") not in ("--","","---") else close_v,
                    "high": float(row.get("High", close_str)) if row.get("High","--") not in ("--","","---") else close_v,
                    "low": float(row.get("Low", close_str)) if row.get("Low","--") not in ("--","","---") else close_v,
                }
            except: continue
    except Exception as e:
        print(f"  ⚠️ TPEx fetch fail: {e}")

    return out, twse_date

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
if not GH_TOKEN:
    print("❌ 請先設環境變數：$env:GH_TOKEN = 'ghp_xxx...'")
    sys.exit(1)
DATA_GIST = "e1159b02a87d3c6ee9f33fb9ef61bb80"
GPU_GIST = "c1bef892d33589baef2142ce250d18c2"


def fetch_gist(gist_id, fname):
    r = urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/gists/{gist_id}",
                                headers={"Authorization": f"token {GH_TOKEN}"}),
        timeout=30,
    )
    d = json.loads(r.read())
    return json.loads(d["files"][fname]["content"])


def write_gist(gist_id, fname, content_obj):
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        method="PATCH",
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "files": {fname: {"content": json.dumps(content_obj, ensure_ascii=False)}}
        }).encode(),
    )
    r = urllib.request.urlopen(req, timeout=120)
    return r.status


# 1. 載 cache + strategy
print("[1/5] 載 cache + 89.905 策略 ...")
data = download_data()
strategy = fetch_gist(GPU_GIST, "best_strategy.json")
p = strategy.get("params", strategy)
print(f"  Cache: {len(data)} tickers / Strategy: {strategy.get('score', 0):.3f}")

# 🚨 Sanity check：策略合理性（防 GPU Gist 被改錯策略）
_score_check = float(strategy.get('score', 0))
if _score_check < 60 or _score_check > 200:
    raise RuntimeError(f"❌ 策略分數異常 {_score_check}（合理範圍 60-200）→ GPU Gist 可能被改錯，aborting")
# 必要 params 存在
for _k in ["stop_loss", "hold_days", "max_positions", "buy_threshold"]:
    if _k not in p:
        raise RuntimeError(f"❌ 策略缺 {_k}（GPU Gist 可能損壞）")
# 賣出參數合理範圍
if not (-50 <= p.get("stop_loss", -20) <= -5):
    raise RuntimeError(f"❌ stop_loss={p.get('stop_loss')} 異常（合理 -50 ~ -5）")
if not (5 <= int(p.get("hold_days", 30)) <= 90):
    raise RuntimeError(f"❌ hold_days={p.get('hold_days')} 異常（合理 5-90）")
print(f"  ✅ 策略 sanity 通過")

# 2. precompute + cpu_replay 跑全期
# ⭐ 固定起點 = 2020-01-02（cache 真實起點）
# 但只納入「真的從 2020-01-02 就有資料」的 ticker，避免最近上市的拖低 min_len
# precompute() 用 min(len(df)) 切共同切片 → 只要有 1 檔 102 天，全部變 102 天 → 災難
import pandas as _pd_anchor
FIXED_START = _pd_anchor.Timestamp("2020-01-02").normalize()
print(f"\n[2/5] 固定起點 {FIXED_START.date()}（cache 真實起點，避免每天滑動）...")
data_t = {}
n_short = 0
for k, v in data.items():
    idx = v.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx_naive = idx.tz_localize(None)
    else:
        idx_naive = idx
    # 必須包含 FIXED_START（或更早，那就切起來）
    first_date = idx_naive.normalize()[0]
    if first_date > FIXED_START:
        # 這 ticker 起點晚於 2020-01-02 → 排除（避免拖低 min_len）
        n_short += 1
        continue
    mask = idx_naive.normalize() >= FIXED_START
    df = v[mask]
    if len(df) >= 100:
        data_t[k] = df
print(f"  {len(data_t)} stocks 通過 / 排除 {n_short} 檔起點晚於 {FIXED_START.date()} 的（避免拖低 min_len）")
pre = precompute(data_t)
trades = cpu_replay(pre, p)
trades.sort(key=lambda t: t.get("buy_date", ""))
print(f"  共 {len(trades)} trades")

# 3. 轉成 Tab 3 期望的格式
# Tab 3 用：return_pct (不是 return), hold_days (不是 days)
print(f"\n[3/5] 轉換為 Tab 3 格式 ...")

# 雙價系統：抓官方 unadjusted close 給「持有中」trade 顯示用
print(f"  抓 TWSE/TPEx 官方 unadjusted close ...")
official_quotes, official_date = fetch_official_close()
print(f"  官方資料: {len(official_quotes)} 檔, date={official_date}")

tab3_trades = []
n_display = 0
for t in trades:
    ticker = t.get("ticker", "")
    item = {
        "ticker": ticker,
        "name": t.get("name", ""),
        "buy_date": t.get("buy_date", ""),
        "sell_date": t.get("sell_date", ""),
        "buy_price": float(t.get("buy_price", 0)),
        "sell_price": float(t.get("sell_price", 0)),
        "return_pct": float(t.get("return", 0)),
        "hold_days": int(t.get("days", 0)),
        "reason": t.get("reason", ""),
    }
    if "peak_price" in t:
        item["peak_price"] = float(t["peak_price"])

    # 雙價：對「持有中」trade 用官方 unadjusted close 顯示
    # （歷史完成 trade 保留 adjusted sell_price，太多筆無法逐日對 TWSE）
    if t.get("reason") == "持有中" and ticker:
        tk_no_suffix = ticker.split(".", 1)[0] if "." in ticker else ticker
        official = official_quotes.get(tk_no_suffix)
        if official and official.get("close", 0) > 0:
            buy_p = item["buy_price"]
            unadj_close = official["close"]
            # display_return_pct = unadjusted 算的（看盤直觀）
            disp_ret = round((unadj_close / buy_p - 1) * 100 - 0.585, 2) if buy_p > 0 else 0
            item["display_price"] = unadj_close
            item["display_return_pct"] = disp_ret
            n_display += 1

    tab3_trades.append(item)

print(f"  {n_display} 檔「持有中」加入官方 unadjusted display_price")

# 4. 算 stats（mirror daily_scan line 469-477）
completed = [t for t in tab3_trades if t.get("reason") != "持有中"]
holding = [t for t in tab3_trades if t.get("reason") == "持有中"]
rets = [t["return_pct"] for t in completed]
wins = [r for r in rets if r > 0]
losses = [r for r in rets if r <= 0]

# 起始/結束日
all_buy_dates = [t["buy_date"] for t in tab3_trades if t.get("buy_date")]
all_dates_in_pre = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in pre["dates"]]
start_date = all_dates_in_pre[60] if len(all_dates_in_pre) > 60 else all_dates_in_pre[0]  # warmup 後
end_date = all_dates_in_pre[-1]

from datetime import datetime, timezone, timedelta
TW_TZ = timezone(timedelta(hours=8))
stats = {
    "total_trades": len(completed),
    "total_return_pct": round(sum(rets), 1) if rets else 0,
    "win_rate": round(len(wins) / len(rets) * 100, 1) if rets else 0,
    "avg_return": round(sum(rets) / len(rets), 1) if rets else 0,
    "avg_win": round(sum(wins) / len(wins), 1) if wins else 0,
    "avg_loss": round(sum(losses) / len(losses), 1) if losses else 0,
    "max_win": round(max(rets), 1) if rets else 0,
    "max_loss": round(min(rets), 1) if rets else 0,
    "avg_hold_days": round(sum(t.get("hold_days", 0) for t in completed) / len(completed), 1) if completed else 0,
    "start_date": start_date,
    "end_date": end_date,
    "total_days": int(pre["n_days"]),
    "strategy_version": "auto",
    "strategy_score": float(strategy.get("score", 89.905)),
    # ⭐ 新增：pipeline 跑完時間戳（Web 用來判斷新鮮度）
    "pipeline_updated": datetime.now(TW_TZ).isoformat(),
    "pipeline_source": "rebuild_tab3 (cpu_replay 1500 day)",
}

print(f"  完成 {stats['total_trades']} 筆 / 持有 {len(holding)} 檔")
print(f"  總報酬 {stats['total_return_pct']}% / 勝率 {stats['win_rate']}%")
print(f"  期間 {stats['start_date']} ~ {stats['end_date']}")

print(f"\n  持有中：")
for h in holding:
    print(f"    {h['name']} ({h['ticker']}) buy {h['buy_date']} @ {h['buy_price']}")

# 5. 推 Data Gist 蓋掉 backtest_results.json
print(f"\n[4/5] 比對舊版 Tab 3 數據 ...")
try:
    old_bt = fetch_gist(DATA_GIST, "backtest_results.json")
    old_stats = old_bt.get("stats", {})
    print(f"  舊版：{old_stats.get('total_trades')} 筆 / {old_stats.get('total_return_pct')}% / wr {old_stats.get('win_rate')}%")
    print(f"  新版：{stats['total_trades']} 筆 / {stats['total_return_pct']}% / wr {stats['win_rate']}%")
    old_hold = [t for t in old_bt.get("trades", []) if t.get("reason") == "持有中"]
    print(f"  舊版持有：{[(h.get('name'), h.get('ticker')) for h in old_hold]}")
    print(f"  新版持有：{[(h['name'], h['ticker']) for h in holding]}")
except Exception as e:
    print(f"  讀舊版失敗: {e}")

new_bt = {"stats": stats, "trades": tab3_trades}

print(f"\n[5/5] 推 Data Gist 蓋掉 backtest_results.json ...")
try:
    status = write_gist(DATA_GIST, "backtest_results.json", new_bt)
    print(f"  ✅ Status {status} - Tab 3 已更新為 cpu_replay 真實結果")
    print(f"  Web App 重新整理後會看到新數據")
except Exception as e:
    print(f"  ❌ Push fail: {e}")
    sys.exit(1)
sys.exit(0)


# ─────── 以下廢棄（保留參考）— Step 6: 同時寫 scan_results 的 pending ───────
print(f"\n[6/6] 用 cpu_replay 結果算明天 pending 寫進 scan_results.json ...")


def should_sell_full(bp, cur, peak, days_held, params):
    """89.905 啟用的 5 條賣出規則（mirror sell_rules.py 但只含不靠 indicator 的部分）"""
    if bp <= 0 or cur <= 0 or days_held < 1:
        return None
    ret = (cur / bp - 1) * 100
    peak_gain = (peak / bp - 1) * 100 if bp > 0 else 0
    # 1. 停損 / 保本
    eff_stop = params.get("stop_loss", -20)
    if params.get("use_breakeven", 0) and peak_gain >= params.get("breakeven_trigger", 20):
        eff_stop = 0
    if ret <= eff_stop:
        return f"保本出場 {ret:+.1f}%（曾漲 +{peak_gain:.1f}%）" if eff_stop == 0 else f"停損 {ret:+.1f}%"
    # 2. 停利
    if params.get("use_take_profit", 1) and ret >= params.get("take_profit", 80):
        return f"停利 +{ret:.1f}%"
    # 3. 移動停利
    trailing = params.get("trailing_stop", 0)
    if trailing > 0 and peak > bp * 1.01:
        dd = (cur / peak - 1) * 100
        if dd <= -trailing:
            return f"移動停利 {dd:.1f}%"
    # 4. 鎖利
    if params.get("use_profit_lock", 0):
        if peak_gain >= params.get("lock_trigger", 30) and ret < params.get("lock_floor", 10):
            return f"鎖利出場（曾 +{peak_gain:.1f}% 跌回 {ret:+.1f}%）"
    # 5. 到期
    if days_held >= int(params.get("hold_days", 30)):
        return f"到期 {days_held} 天 {ret:+.1f}%"
    return None


# 取得最新「持有中」+ 算每檔賣出條件
from datetime import datetime as _dt, timezone as _tz, timedelta as _td
TW = _tz(_td(hours=8))

new_pending_sells = []
new_pending_buy = None
for h in holding:
    bp = h["buy_price"]
    cur = h["sell_price"]  # rebuild_tab3 寫的 sell_price = 末日 close
    peak = h.get("peak_price", bp)
    days_held = h.get("hold_days", 0)
    reason = should_sell_full(bp, cur, peak, days_held, p)
    if reason:
        new_pending_sells.append({
            "ticker": h["ticker"],
            "name": h["name"],
            "reason": reason,
        })
        print(f"  📤 PENDING SELL: {h['name']} ({h['ticker']}) — {reason}")

# 算 pending_buy：如果有空位（賣完或本來就空），找 universe top-1 score
max_pos = int(p.get("max_positions", 2))
holdings_after_sell = len(holding) - len(new_pending_sells)
if holdings_after_sell < max_pos:
    # 用 cpu_replay 末日的 universe + score 找 top-1（mirror cpu_replay buy 邏輯）
    last_day = pre["n_days"] - 1
    top100_mask = pre.get("top100_mask")
    if top100_mask is not None:
        in_uni_si = np.where(top100_mask[:, last_day] >= 0.5)[0]
        # 已持有的不能再買
        held_tks = {h["ticker"] for h in holding} - {ps["ticker"] for ps in new_pending_sells}

        # 算每檔 score（mirror cpu_replay _score_stock + line 1601-1609 inline 加分）
        rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
        close = pre["close"]; macd_line = pre["macd_line"]; macd_hist = pre["macd_hist"]
        k_val = pre["k_val"]; d_val = pre["d_val"]
        williams_r = pre["williams_r"]; near_high = pre["near_high"]
        is_green = pre.get("is_green"); gap = pre.get("gap"); ma60 = pre.get("ma60")
        vol_prev = pre.get("vol_prev")
        squeeze_fire = pre.get("squeeze_fire"); new_high_60 = pre.get("new_high_60")
        adx_arr = pre.get("adx"); bias_arr = pre.get("bias")
        obv_rising_arr = pre.get("obv_rising"); atr_pct_arr = pre.get("atr_pct")
        sector_hot = pre.get("sector_hot")
        up_days_arr = pre.get("up_days"); week52_arr = pre.get("week52_pos")
        vol_up_days_arr = pre.get("vol_up_days"); mom_accel_arr = pre.get("mom_accel")
        ma_fw = int(p.get("ma_fast_w", 5))
        mom_days = int(p.get("momentum_days", 5))
        maf = pre["ma_d"].get(ma_fw, pre["ma_d"][5])
        mom = pre["mom_d"].get(mom_days, pre["mom_d"][5])

        def score(si, d):
            sc = 0.0
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
            if int(p.get("w_squeeze",0))>0 and squeeze_fire is not None and squeeze_fire[si,d]>0.5: sc+=int(p["w_squeeze"])
            if int(p.get("w_new_high",0))>0 and new_high_60 is not None and new_high_60[si,d]>0.5: sc+=int(p["w_new_high"])
            if int(p.get("w_adx",0))>0 and adx_arr is not None and adx_arr[si,d]>=p.get("adx_th",25): sc+=int(p["w_adx"])
            if int(p.get("w_bias",0))>0 and bias_arr is not None and bias_arr[si,d]>=0 and bias_arr[si,d]<=p.get("bias_max",15): sc+=int(p["w_bias"])
            if int(p.get("w_obv",0))>0 and obv_rising_arr is not None and obv_rising_arr[si,d]>0.5: sc+=int(p["w_obv"])
            if int(p.get("w_atr",0))>0 and atr_pct_arr is not None and atr_pct_arr[si,d]>=p.get("atr_min",2): sc+=int(p["w_atr"])
            if int(p.get("w_up_days",0))>0 and up_days_arr is not None and up_days_arr[si,d]>=p.get("up_days_min",3): sc+=int(p["w_up_days"])
            if int(p.get("w_week52",0))>0 and week52_arr is not None and week52_arr[si,d]>=p.get("week52_min",0.7): sc+=int(p["w_week52"])
            if int(p.get("w_vol_up_days",0))>0 and vol_up_days_arr is not None and vol_up_days_arr[si,d]>=p.get("vol_up_days_min",3): sc+=int(p["w_vol_up_days"])
            if int(p.get("w_mom_accel",0))>0 and mom_accel_arr is not None and mom_accel_arr[si,d]>=p.get("mom_accel_min",2): sc+=int(p["w_mom_accel"])
            # inline (line 1601-1609)
            cg = int(p.get("consecutive_green", 0))
            if cg >= 1 and is_green is not None:
                ok = True
                for g in range(cg):
                    if d - g < 0 or is_green[si, d-g] != 1: ok = False; break
                if ok: sc += 1
            if p.get("gap_up", 0) and gap is not None and gap[si, d] >= 1.0: sc += 1
            if p.get("above_ma60", 0) and ma60 is not None and close[si, d] >= ma60[si, d]: sc += 1
            if p.get("vol_gt_yesterday", 0) and d >= 1 and vol_prev is not None and vol_ratio[si, d] > vol_prev[si, d]: sc += 1
            return sc

        buy_th = int(p.get("buy_threshold", 8))
        candidates = sorted(
            [(score(si, last_day), si) for si in in_uni_si if pre["tickers"][si] not in held_tks],
            reverse=True,
        )
        for sc, si in candidates:
            if sc < buy_th:
                break
            new_pending_buy = {
                "ticker": pre["tickers"][si],
                "name": "",  # 名字晚點補（cpu_replay 不存名稱對應）
                "score": float(sc),
                "close": float(pre["close"][si, last_day]),
            }
            print(f"  🎯 PENDING BUY: {pre['tickers'][si]} score={sc} close={float(pre['close'][si, last_day]):.2f}")
            break

# 讀現有 scan_results 維持其他欄位（buy_signals top10 等）
try:
    cur_scan = fetch_gist(DATA_GIST, "scan_results.json")
    cur_scan["pending_sells"] = new_pending_sells
    cur_scan["pending_buy"] = new_pending_buy
    cur_scan["pending_source"] = f"rebuild_tab3 cpu_replay 1500d @ {_dt.now(TW).isoformat()}"
    write_gist(DATA_GIST, "scan_results.json", cur_scan)
    print(f"  ✅ scan_results.json pending 已更新（cpu_replay 算的真公式）")
except Exception as e:
    print(f"  ⚠️ 寫 pending 失敗: {e}")
