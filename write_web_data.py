r"""
從 Windows 全期完整 cache 寫所有 Web 需要的 Gist：
  1. history_cache.json (80 天 K → Gist) — 給 sell_rules 算 MA60 等
  2. scan_results.json (pending + buy_signals top 20) — 給 4 個 Tab 顯示
  3. backtest_results.json 由 rebuild_tab3.py 寫
  4. indicator_state.json 由 init_state_gist.py 寫

用 cpu_replay 真公式算（基於全期 cache，從 2020-01-02 起，每天增加一天）
→ 100% 對齊 Tab 3 backtest
取代雲端 daily_scan 的工作（daily_scan 仍當保底）
"""
import os, sys, json, types, urllib.request, base64, time
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "stock-evolution"))
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from gpu_cupy_evolve import precompute, cpu_replay, download_data, get_name as _gpu_get_name

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
if not GH_TOKEN:
    print("❌ 請先設 $env:GH_TOKEN = 'ghp_xxx...'")
    sys.exit(1)

DATA_GIST = "e1159b02a87d3c6ee9f33fb9ef61bb80"
HISTORY_GIST = "572d4ca53b0bfbd37dd5485becdcce49"
GPU_GIST = "c1bef892d33589baef2142ce250d18c2"
TW_TZ = timezone(timedelta(hours=8))


def fetch_gist(gist_id, fname):
    r = urllib.request.urlopen(
        urllib.request.Request(f"https://api.github.com/gists/{gist_id}",
                                headers={"Authorization": f"token {GH_TOKEN}"}),
        timeout=30,
    )
    d = json.loads(r.read())
    return json.loads(d["files"][fname]["content"])


def write_gist(gist_id, fname, obj):
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        method="PATCH",
        headers={"Authorization": f"token {GH_TOKEN}", "Content-Type": "application/json"},
        data=json.dumps({"files": {fname: {"content": json.dumps(obj, ensure_ascii=False)}}}).encode(),
    )
    return urllib.request.urlopen(req, timeout=120).status


# ─────── 1. 載 cache + strategy ───────
print("[1/5] 載 cache + 89.905 策略 ...")
data = download_data()
strategy = fetch_gist(GPU_GIST, "best_strategy.json")
p = strategy.get("params", strategy)
print(f"  Cache: {len(data)} tickers / Strategy: {strategy.get('score', 0):.3f}")

# ─────── 2. precompute 全期 ───────
FIXED_START = pd.Timestamp("2020-01-02").normalize()
print(f"\n[2/5] precompute 全期（起點固定 {FIXED_START.date()}）...")
data_t = {}
for k, v in data.items():
    idx = v.index
    idx_naive = idx.tz_localize(None) if hasattr(idx, "tz") and idx.tz is not None else idx
    first_date = idx_naive.normalize()[0]
    if first_date > FIXED_START:
        continue
    mask = idx_naive.normalize() >= FIXED_START
    df = v[mask]
    if len(df) >= 100:
        data_t[k] = df
print(f"  {len(data_t)} stocks")
pre = precompute(data_t)
tickers = pre["tickers"]
dates = pre["dates"]
last_day = pre["n_days"] - 1
last_date_str = pd.Timestamp(dates[last_day]).strftime("%Y-%m-%d")
print(f"  末日: {last_date_str}")

# ─────── 3. cpu_replay → 拿 trades ───────
print(f"\n[3/5] cpu_replay 算 trades ...")
trades = cpu_replay(pre, p)
trades.sort(key=lambda t: t.get("buy_date", ""))
holdings = [t for t in trades if t.get("reason") == "持有中"]
print(f"  {len(trades)} trades / {len(holdings)} 持有中")

# ─────── 4a. 寫 history_cache.json (80 天 K) ───────
print(f"\n[4/5] 切 80 天 K 線寫 history_cache.json ...")
WINDOW = 80
history_stocks = {}
for tk, df in data_t.items():
    tail = df.tail(WINDOW)
    history_stocks[tk] = {
        "c": [round(float(x), 4) for x in tail["Close"].values],
        "h": [round(float(x), 4) for x in tail["High"].values],
        "l": [round(float(x), 4) for x in tail["Low"].values],
        "v": [int(x) for x in tail["Volume"].values],
        "o": [round(float(x), 4) for x in tail["Open"].values] if "Open" in tail.columns else [],
    }
    # h250 / l250 給 week52_pos 用（250 天 high/low）
    tail250 = df.tail(250)
    history_stocks[tk]["h250"] = [round(float(x), 4) for x in tail250["High"].values]
    history_stocks[tk]["l250"] = [round(float(x), 4) for x in tail250["Low"].values]

# 全市場交易日序列
top_dates = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for tk, df in data_t.items()
                    for d in df.tail(WINDOW).index})

history_cache = {
    "stocks": history_stocks,
    "dates": top_dates[-WINDOW:],
    "updated": last_date_str,
    "source": f"windows_pipeline write_web_data @ {datetime.now(TW_TZ).isoformat()}",
}
print(f"  {len(history_stocks)} stocks × {WINDOW} 天 + 250 天 h/l")
status = write_gist(HISTORY_GIST, "history_cache.json", history_cache)
print(f"  ✅ history_cache.json status {status}")

# ─────── 4b. 算 pending + buy_signals + 寫 scan_results.json ───────
print(f"\n[5/5] 算 pending + buy_signals 寫 scan_results.json ...")


def should_sell_full(bp, cur, peak, days_held, params):
    """89.905 5 條啟用規則（mirror sell_rules.py）"""
    if bp <= 0 or cur <= 0 or days_held < 1:
        return None
    ret = (cur / bp - 1) * 100
    peak_gain = (peak / bp - 1) * 100 if bp > 0 else 0
    eff_stop = params.get("stop_loss", -20)
    if params.get("use_breakeven", 0) and peak_gain >= params.get("breakeven_trigger", 20):
        eff_stop = 0
    if ret <= eff_stop:
        return f"保本出場 {ret:+.1f}%（曾漲 +{peak_gain:.1f}%）" if eff_stop == 0 else f"停損 {ret:+.1f}%"
    if params.get("use_take_profit", 1) and ret >= params.get("take_profit", 80):
        return f"停利 +{ret:.1f}%"
    trailing = params.get("trailing_stop", 0)
    if trailing > 0 and peak > bp * 1.01:
        dd = (cur / peak - 1) * 100
        if dd <= -trailing:
            return f"移動停利 {dd:.1f}%"
    if params.get("use_profit_lock", 0):
        if peak_gain >= params.get("lock_trigger", 30) and ret < params.get("lock_floor", 10):
            return f"鎖利出場（曾 +{peak_gain:.1f}% 跌回 {ret:+.1f}%）"
    if days_held >= int(params.get("hold_days", 30)):
        return f"到期 {days_held} 天 {ret:+.1f}%"
    return None


# 對「持有中」算明天 pending_sells
pending_sells = []
for h in holdings:
    bp = h["buy_price"]
    cur = h["sell_price"]
    peak = h.get("peak_price", bp)
    days = h.get("days", 0)
    reason = should_sell_full(bp, cur, peak, days, p)
    if reason:
        pending_sells.append({
            "ticker": h["ticker"],
            "name": h["name"],
            "reason": reason,
        })
        print(f"  📤 PENDING SELL: {h['name']} ({h['ticker']}) — {reason}")

# 算 cpu_replay score 找 buy_signals top 20
rsi = pre["rsi"]; bb_pos = pre["bb_pos"]; vol_ratio = pre["vol_ratio"]
close = pre["close"]; macd_line = pre["macd_line"]; macd_hist = pre["macd_hist"]
k_val = pre["k_val"]; d_val = pre["d_val"]
williams_r = pre["williams_r"]; near_high = pre["near_high"]
is_green = pre.get("is_green"); gap = pre.get("gap"); ma60 = pre.get("ma60")
vol_prev = pre.get("vol_prev")
squeeze_fire = pre.get("squeeze_fire"); new_high_60 = pre.get("new_high_60")
adx_arr = pre.get("adx"); bias_arr = pre.get("bias")
obv_rising_arr = pre.get("obv_rising"); atr_pct_arr = pre.get("atr_pct")
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


# 末日 universe top scores
top100_mask = pre["top100_mask"]
in_uni = np.where(top100_mask[:, last_day] >= 0.5)[0]
held_tks = {h["ticker"] for h in holdings} - {ps["ticker"] for ps in pending_sells}
buy_th = int(p.get("buy_threshold", 8))

candidates = []
for si in in_uni:
    sc = score(si, last_day)
    if sc >= buy_th:
        tk = tickers[si]
        if tk not in held_tks:  # 排除已持有的
            candidates.append({
                "rank": 0,
                "ticker": tk,
                "name": "",  # 名稱要從別處查（之後用 yfinance Ticker.info 補）
                "score": float(sc),
                "close": float(close[si, last_day]),
                "vol_ratio": round(float(vol_ratio[si, last_day]), 1),
            })

candidates.sort(key=lambda x: (-x["score"], -x["vol_ratio"], x["ticker"]))
for i, c in enumerate(candidates[:20]):
    c["rank"] = i + 1

buy_signals = candidates[:20]
pending_buy = candidates[0] if candidates and len(holdings) - len(pending_sells) < int(p.get("max_positions", 2)) else None
if pending_buy:
    print(f"  🎯 PENDING BUY (top 1): {pending_buy['ticker']} score={pending_buy['score']}")
print(f"  buy_signals top 20 寫入 ({len(buy_signals)} 達標)")

# 補 ticker name（用 gpu_cupy_evolve.get_name → CN_NAMES dict）
def get_name(tk):
    return _gpu_get_name(tk)

for c in buy_signals:
    c["name"] = get_name(c["ticker"])
if pending_buy:
    pending_buy["name"] = get_name(pending_buy["ticker"])
for ps in pending_sells:
    if not ps.get("name"):
        ps["name"] = get_name(ps["ticker"])

# 市場概況
twse_n = sum(1 for tk in tickers if tk.endswith(".TW") and not tk.endswith(".TWO"))
otc_n = sum(1 for tk in tickers if tk.endswith(".TWO"))

scan_results = {
    "date": last_date_str,
    "timestamp": datetime.now(TW_TZ).isoformat(),
    "strategy_version": "auto",
    "strategy_score": float(strategy.get("score", 89.905)),
    "buy_signals": buy_signals,
    "sell_signals": [],
    "holdings_status": [{"ticker": h["ticker"], "name": h["name"],
                         "current_price": h["sell_price"]} for h in holdings],
    "market_summary": {"twse_count": twse_n, "otc_count": otc_n, "scan_count": 100},
    "pending_sells": pending_sells,
    "pending_buy": pending_buy,
    "source": f"windows_pipeline write_web_data @ {datetime.now(TW_TZ).isoformat()}",
}

status = write_gist(DATA_GIST, "scan_results.json", scan_results)
print(f"  ✅ scan_results.json status {status}")

print("\n" + "=" * 60)
print(f"🎉 Web 資料全部更新完成（cpu_replay 真公式）")
print(f"  history_cache: {len(history_stocks)} stocks × 80 天")
print(f"  scan_results: {len(buy_signals)} buy_signals top 20")
print(f"  pending_sells: {len(pending_sells)} / pending_buy: {'有' if pending_buy else '無'}")
print("=" * 60)
