r"""
從 Windows 1500 天完整 cache 算出每檔的 indicator state，推到 state Gist
讓 daily_scan 用 compute_indicators_with_state 對齊 cpu_replay 結果

state Gist ID: 18a7270d897c8821b291cfd61796bd80
file: indicator_state.json

State 結構（每檔）：
  rsi_ag, rsi_al      — Wilder RSI 累積值
  ema12, ema26        — MACD EMA
  macd_sig, mh, mh_prev — MACD signal/histogram
  atr14               — ATR Wilder
  adx_a14, adx_sp, adx_sm, adx_val — ADX 累積
  kd_k, kd_d, kd_k_prev, kd_d_prev — KD
  macd_line, macd_hist, macd_hist_prev — for compute_indicators_with_state line 451
"""
import os, sys, json, types, urllib.request, base64
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "stock-evolution"))
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
from gpu_cupy_evolve import download_data

# Token: 從環境變數讀 (避免硬編碼觸發 secret scanning)
GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GIST_TOKEN")
if not GH_TOKEN:
    print("❌ 請先設環境變數：$env:GH_TOKEN = 'ghp_xxx...'")
    print("   或在 PowerShell：$env:GH_TOKEN = (你的 gist scope token)")
    sys.exit(1)
STATE_GIST = "18a7270d897c8821b291cfd61796bd80"


def init_state(c, h, lo, vol):
    """從完整 K 線算出 daily_scan 用的 state（mirror scanner.compute_indicators 累積部分）"""
    n = len(c)
    if n < 30:
        return None
    state = {}

    # RSI (Wilder)
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    ag = float(np.mean(gain[:14]))
    al = float(np.mean(loss[:14]))
    for i in range(15, n):
        ag = (ag * 13 + gain[i - 1]) / 14
        al = (al * 13 + loss[i - 1]) / 14
    state["rsi_ag"] = round(ag, 6)
    state["rsi_al"] = round(al, 6)

    # MACD EMA12 / EMA26 / signal
    ema12 = float(c[0]); ema26 = float(c[0])
    for i in range(1, n):
        ema12 = ema12 * (1 - 2/13) + c[i] * 2/13
        ema26 = ema26 * (1 - 2/27) + c[i] * 2/27
    state["ema12"] = round(ema12, 4)
    state["ema26"] = round(ema26, 4)

    ml_arr = []
    e12 = float(c[0]); e26 = float(c[0])
    for i in range(1, n):
        e12 = e12 * (1 - 2/13) + c[i] * 2/13
        e26 = e26 * (1 - 2/27) + c[i] * 2/27
        ml_arr.append(e12 - e26)
    sig = ml_arr[0]
    mh_prev = ml_arr[0] - sig
    mh_now = mh_prev
    for ml_v in ml_arr[1:]:
        mh_prev = mh_now
        sig = sig * (1 - 2/10) + ml_v * 2/10
        mh_now = ml_v - sig
    state["macd_sig"] = round(sig, 4)
    state["mh"] = round(mh_now, 4)
    state["mh_prev"] = round(mh_prev, 4)
    state["macd_line"] = round(ml_arr[-1], 4)
    state["macd_hist"] = round(mh_now, 4)
    state["macd_hist_prev"] = round(mh_prev, 4)

    # ATR + ADX (Wilder)
    tr = np.zeros(n)
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i-1]), abs(lo[i] - c[i-1]))
        up = h[i] - h[i-1]; dn = lo[i-1] - lo[i]
        pdm[i] = up if up > dn and up > 0 else 0
        mdm[i] = dn if dn > up and dn > 0 else 0
    a14 = float(np.mean(tr[1:15])); sp = float(np.mean(pdm[1:15])); sm = float(np.mean(mdm[1:15]))
    dx_arr = []
    for i in range(15, n):
        a14 = (a14 * 13 + tr[i]) / 14
        sp = (sp * 13 + pdm[i]) / 14
        sm = (sm * 13 + mdm[i]) / 14
        pdi = sp / a14 * 100 if a14 > 0 else 0
        mdi = sm / a14 * 100 if a14 > 0 else 0
        dx = abs(pdi - mdi) / (pdi + mdi) * 100 if pdi + mdi > 0 else 0
        dx_arr.append(dx)
    if len(dx_arr) >= 14:
        adx_v = float(np.mean(dx_arr[:14]))
        for dx in dx_arr[14:]:
            adx_v = (adx_v * 13 + dx) / 14
    else:
        adx_v = 0.0
    state["atr14"] = round(a14, 4)
    state["adx_a14"] = round(a14, 4)
    state["adx_sp"] = round(sp, 4)
    state["adx_sm"] = round(sm, 4)
    state["adx_val"] = round(adx_v, 4)

    # KD
    kv = 50.0; dv = 50.0; kv_prev = 50.0; dv_prev = 50.0
    for i in range(1, n):
        lo9 = float(np.min(lo[max(0, i-9):i+1]))
        hi9 = float(np.max(h[max(0, i-9):i+1]))
        rsv = (c[i] - lo9) / (hi9 - lo9) * 100 if hi9 > lo9 else 50
        kv_prev = kv; dv_prev = dv
        kv = kv * 2/3 + rsv / 3
        dv = dv * 2/3 + kv / 3
    state["kd_k"] = round(kv, 4)
    state["kd_d"] = round(dv, 4)
    state["kd_k_prev"] = round(kv_prev, 4)
    state["kd_d_prev"] = round(dv_prev, 4)

    return state


# 載完整 cache
print("[1/4] 載完整 cache ...")
data = download_data()
print(f"  {len(data)} tickers")

# 算每檔 state
print("[2/4] 算 indicator state ...")
states = {}
n_done = 0
n_fail = 0
import pandas as pd
for tk, df in data.items():
    try:
        if len(df) < 30:
            n_fail += 1; continue
        c = df["Close"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        lo = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        s = init_state(c, h, lo, vol)
        if s is not None:
            states[tk] = s
            n_done += 1
        else:
            n_fail += 1
    except Exception as e:
        n_fail += 1
    if (n_done + n_fail) % 200 == 0:
        print(f"  進度 {n_done + n_fail}/{len(data)} 成功 {n_done}")
print(f"  完成 {n_done} / 失敗 {n_fail}")

# 包成 state Gist 格式
last_date = max(df.index[-1] for df in data.values())
last_date_str = pd.Timestamp(last_date).strftime("%Y-%m-%d")
state_obj = {
    "states": states,
    "updated": last_date_str,
}

# 確認大小
content_str = json.dumps(state_obj, ensure_ascii=False)
size_mb = len(content_str.encode()) / 1024 / 1024
print(f"\n[3/4] 序列化大小: {size_mb:.2f} MB")
if size_mb > 95:
    print(f"  ⚠️ 超過 Gist 100 MB 限制 — 不推")
    sys.exit(1)

# 推 state Gist
print(f"[4/4] 推 state Gist {STATE_GIST} ...")
import urllib.request
req = urllib.request.Request(
    f"https://api.github.com/gists/{STATE_GIST}",
    method="PATCH",
    headers={
        "Authorization": f"token {GH_TOKEN}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "files": {
            "indicator_state.json": {"content": content_str}
        }
    }).encode(),
)
try:
    r = urllib.request.urlopen(req, timeout=120)
    print(f"  ✅ Status {r.status} - state Gist updated with {len(states)} tickers")
except Exception as e:
    print(f"  ❌ Push fail: {e}")
    sys.exit(1)

print("\n下次 daily_scan 跑時，所有 ticker 都會走 compute_indicators_with_state")
print("→ score 跟 cpu_replay 對齊 → Tab 3 訊號跟 realistic 結果一致")
