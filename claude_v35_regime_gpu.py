"""
V35 Regime Commander — GPU 進化搜尋（fork of gpu_cupy_evolve.py）
版本：0.3（2026-04-25 第三次稽查修致命 BUG V35-E：cpu_replay 必須 mirror regime gate）

設計理念
========
攻擊 89.90 的唯一已知弱點：2022 熊市勝率 57%（vs 整體 69.4%）
Sanity test 結果確認：BULL 72.6% / BEAR 54.5% / CHOP 57.1% 差距明顯 → 有攻擊面
做法：regime-aware 動態 buy_threshold，不同市場 regime 用不同進場嚴格度

Regime 偵測（規則式，不用 ML）
  - 用 base precompute 已算好的 market_close
  - default ma20=20, ma60=60 算 regime
  - regime 0 = BULL：MA20 > MA60 AND market_close > MA20
  - regime 1 = BEAR：MA20 < MA60 AND market_close < MA20
  - regime 2 = CHOP：其他（盤整 / 轉折 / warmup）

新增 4 個「有效」params（其他 5 slot 鎖死 [0]，避免浪費搜索）
  bull_buy_th_delta   [0, -3, -2, -1, 1]  Bull 時 buy_threshold 調整（0 放首 = SEED default）
  bear_buy_th_delta   [0, 1, 2, 3]        Bear 時（正值=更嚴格，寧空倉）
  chop_buy_th_delta   [0, -1, 1, 2]       Chop 時（0 放首）
  regime_gate_mode    [3, 0]              3=DISABLED(default), 0=threshold gate 啟用

鎖死 [0] 的 slot（佔位 param order，但 GPU 不會亂選）
  bull_max_pos_ovr, bear_max_pos_ovr, chop_max_pos_ovr  （V35.1 才實作）
  regime_ma20_len, regime_ma60_len                       （V35.2 才實作）

SEED 行為保證（關鍵）
=====================
SEED 從 Gist 讀 params 時，沒有 V35 key → fallback 到 PARAMS_SPACE[key][0]：
  regime_gate_mode     → 3 (DISABLED)  → kernel 跳 if 分支，eff_threshold = buy_threshold
  bull/chop/bear_delta → 0            → 就算 mode 不小心被污染，eff = buy_threshold + 0
  max_pos_ovr 三個    → 0 (未實作)   → 無影響
  regime_ma_len 兩個   → default     → 無影響（kernel 不讀）

雙重保險：SEED 行為 100% 退化成 base（跟 89.90 在 base 跑一模一樣）

架構（同 V34 的 monkey-patch 模式）
====
1. Import base gpu_cupy_evolve
2. PARAMS_SPACE / PARAM_ORDER 末尾加 9 slot（74→83），但 5 個鎖 [0] 只剩 4 個真正 GPU 會選
3. 重建 kernel：signature 在 params 前加 regime_arr；2 處 buy gate 前算 eff_buy_threshold
4. Kernel wrapper 在 tail(6) 前插入 d_regime_arr（抗 base 前段改動）
5. 擴充 precompute：算 regime array 放 GPU
6. 直接呼叫 base.main() — 其他邏輯全繼承

cpu_replay 設計（2026-04-25 稽查後修正：v0.3）
================
**v0.2 版本錯誤設計**：cpu_replay 不 mirror regime gate。
**問題**：真正 regime-aware 強策略（kernel 靠避開 BEAR 期得分 85）在 cpu_replay（不看 regime）跑出來會被 BEAR 期拖累 → total 跟 SEED 差不多 → vs_seed gate 擋下
→ **真突破全被誤擋**

**v0.3 修復**：在 base.cpu_replay 加 `_eff_buy_th(day)` helper，兩處 buy gate 都用
  - SEED regime_gate_mode=3 (DISABLED) → _v35_active = False → _eff_buy_th 回傳原 buy_threshold → 行為同 base
  - 新候選 regime_gate_mode=0 → _v35_active = True → cpu_replay mirror kernel 的 regime 邏輯
  - kernel 跟 cpu_replay 對同組 params 跑出相同 trades
  - vs_seed gate 公平比較

與 V34 的關係
=============
- V35 和 V34 互不依賴（獨立 fork）
- 不能在同一 Python process 同時 import 兩者（都會擴 PARAM_ORDER）
- 未來 ensemble 方向：V34 margin + V35 regime 同時上線分資金
"""
import os, sys
import numpy as np
import cupy as cp

# path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path:
    sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base


# ══════════════════════════════════════════════════════════════
# 1. 擴充 PARAMS_SPACE + PARAM_ORDER
#    關鍵設計：每個 param [0] 位置放「SEED default（= 退化成 base 行為）」
#    Gist SEED 沒有 V35 key → params_np fallback 用 _opts[0] → 退化保證
# ══════════════════════════════════════════════════════════════
V35_NEW_PARAMS = {
    # 有效 params（GPU 會選）
    "bull_buy_th_delta": [0, -3, -2, -1, 1],   # [0] 放首 = SEED default
    "bear_buy_th_delta": [0, 1, 2, 3],          # [0] 放首
    "chop_buy_th_delta": [0, -1, 1, 2],         # [0] 放首
    "regime_gate_mode":  [3, 0],                # 3=DISABLED 放首 = SEED default，0=啟用 threshold gate

    # 鎖死 [0] 的 slot（留著給 V35.1/V35.2，本版 GPU 選不了）
    "bull_max_pos_ovr":  [0],
    "bear_max_pos_ovr":  [0],
    "chop_max_pos_ovr":  [0],
    "regime_ma20_len":   [20],   # 本版 precompute 固定用 20
    "regime_ma60_len":   [60],   # 本版 precompute 固定用 60
}

for _k, _v in V35_NEW_PARAMS.items():
    base.PARAMS_SPACE[_k] = _v

# PARAM_ORDER 順序必須跟 kernel p[74..82] 對應（順序敏感）
_V35_PARAM_NAMES = [
    "bull_buy_th_delta",  # p[74]
    "bear_buy_th_delta",  # p[75]
    "chop_buy_th_delta",  # p[76]
    "bull_max_pos_ovr",   # p[77] (本版 kernel 不讀)
    "bear_max_pos_ovr",   # p[78] (本版 kernel 不讀)
    "chop_max_pos_ovr",   # p[79] (本版 kernel 不讀)
    "regime_ma20_len",    # p[80] (precompute 用，kernel 不讀)
    "regime_ma60_len",    # p[81] (precompute 用，kernel 不讀)
    "regime_gate_mode",   # p[82]
]

_V35_PARAM_START = len(base.PARAM_ORDER)
_EXPECTED_BASE_N_PARAMS = 74
if _V35_PARAM_START != _EXPECTED_BASE_N_PARAMS:
    raise RuntimeError(
        f"[V35] base.PARAM_ORDER 長度 {_V35_PARAM_START}，預期 {_EXPECTED_BASE_N_PARAMS}。\n"
        f"可能已混入 V34 或 base 改過。V35 必須 fresh 跑（不要在 V34 後直接 import）"
    )
base.PARAM_ORDER.extend(_V35_PARAM_NAMES)
assert len(base.PARAM_ORDER) == 83, f"[V35] PARAM_ORDER 擴充後應為 83，實際 {len(base.PARAM_ORDER)}"


# ══════════════════════════════════════════════════════════════
# 2. 改 CUDA kernel 字串
# ══════════════════════════════════════════════════════════════
def _build_v35_kernel_src(base_src: str) -> str:
    """
    3 處修改：
      A. signature: 在 `const float* params` 前加 `const float* regime_arr`
      B. p[74..76] 的 MA/MOM reads 後移到 p[83..85]，中間塞 V35 param reads
      C. 2 處 `if (sc >= buy_threshold ...)` 前加 regime-effective threshold 計算

    關鍵：regime_gate_mode == 3 = DISABLED → kernel 跳 if 分支 → eff = buy_threshold
    確保 SEED（mode=3）行為 100% 等於 base
    """
    src = base_src

    # --- A. signature ---
    old_sig = "    const float* params, const int n_params_per_combo,"
    new_sig = "    const float* regime_arr,\n    const float* params, const int n_params_per_combo,"
    if old_sig not in src:
        raise RuntimeError("[V35] signature anchor not found")
    src = src.replace(old_sig, new_sig)

    # --- B. MA/MOM reads 後移 + 插入 V35 params ---
    old_ma = (
        "int ma_fast_idx = (int)p[74];\n"
        "    int ma_slow_idx = (int)p[75];\n"
        "    int mom_idx = (int)p[76];"
    )
    new_ma = (
        "// V35 Regime params (p[74..82])\n"
        "    float bull_buy_th_delta = p[74];\n"
        "    float bear_buy_th_delta = p[75];\n"
        "    float chop_buy_th_delta = p[76];\n"
        "    // p[77..79] max_pos override (V35.1, kernel 不讀)\n"
        "    // p[80..81] regime_ma length (precompute 用，kernel 不讀)\n"
        "    int regime_gate_mode = (int)p[82];\n"
        "    int ma_fast_idx = (int)p[83];\n"
        "    int ma_slow_idx = (int)p[84];\n"
        "    int mom_idx = (int)p[85];"
    )
    if old_ma not in src:
        raise RuntimeError("[V35] MA/MOM reads anchor not found")
    src = src.replace(old_ma, new_ma)

    # --- C. 2 處 buy gate 加 regime-effective threshold ---
    # 關鍵：regime_gate_mode == 0 → 啟用 threshold gate
    #       regime_gate_mode == 3 (DISABLED) → 跳 if 分支 → eff = buy_threshold（= base 行為）
    regime_prologue = (
        "// V35 Regime-aware threshold\n"
        "                float eff_buy_threshold = buy_threshold;\n"
        "                if (regime_gate_mode == 0) {\n"
        "                    int _r = (int)regime_arr[day];\n"
        "                    if (_r == 0) eff_buy_threshold = buy_threshold + bull_buy_th_delta;\n"
        "                    else if (_r == 1) eff_buy_threshold = buy_threshold + bear_buy_th_delta;\n"
        "                    else eff_buy_threshold = buy_threshold + chop_buy_th_delta;\n"
        "                    if (eff_buy_threshold < 1.0f) eff_buy_threshold = 1.0f;\n"
        "                }\n"
    )

    old_gate_1 = (
        "if (sc >= buy_threshold && (sc > cand_sc || "
        "(sc == cand_sc && vol_ratio[d] > cand_vol))) "
        "{ cand_si = si; cand_sc = sc; cand_vol = vol_ratio[d]; }"
    )
    old_gate_2 = (
        "if (sc >= buy_threshold && (sc > best_buy_score || "
        "(sc == best_buy_score && vol_ratio[d] > best_buy_vol))) {\n"
        "                    best_si = si; best_buy_score = sc; best_buy_vol = vol_ratio[d];\n"
        "                }"
    )

    new_gate_1 = (
        regime_prologue
        + "                if (sc >= eff_buy_threshold && (sc > cand_sc || "
        "(sc == cand_sc && vol_ratio[d] > cand_vol))) "
        "{ cand_si = si; cand_sc = sc; cand_vol = vol_ratio[d]; }"
    )
    new_gate_2 = (
        regime_prologue
        + "                if (sc >= eff_buy_threshold && (sc > best_buy_score || "
        "(sc == best_buy_score && vol_ratio[d] > best_buy_vol))) {\n"
        "                    best_si = si; best_buy_score = sc; best_buy_vol = vol_ratio[d];\n"
        "                }"
    )

    c1 = src.count(old_gate_1)
    c2 = src.count(old_gate_2)
    if c1 != 1:
        raise RuntimeError(f"[V35] gate_1 anchor count {c1} != 1 — base kernel 可能改了")
    if c2 != 1:
        raise RuntimeError(f"[V35] gate_2 anchor count {c2} != 1 — base kernel 可能改了")
    src = src.replace(old_gate_1, new_gate_1)
    src = src.replace(old_gate_2, new_gate_2)

    return src


print("[V35] 重建 CUDA kernel 字串（regime-aware）...")
_V35_KERNEL_SRC = _build_v35_kernel_src(base.CUDA_KERNEL.code)
print(f"[V35] kernel source: {len(_V35_KERNEL_SRC)} chars")
_V35_RAW_KERNEL = cp.RawKernel(_V35_KERNEL_SRC, "backtest")


# ══════════════════════════════════════════════════════════════
# 3. 擴充 precompute：算 regime array 並放 GPU
# ══════════════════════════════════════════════════════════════
_orig_precompute = base.precompute
_v35_regime_gpu = None  # cupy float32 array shape (n_days,)


def _compute_regime_array(market_close: np.ndarray, ma20_len: int, ma60_len: int, n_days: int) -> np.ndarray:
    """
    依 market_close 算 regime array，shape (n_days,) dtype float32，值 {0.0, 1.0, 2.0}
      0 = BULL:  MA_short > MA_long AND close > MA_short
      1 = BEAR:  MA_short < MA_long AND close < MA_short
      2 = CHOP:  其他（包含 warmup 期）
    """
    ma_short = np.zeros(n_days, dtype=np.float32)
    ma_long = np.zeros(n_days, dtype=np.float32)
    for i in range(n_days):
        if i >= ma20_len:
            ma_short[i] = market_close[max(0, i - ma20_len + 1):i + 1].mean()
        if i >= ma60_len:
            ma_long[i] = market_close[max(0, i - ma60_len + 1):i + 1].mean()

    regime = np.full(n_days, 2.0, dtype=np.float32)  # 預設 CHOP
    for i in range(max(ma20_len, ma60_len), n_days):
        if ma_short[i] > ma_long[i] and market_close[i] > ma_short[i]:
            regime[i] = 0.0
        elif ma_short[i] < ma_long[i] and market_close[i] < ma_short[i]:
            regime[i] = 1.0
    return regime


def v35_precompute(data):
    """
    base.precompute 之後算 regime array 放 GPU
    """
    global _v35_regime_gpu
    pre = _orig_precompute(data)

    close = pre["close"]
    n_days = pre["n_days"]

    # market_close NaN 檢查（防 base 未來改動引入 NaN）
    if np.any(np.isnan(close)):
        _nan_pct = np.isnan(close).mean() * 100
        print(f"[V35] ⚠️ close 含 {_nan_pct:.2f}% NaN，使用 nanmean 算 market_close")
        market_close = np.nanmean(close, axis=0)
    else:
        market_close = close.mean(axis=0)

    # 防 market_close 本身有 NaN（極端情況全部股票當天 NaN）
    if np.any(np.isnan(market_close)):
        _n = np.isnan(market_close).sum()
        print(f"[V35] ⚠️ market_close {_n} 天全 NaN，用 forward fill")
        # forward fill
        last_valid = None
        for i in range(n_days):
            if np.isnan(market_close[i]):
                market_close[i] = last_valid if last_valid is not None else 1.0
            else:
                last_valid = market_close[i]

    default_ma20 = 20
    default_ma60 = 60
    regime = _compute_regime_array(market_close, default_ma20, default_ma60, n_days)

    # regime sanity check
    assert regime.shape == (n_days,), f"[V35] regime shape {regime.shape} 錯誤"
    assert regime.dtype == np.float32
    _unique = set(np.unique(regime).tolist())
    _expected = {0.0, 1.0, 2.0}
    if not _unique.issubset(_expected):
        raise RuntimeError(f"[V35] regime 出現非預期值 {_unique - _expected}")

    # 分布統計
    bull_n = int((regime == 0).sum())
    bear_n = int((regime == 1).sum())
    chop_n = int((regime == 2).sum())
    print(f"[V35] regime 分布（ma20={default_ma20}, ma60={default_ma60}，共 {n_days} 天）:")
    print(f"[V35]   🟢 BULL: {bull_n} 天 ({bull_n/n_days*100:.1f}%)")
    print(f"[V35]   🔴 BEAR: {bear_n} 天 ({bear_n/n_days*100:.1f}%)")
    print(f"[V35]   🟡 CHOP: {chop_n} 天 ({chop_n/n_days*100:.1f}%)")

    # reverse WF 的 test 期（舊期，含 2022 熊市）regime 分布
    test_end = 60 + (n_days - 60) // 3
    if test_end < n_days:
        regime_test = regime[60:test_end]
        tb = int((regime_test == 0).sum())
        te = int((regime_test == 1).sum())
        tc = int((regime_test == 2).sum())
        tot = max(len(regime_test), 1)
        print(f"[V35] test 期（reverse WF 舊期）regime: "
              f"BULL {tb}({tb/tot*100:.0f}%) / BEAR {te}({te/tot*100:.0f}%) / CHOP {tc}({tc/tot*100:.0f}%)")
        if te / tot < 0.15:
            print(f"[V35] ⚠️ 舊期 BEAR 比例偏低（{te/tot*100:.0f}%），regime-aware 優勢可能有限")
        else:
            print(f"[V35] ✅ 舊期 BEAR 比例 {te/tot*100:.0f}% 充足，regime-aware 有發揮空間")

    # 丟上 GPU + memory layout assert
    _tmp = np.ascontiguousarray(regime, dtype=np.float32)
    _v35_regime_gpu = cp.asarray(_tmp)
    # BUG G fix: strides 檢查
    assert _v35_regime_gpu.flags["C_CONTIGUOUS"], "[V35] regime tensor 不是 C-contiguous"
    assert _v35_regime_gpu.dtype == cp.float32, f"[V35] regime dtype {_v35_regime_gpu.dtype} 不是 float32"
    assert _v35_regime_gpu.shape == (n_days,), f"[V35] regime shape {_v35_regime_gpu.shape} 不是 ({n_days},)"
    assert _v35_regime_gpu.strides == (4,), f"[V35] regime strides {_v35_regime_gpu.strides} 不是 (4,)"
    print(f"[V35] ✅ regime tensor on GPU: shape={_v35_regime_gpu.shape} dtype={_v35_regime_gpu.dtype} strides={_v35_regime_gpu.strides}")

    # V35_REGIME_DUMP=1 印詳細轉換時序
    if os.environ.get("V35_REGIME_DUMP") == "1":
        dates = pre["dates"]
        transitions = []
        prev = -1
        for i in range(n_days):
            cur = int(regime[i])
            if cur != prev and prev >= 0:
                transitions.append((i, prev, cur, str(dates[i].date())))
            prev = cur
        print(f"[V35] regime 轉換 {len(transitions)} 次:")
        names = ["BULL", "BEAR", "CHOP"]
        for di, frm, to, ds in transitions[:25]:
            print(f"[V35]   day {di} ({ds}): {names[frm]} -> {names[to]}")
        if len(transitions) > 25:
            print(f"[V35]   ... 另 {len(transitions)-25} 次")

    pre["regime_arr"] = regime
    pre["regime_arr_gpu"] = _v35_regime_gpu
    return pre


base.precompute = v35_precompute


# ══════════════════════════════════════════════════════════════
# 4. Kernel wrapper — 在 tail(6) 前插入 d_regime_arr
# ══════════════════════════════════════════════════════════════
class V35KernelWrapper:
    """
    base args 尾 6 個固定：d_params, n_params_int, d_results, n_combos_int, tr_start_int, tr_end_int
    V35 在倒數第 7 位（d_params 之前）插入 d_regime_arr

    BUG D fix: 加嚴 args 結構檢查
      - len(args) 必須 >= tail + 30（base 至少 30+ 個 indicator tensor）
      - 最後 6 個必須是 int/cupy array 結構
    """
    _TAIL_LEN = 6
    _MIN_INDICATOR_ARGS = 30  # base 有 30+ indicator tensor

    def __init__(self, real_kernel, kernel_src):
        self.real = real_kernel
        self.code = kernel_src

    def __call__(self, grid, block, args):
        if _v35_regime_gpu is None:
            raise RuntimeError("[V35] regime tensor 未載入，請先跑 precompute()")
        args = tuple(args)

        if len(args) < self._TAIL_LEN + self._MIN_INDICATOR_ARGS:
            raise RuntimeError(
                f"[V35] kernel args 長度 {len(args)} 太短（base 至少應有 "
                f"{self._TAIL_LEN + self._MIN_INDICATOR_ARGS}+ 個）— base 簽名可能已改"
            )

        # 檢查 tail(6) 結構：應該是 (d_params, n_params_int, d_results, n_combos_int, tr_start_int, tr_end_int)
        tail = args[-self._TAIL_LEN:]
        # d_params 應為 cupy array
        if not isinstance(tail[0], cp.ndarray):
            raise RuntimeError(f"[V35] args[-6] 應為 cupy array (d_params)，實際 {type(tail[0])}")
        # n_params_int 應為 numpy int 或 int
        if not isinstance(tail[1], (np.integer, int)):
            raise RuntimeError(f"[V35] args[-5] 應為 int (n_params)，實際 {type(tail[1])}")
        # d_results 應為 cupy array
        if not isinstance(tail[2], cp.ndarray):
            raise RuntimeError(f"[V35] args[-4] 應為 cupy array (d_results)，實際 {type(tail[2])}")
        # 其他 3 個 int 省略檢查（前 3 個對就很強保證）

        new_args = args[:-self._TAIL_LEN] + (_v35_regime_gpu,) + args[-self._TAIL_LEN:]
        return self.real(grid, block, new_args)


base.CUDA_KERNEL = V35KernelWrapper(_V35_RAW_KERNEL, _V35_KERNEL_SRC)


# ══════════════════════════════════════════════════════════════
# 5. cpu_replay（passthrough — SEED 有 mode=3 保證一致，新候選故意有落差當 overfit filter）
# ══════════════════════════════════════════════════════════════
_orig_cpu_replay = base.cpu_replay


def v35_cpu_replay(pre, p):
    """
    v0.3：base.cpu_replay 已加 regime-aware 邏輯（_eff_buy_th helper）
    V35 不需要覆寫，passthrough 即可。

    base.cpu_replay 讀 pre["regime_arr"]（V35 precompute 塞入）+ p["regime_gate_mode"]
      - mode=3 (DISABLED) or V35 key 不存在 → 退化成純 base 行為
      - mode=0 + 有 delta → mirror kernel 的 regime gate 邏輯

    這樣 kernel 跟 cpu_replay 對同 params 產出相同 trades，vs_seed gate 公平。
    """
    return _orig_cpu_replay(pre, p)


base.cpu_replay = v35_cpu_replay


# ══════════════════════════════════════════════════════════════
# 6. Self-test：validate SEED 行為退化成 base
# ══════════════════════════════════════════════════════════════
def _v35_self_test():
    """
    開跑前最後防線：模擬 SEED params 餵 PARAMS_SPACE[key][0] 後，
    驗證 regime_gate_mode == 3 (DISABLED)、delta 全為 0
    """
    # SEED 拿不到 V35 key，fallback 用 _opts[0]
    _test_vals = {k: v[0] for k, v in V35_NEW_PARAMS.items()}
    errors = []
    if _test_vals["regime_gate_mode"] != 3:
        errors.append(f"regime_gate_mode default={_test_vals['regime_gate_mode']}，應為 3 (DISABLED)")
    for dk in ["bull_buy_th_delta", "bear_buy_th_delta", "chop_buy_th_delta"]:
        if _test_vals[dk] != 0:
            errors.append(f"{dk} default={_test_vals[dk]}，應為 0")
    if errors:
        raise RuntimeError(f"[V35] SEED 行為退化 self-test 失敗：{errors}")
    print(f"[V35] ✅ SEED self-test 通過：regime_gate_mode 預設 3 (DISABLED)，所有 delta 預設 0")


_v35_self_test()


# ══════════════════════════════════════════════════════════════
# 7. 進入 base.main()
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[V35] PARAMS_SPACE 大小: {len(base.PARAMS_SPACE)}")
    print(f"[V35] PARAM_ORDER 長度: {len(base.PARAM_ORDER)}（應為 83）")
    if len(base.PARAM_ORDER) != 83:
        raise RuntimeError(f"[V35] PARAM_ORDER 長度 {len(base.PARAM_ORDER)} 不是 83")

    print(f"[V35] 💡 SEED regime_gate_mode=3 (DISABLED) → kernel 退化成 base → baseline 完美一致")
    print(f"[V35] 💡 攻擊面：89.90 在 2022 熊市勝率 54.5%（整體 63.9%）→ sanity 已確認 -9.4% gap")
    print(f"[V35] 💡 GPU 會選：regime_gate_mode=0 AND bear_delta=+2 or +3 → 擋 BEAR 爛訊號")
    print(f"[V35] 啟動 base.main()...\n")
    base.main()
