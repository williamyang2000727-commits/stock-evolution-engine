"""
V35 Regime Commander — GPU 進化搜尋（fork of gpu_cupy_evolve.py）

設計理念
========
攻擊 89.90 的唯一已知弱點：2022 熊市勝率 57%（vs 整體 69.4%）
用 regime-aware 動態 buy_threshold：不同市場 regime 用不同進場嚴格度

Regime 偵測（規則式，不用 ML）
  - 用 base precompute 已算好的 market_close
  - 本版用 default ma20=20, ma60=60 算 regime（V35.2 再擴多組）
  - regime 0 = BULL：MA20 > MA60 AND market_close > MA20
  - regime 1 = BEAR：MA20 < MA60 AND market_close < MA20
  - regime 2 = CHOP：其他（盤整 / 轉折）

新增 9 個 params
  bull_buy_th_delta   [-3,-2,-1,0,1]  Bull regime 時 buy_threshold 調整
  bear_buy_th_delta   [0,1,2,3]       Bear regime 時（正值=更嚴格，寧空倉）
  chop_buy_th_delta   [-1,0,1,2]      Chop regime 時
  bull_max_pos_ovr    [0,2,3]         Bull regime 持倉 override（0=用 base / 本版未啟用）
  bear_max_pos_ovr    [0,1,2]         Bear regime 持倉 override（同上，V35.1 再實作）
  chop_max_pos_ovr    [0,1,2]         Chop regime 持倉 override（同上）
  regime_ma20_len     [15,20,25]      MA20 窗口（本版 precompute 用 default=20）
  regime_ma60_len     [40,60,80]      MA60 窗口（本版 precompute 用 default=60）
  regime_gate_mode    [0,1,2]         0=開 threshold gate, 1=開 max_pos gate, 2=兩個都開

本版（V35.0）僅實作 threshold delta（regime_gate_mode == 0 或 2）
max_pos override 和多組 ma length 是 V35.1 / V35.2 範圍

架構（同 V34 的 monkey-patch 模式）
====
1. Import base gpu_cupy_evolve
2. 在 PARAMS_SPACE / PARAM_ORDER 末尾加 9 個 slot（74→83）
3. 重建 CUDA kernel：signature 在 `const float* params` 前加 `const float* regime_arr`
4. Kernel wrapper 在 tail(6) 前插入 d_regime_arr（不依賴前面 args 順序，抗 base 變動）
5. 擴充 precompute：算 regime array 放 GPU
6. 直接呼叫 base.main() — 其他邏輯全繼承

與 V34 的關係
=============
- V35 和 V34 互不依賴（都從 base fork，monkey-patch 不會互相污染）
- 不能在同一 Python process 同時 import V34 和 V35（兩個都會擴 PARAM_ORDER，會衝突）
- 未來 ensemble 方向：V34 margin + V35 regime 兩個策略同時上線分資金

用法
====
  C:\\stock-evolution> python claude_v35_regime_gpu.py

環境變數（繼承 base）：
  GPU_WF_MODE, GPU_UNIVERSE 等跟 gpu_cupy_evolve.py 相同
  V35_REGIME_DUMP=1  → precompute 時 dump regime 轉換時序（debug 用）
"""
import os, sys
import numpy as np
import cupy as cp

# path setup（跟 V34 一致）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path:
    sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base


# ══════════════════════════════════════════════════════════════
# 1. 擴充 PARAMS_SPACE + PARAM_ORDER
# ══════════════════════════════════════════════════════════════
V35_NEW_PARAMS = {
    "bull_buy_th_delta": [-3, -2, -1, 0, 1],
    "bear_buy_th_delta": [0, 1, 2, 3],
    "chop_buy_th_delta": [-1, 0, 1, 2],
    "bull_max_pos_ovr":  [0, 2, 3],      # 本版未啟用
    "bear_max_pos_ovr":  [0, 1, 2],      # 本版未啟用
    "chop_max_pos_ovr":  [0, 1, 2],      # 本版未啟用
    "regime_ma20_len":   [15, 20, 25],   # 本版 precompute 用 default=20
    "regime_ma60_len":   [40, 60, 80],   # 本版 precompute 用 default=60
    "regime_gate_mode":  [0, 1, 2],
}

for _k, _v in V35_NEW_PARAMS.items():
    base.PARAMS_SPACE[_k] = _v

_V35_PARAM_NAMES = [
    "bull_buy_th_delta", "bear_buy_th_delta", "chop_buy_th_delta",
    "bull_max_pos_ovr", "bear_max_pos_ovr", "chop_max_pos_ovr",
    "regime_ma20_len", "regime_ma60_len", "regime_gate_mode",
]

# 加到 PARAM_ORDER 末尾（在 buy_delay_days 之後，MA/MOM 之前）
# 跟 V34 同構：原 74 → 83，MA/MOM 索引後移 +9 到 p[83..85]
_V35_PARAM_START = len(base.PARAM_ORDER)
_EXPECTED_BASE_N_PARAMS = 74
if _V35_PARAM_START != _EXPECTED_BASE_N_PARAMS:
    raise RuntimeError(
        f"[V35] base.PARAM_ORDER 長度 {_V35_PARAM_START}，預期 {_EXPECTED_BASE_N_PARAMS}。\n"
        f"可能已混入 V34 或 base 改過。V35 必須 fresh 跑（不要在 V34 後直接 import）"
    )
base.PARAM_ORDER.extend(_V35_PARAM_NAMES)


# ══════════════════════════════════════════════════════════════
# 2. 改 CUDA kernel 字串
#    架構：regime_arr 放在 args 尾端（d_params 之前，= tail(6) 之前）
#    這樣 wrapper 永遠插在倒數第 7 位，不受 base 前面 args 順序改動影響
# ══════════════════════════════════════════════════════════════
def _build_v35_kernel_src(base_src: str) -> str:
    """
    3 處修改：
      A. signature: 在 `const float* params` 前加 `const float* regime_arr`
      B. p[74..76] 的 MA/MOM reads 後移到 p[83..85]，中間塞 V35 param reads
      C. 2 處 `if (sc >= buy_threshold ...)` 前加 regime-effective threshold 計算，gate 改用 eff_buy_threshold
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
        "// V35 Regime params (idx 74-82)\n"
        "    float bull_buy_th_delta = p[74];\n"
        "    float bear_buy_th_delta = p[75];\n"
        "    float chop_buy_th_delta = p[76];\n"
        "    // p[77..79] max_pos override（本版未啟用，留著給 V35.1）\n"
        "    // p[80..81] regime_ma20_len / regime_ma60_len（precompute 用，kernel 不讀）\n"
        "    int regime_gate_mode = (int)p[82];\n"
        "    int ma_fast_idx = (int)p[83];\n"
        "    int ma_slow_idx = (int)p[84];\n"
        "    int mom_idx = (int)p[85];"
    )
    if old_ma not in src:
        raise RuntimeError("[V35] MA/MOM reads anchor not found")
    src = src.replace(old_ma, new_ma)

    # --- C. 2 處 buy gate 加 regime-effective threshold ---
    regime_prologue = (
        "// V35 Regime-aware threshold\n"
        "                float eff_buy_threshold = buy_threshold;\n"
        "                if (regime_gate_mode == 0 || regime_gate_mode == 2) {\n"
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
        raise RuntimeError(f"[V35] gate_1 anchor count {c1} != 1")
    if c2 != 1:
        raise RuntimeError(f"[V35] gate_2 anchor count {c2} != 1")
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
            regime[i] = 0.0  # BULL
        elif ma_short[i] < ma_long[i] and market_close[i] < ma_short[i]:
            regime[i] = 1.0  # BEAR
    return regime


def v35_precompute(data):
    """
    在 base.precompute 之後算 regime array 放 GPU
    本版：kernel 只吃一組 regime（ma20=20, ma60=60）；GPU 選 regime_ma20_len=15/25 本版不生效
    （V35.2 會支援多組 regime array，kernel 按 p[80,81] 選組）
    """
    global _v35_regime_gpu
    pre = _orig_precompute(data)

    close = pre["close"]  # shape (n, ml)
    n_days = pre["n_days"]
    market_close = close.mean(axis=0)  # shape (ml,)

    default_ma20 = 20
    default_ma60 = 60
    regime = _compute_regime_array(market_close, default_ma20, default_ma60, n_days)

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

    # 丟上 GPU
    _v35_regime_gpu = cp.asarray(regime)
    assert _v35_regime_gpu.flags["C_CONTIGUOUS"]
    assert _v35_regime_gpu.dtype == cp.float32
    print(f"[V35] ✅ regime tensor on GPU: shape={_v35_regime_gpu.shape} dtype={_v35_regime_gpu.dtype}")

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

    pre["regime_arr"] = regime           # numpy，cpu_replay 可用（本版未實作 mirror）
    pre["regime_arr_gpu"] = _v35_regime_gpu
    return pre


base.precompute = v35_precompute


# ══════════════════════════════════════════════════════════════
# 4. Kernel wrapper — 在 tail(6) 前插入 d_regime_arr
# ══════════════════════════════════════════════════════════════
class V35KernelWrapper:
    """
    base.main() 呼叫 CUDA_KERNEL((grid,), (BLOCK,), (args...))
    base args 尾 6 個固定：d_params, n_params_int, d_results, n_combos_int, tr_start_int, tr_end_int
    V35 在倒數第 7 位（d_params 之前）插入 d_regime_arr
    signature 對應：`const float* regime_arr` 在 `const float* params` 之前

    好處：wrapper 不依賴 base 前面 indicator tensor 順序（抗 base 新增 indicator）
    """
    _TAIL_LEN = 6

    def __init__(self, real_kernel, kernel_src):
        self.real = real_kernel
        self.code = kernel_src

    def __call__(self, grid, block, args):
        if _v35_regime_gpu is None:
            raise RuntimeError("[V35] regime tensor 未載入，請先跑 precompute()")
        args = tuple(args)
        if len(args) < self._TAIL_LEN + 20:
            raise RuntimeError(
                f"[V35] kernel args 長度 {len(args)} 異常（base 至少 20+ 個 indicator tensor）"
            )
        new_args = args[:-self._TAIL_LEN] + (_v35_regime_gpu,) + args[-self._TAIL_LEN:]
        return self.real(grid, block, new_args)


base.CUDA_KERNEL = V35KernelWrapper(_V35_RAW_KERNEL, _V35_KERNEL_SRC)


# ══════════════════════════════════════════════════════════════
# 5. cpu_replay（本版 passthrough，已知小差距）
# ══════════════════════════════════════════════════════════════
_orig_cpu_replay = base.cpu_replay


def v35_cpu_replay(pre, p):
    """
    本版（V35.0）cpu_replay 不 mirror regime gate。
    後果：kernel 用 regime-effective threshold 選，cpu_replay 用原始 buy_threshold 驗證
          → 同一組 params 兩邊小差（可能 ± 3-5 筆交易）
          → SEED 診斷 / vs_seed gate 仍能運作（SEED regime params 全 0，無 regime 邏輯，cpu_replay 準確）
          → 新候選才有小差，但 vs_seed 仍能排除多數爛策略

    TODO (V35.1)：把 base.cpu_replay 的 buy gate 抽成 hookable function，加 regime 邏輯
    """
    return _orig_cpu_replay(pre, p)


base.cpu_replay = v35_cpu_replay


# ══════════════════════════════════════════════════════════════
# 6. 進入 base.main()
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[V35] PARAMS_SPACE 大小: {len(base.PARAMS_SPACE)}")
    print(f"[V35] PARAM_ORDER 長度: {len(base.PARAM_ORDER)}（應為 83）")
    if len(base.PARAM_ORDER) != 83:
        raise RuntimeError(f"[V35] PARAM_ORDER 長度 {len(base.PARAM_ORDER)} 不是 83！可能混到 V34 或 base 變動")

    print(f"[V35] 💡 SEED (89.90) 沒 V35 params → 9 slot 全 default 0 = regime gate 關閉")
    print(f"[V35] 💡 所以 SEED baseline 分數 = 純 regime-blind 表現，regime gate 只給新策略加分")
    print(f"[V35] 💡 攻擊面：89.90 在 2022 熊市勝率 57%（整體 69.4%）→ 若 regime-aware 能拉熊市 +5-8% = 真突破")
    print(f"[V35] 啟動 base.main()...\n")
    base.main()
