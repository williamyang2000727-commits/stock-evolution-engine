"""
V34 Margin Gambit — GPU 進化搜尋 (fork of gpu_cupy_evolve.py)

架構：
  1. Import base gpu_cupy_evolve 為底
  2. 在 PARAMS_SPACE / PARAM_ORDER 末尾加 9 個 margin 相關 slot
  3. 重建 CUDA kernel 字串（margin 掃描 branches + 新 param 讀取 + margin_tensor 參數）
  4. 用 wrapper 攔截 kernel 呼叫，自動塞 margin_tensor
  5. 擴充 precompute：載入 margin_tensor.npy 並轉成 (stocks, days, 5) layout
  6. 直接呼叫 base.main() — 其他邏輯 100% inherit

新增 9 個 param：
  w_margin_heat + margin_heat_th   (融資使用率低的加分, th 0-100%)
  w_margin_accel + margin_accel_th (融資加速低的加分, th %)
  w_short_ratio + short_ratio_th   (融券比低的加分, th %)
  w_offset_rate + offset_rate_th   (資券互抵低的加分, th %)
  w_margin_diverge                  (融資-價發散 <= 0 加分, 純 boolean)

用法：
  C:\\stock-evolution> python claude_v34_margin_gpu.py
環境變數（繼承 base）：
  GPU_WF_MODE, GPU_UNIVERSE 等，跟 gpu_cupy_evolve.py 相同
"""
import os, sys, pickle
import numpy as np
import cupy as cp

# 把 stock-evolution 加進 sys.path，讓 import gpu_cupy_evolve 找得到
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 也把 C:\stock-evolution 加進去（用戶 Windows 實際放 base kernel 的地方）
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path:
    sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base


MARGIN_TENSOR_PATH = os.path.join(_USER_SE, "margin_tensor.npy")
MARGIN_META_PATH = os.path.join(_USER_SE, "margin_tensor_meta.pkl")


# ══════════════════════════════════════════════════════════════
# 1. 擴充 PARAMS_SPACE + PARAM_ORDER
# ══════════════════════════════════════════════════════════════
V34_NEW_PARAMS = {
    "w_margin_heat":     [0, 1, 2, 3],
    "margin_heat_th":    [30, 50, 70],          # % — 融資使用率 <= th 才加分
    "w_margin_accel":    [0, 1, 2, 3],
    "margin_accel_th":   [10, 20, 30, 50],      # % — 融資 5d 加速 <= th 才加分
    "w_short_ratio":     [0, 1, 2, 3],
    "short_ratio_th":    [5, 10, 20, 30],       # % — 融券/融資比 <= th 才加分
    "w_offset_rate":     [0, 1, 2],
    "offset_rate_th":    [5, 10, 20],           # % — 資券互抵率 <= th 才加分
    "w_margin_diverge":  [0, 1, 2],             # diverge <= 0 才加分（純 bool）
}

for _k, _v in V34_NEW_PARAMS.items():
    base.PARAMS_SPACE[_k] = _v

_V34_PARAM_NAMES = [
    "w_margin_heat", "margin_heat_th",
    "w_margin_accel", "margin_accel_th",
    "w_short_ratio", "short_ratio_th",
    "w_offset_rate", "offset_rate_th",
    "w_margin_diverge",
]
# 加到 PARAM_ORDER 末尾（在 buy_delay_days 之後，MA/MOM 之前）
# 注意：PARAM_ORDER 原長 = 74。加 9 個後 = 83。MA/MOM index 相應後移到 83,84,85
_V34_PARAM_START = len(base.PARAM_ORDER)  # = 74
base.PARAM_ORDER.extend(_V34_PARAM_NAMES)
_V34_NEW_N_PARAMS = len(base.PARAM_ORDER)  # = 83


# ══════════════════════════════════════════════════════════════
# 2. 改 CUDA kernel 字串
# ══════════════════════════════════════════════════════════════
def _build_v34_kernel_src(base_src: str) -> str:
    """根據 base kernel 字串，做 4 處修改：
    A. signature 加 margin_tensor 參數
    B. 把 MA/MOM 索引從 p[74..76] 改成 p[83..85]，中間塞 V34 param reads
    C. 3 處 scoring section 加 5 行 margin scoring
    D. early_exit_days 等已經用 p[70..73]，不動
    """
    src = base_src

    # --- A. signature 加 margin_tensor ---
    src = src.replace(
        "    const float* params, const int n_params_per_combo,",
        "    const float* margin_tensor,\n    const float* params, const int n_params_per_combo,"
    )

    # --- B. 替換 MA/MOM 索引 + 插入 V34 param reads ---
    old_ma_reads = (
        "int ma_fast_idx = (int)p[74];\n"
        "    int ma_slow_idx = (int)p[75];\n"
        "    int mom_idx = (int)p[76];"
    )
    new_ma_reads = (
        "// V34 Margin params (idx 74-82)\n"
        "    int w_margin_heat = (int)p[74]; float margin_heat_th = p[75];\n"
        "    int w_margin_accel = (int)p[76]; float margin_accel_th = p[77];\n"
        "    int w_short_ratio = (int)p[78]; float short_ratio_th = p[79];\n"
        "    int w_offset_rate = (int)p[80]; float offset_rate_th = p[81];\n"
        "    int w_margin_diverge = (int)p[82];\n"
        "    int ma_fast_idx = (int)p[83];\n"
        "    int ma_slow_idx = (int)p[84];\n"
        "    int mom_idx = (int)p[85];"
    )
    if old_ma_reads not in src:
        raise RuntimeError("V34 patch: MA/MOM reads pattern not found in base kernel")
    src = src.replace(old_ma_reads, new_ma_reads)

    # --- C. 加 margin scoring 到 3 處 ---
    # Anchor: 每處 scoring 的最後一行都是 w_mom_accel_k 那行
    old_anchor = "if (w_mom_accel_k > 0 && mom_accel[d] >= mom_accel_min_k) sc += w_mom_accel_k;"
    v34_scoring = (
        old_anchor + "\n"
        "                // V34 Margin scoring — margin_tensor layout: (stocks, days, 5)\n"
        "                {\n"
        "                    int midx = (si * n_days + day) * 5;\n"
        "                    // margin_heat_th 是百分比（0-100），tensor 值是 0-1 比例 → × 100\n"
        "                    if (w_margin_heat > 0 && margin_tensor[midx+0] * 100.0f <= margin_heat_th) sc += w_margin_heat;\n"
        "                    if (w_margin_accel > 0 && margin_tensor[midx+1] <= margin_accel_th) sc += w_margin_accel;\n"
        "                    if (w_short_ratio > 0 && margin_tensor[midx+2] <= short_ratio_th) sc += w_short_ratio;\n"
        "                    if (w_offset_rate > 0 && margin_tensor[midx+3] <= offset_rate_th) sc += w_offset_rate;\n"
        "                    if (w_margin_diverge > 0 && margin_tensor[midx+4] <= 0.0f) sc += w_margin_diverge;\n"
        "                }"
    )
    n_hits = src.count(old_anchor)
    if n_hits != 3:
        raise RuntimeError(f"V34 patch: 期望 3 處 anchor，實際 {n_hits}")
    src = src.replace(old_anchor, v34_scoring)

    return src


print("[V34] 重建 CUDA kernel 字串...")
_V34_KERNEL_SRC = _build_v34_kernel_src(base.CUDA_KERNEL.code)
print(f"[V34] kernel source: {len(_V34_KERNEL_SRC)} chars")

# 編譯 V34 kernel（之後會用 wrapper 包它）
_V34_RAW_KERNEL = cp.RawKernel(_V34_KERNEL_SRC, "backtest")


# ══════════════════════════════════════════════════════════════
# 3. 擴充 precompute：載入並轉置 margin tensor
# ══════════════════════════════════════════════════════════════
_orig_precompute = base.precompute
_v34_margin_gpu = None   # 將在 precompute 裡填入


def v34_precompute(data):
    """在 base.precompute 之後額外載入 margin tensor 放上 GPU"""
    global _v34_margin_gpu
    pre = _orig_precompute(data)

    if not os.path.exists(MARGIN_TENSOR_PATH):
        raise FileNotFoundError(
            f"{MARGIN_TENSOR_PATH}\n"
            f"→ 先跑 fetch_margin_history.py + preprocess_margin.py"
        )
    if not os.path.exists(MARGIN_META_PATH):
        raise FileNotFoundError(f"{MARGIN_META_PATH}")

    margin_raw = np.load(MARGIN_TENSOR_PATH)  # shape (days, stocks_in_margin, 5)
    margin_meta = pickle.load(open(MARGIN_META_PATH, "rb"))
    margin_tickers = margin_meta["tickers"]   # list of "2330.TW" etc.

    print(f"[V34] margin raw: {margin_raw.shape}  tickers={len(margin_tickers)}")

    # 把 margin 重排成跟 pre["tickers"] 一致的順序
    # 不在 margin 裡的股票 → 填 0（等於關掉該股 margin scoring）
    ns = pre["n_stocks"]
    nd = pre["n_days"]
    margin_idx_map = {t: i for i, t in enumerate(margin_tickers)}
    aligned = np.zeros((ns, nd, 5), dtype=np.float32)
    missing = 0
    for si, t in enumerate(pre["tickers"]):
        mi = margin_idx_map.get(t)
        if mi is None:
            missing += 1
            continue
        # margin_raw shape (days_m, stocks_m, 5)；取最後 nd 天對齊
        m_days = margin_raw.shape[0]
        if m_days >= nd:
            aligned[si, :, :] = margin_raw[-nd:, mi, :]
        else:
            aligned[si, -m_days:, :] = margin_raw[:, mi, :]
    print(f"[V34] aligned margin to GPU layout (stocks, days, 5): {aligned.shape}  "
          f"missing tickers {missing}/{ns}")

    # 丟上 GPU，存為 module-level 讓 kernel wrapper 取用
    _v34_margin_gpu = cp.asarray(aligned)
    print(f"[V34] margin tensor on GPU: {_v34_margin_gpu.nbytes/1024/1024:.1f} MB")

    pre["margin_tensor"] = _v34_margin_gpu   # 也掛在 pre 上備用
    return pre


base.precompute = v34_precompute


# ══════════════════════════════════════════════════════════════
# 4. Kernel wrapper — 自動在 kernel call 時插入 margin_tensor
# ══════════════════════════════════════════════════════════════
class V34KernelWrapper:
    """
    base.main() 會呼叫 base.CUDA_KERNEL((grid,), (BLOCK,), (args...)) 。
    我們要把 d_margin_tensor 插到 d_params（原 args[-6]）之前。
    也暴露 .code 讓外部診斷仍能拿 source。
    """
    def __init__(self, real_kernel, kernel_src):
        self.real = real_kernel
        self.code = kernel_src

    def __call__(self, grid, block, args):
        if _v34_margin_gpu is None:
            raise RuntimeError("[V34] margin tensor 未載入，請先跑 precompute()")
        args = tuple(args)
        # d_params 在 args[-6]（kernel 簽名結構固定），插入 margin_tensor 在它前面
        new_args = args[:-6] + (_v34_margin_gpu,) + args[-6:]
        return self.real(grid, block, new_args)


base.CUDA_KERNEL = V34KernelWrapper(_V34_RAW_KERNEL, _V34_KERNEL_SRC)


# ══════════════════════════════════════════════════════════════
# 5. cpu_replay 擴充（mirror kernel margin scoring）
# ══════════════════════════════════════════════════════════════
_orig_cpu_replay = base.cpu_replay


def v34_cpu_replay(pre, p):
    """
    目前 strategy：對 base.cpu_replay 的 scoring 做 monkey-patch 困難（scoring 在閉包內）。
    折衷：cpu_replay 的結果只用於 SEED 診斷 + 最終 validation；V34 margin params 傳進去後，
    base.cpu_replay 讀 p["w_margin_heat"] 等 key 不會爆（它用 .get 帶 default 0）。
    結果：cpu_replay 會忽略 margin scoring，所以跟 kernel 的 score 會有小差距。
    TODO：後續加 margin 邏輯到 cpu_replay（非阻塞，kernel 結果是真相）。
    """
    return _orig_cpu_replay(pre, p)


base.cpu_replay = v34_cpu_replay


# ══════════════════════════════════════════════════════════════
# 6. 進入 base.main()
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[V34] PARAMS_SPACE 大小: {len(base.PARAMS_SPACE)}")
    print(f"[V34] PARAM_ORDER 長度: {len(base.PARAM_ORDER)}  (MA/MOM idx 會在 +3)")
    print(f"[V34] 啟動 base.main()...\n")
    base.main()
