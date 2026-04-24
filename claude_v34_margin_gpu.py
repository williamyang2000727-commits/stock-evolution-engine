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
    # 注意：kernel 裡有的位置用 [d] 索引（買入當天），有的用 [day] 索引（換股 day）
    # 用 "sc += w_mom_accel_k;" 統一當 anchor，前面的日期索引不同但後綴相同
    # Sentinel guard：margin_tensor 值 > -900 才算有效（missing / warmup / NaN 填 -999）
    old_anchor = "if (w_mom_accel_k > 0 && mom_accel[d] >= mom_accel_min_k) sc += w_mom_accel_k;"
    v34_scoring = (
        old_anchor + "\n"
        "                // V34 Margin scoring — tensor layout (stocks, days, 5): heat/accel/short/offset/diverge\n"
        "                // Sentinel -999 = missing/warmup/NaN，> -900 才算有效資料\n"
        "                {\n"
        "                    int midx = (si * n_days + day) * 5;\n"
        "                    float _m0 = margin_tensor[midx+0];\n"
        "                    float _m1 = margin_tensor[midx+1];\n"
        "                    float _m2 = margin_tensor[midx+2];\n"
        "                    float _m3 = margin_tensor[midx+3];\n"
        "                    float _m4 = margin_tensor[midx+4];\n"
        "                    if (w_margin_heat > 0 && _m0 > -900.0f && _m0 * 100.0f <= margin_heat_th) sc += w_margin_heat;\n"
        "                    if (w_margin_accel > 0 && _m1 > -900.0f && _m1 <= margin_accel_th) sc += w_margin_accel;\n"
        "                    if (w_short_ratio > 0 && _m2 > -900.0f && _m2 <= short_ratio_th) sc += w_short_ratio;\n"
        "                    if (w_offset_rate > 0 && _m3 > -900.0f && _m3 <= offset_rate_th) sc += w_offset_rate;\n"
        "                    if (w_margin_diverge > 0 && _m4 > -900.0f && _m4 <= 0.0f) sc += w_margin_diverge;\n"
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

# BUG #6 修正：驗證 base kernel 結構沒變（V34 patch 還對得上 base 的 param 索引）
# 如果 base 改了 PARAM_ORDER 長度，V34 的 p[74-82] 注入會錯位
_EXPECTED_BASE_N_PARAMS = 74  # V34 對應的 base version
if _V34_PARAM_START != _EXPECTED_BASE_N_PARAMS:
    raise RuntimeError(
        f"[V34] base.PARAM_ORDER 長度改變！原本 {_EXPECTED_BASE_N_PARAMS}，現在 {_V34_PARAM_START}。\n"
        f"V34 kernel patch 的 p[74-82] 索引會錯位，必須修正 _build_v34_kernel_src()"
    )

# 編譯 V34 kernel（之後會用 wrapper 包它）
_V34_RAW_KERNEL = cp.RawKernel(_V34_KERNEL_SRC, "backtest")


# ══════════════════════════════════════════════════════════════
# 3. 擴充 precompute：載入並轉置 margin tensor
# ══════════════════════════════════════════════════════════════
_orig_precompute = base.precompute
_v34_margin_gpu = None   # 將在 precompute 裡填入


def v34_precompute(data):
    """在 base.precompute 之後額外載入 margin tensor 放上 GPU

    Sentinel 設計（BUG #4/#5 修正 2026-04-25）：
      - missing stock / warmup 期 / 值為 0 的 early days → 填 -1
      - kernel 和 cpu_replay 用 `value > -900.0f` 檢查，避免把「沒資料」當成「低熱度加分」
    """
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
    margin_tickers = margin_meta["tickers"]

    print(f"[V34] margin raw: {margin_raw.shape}  tickers={len(margin_tickers)}")

    # 先取 ns/nd（BUG #12 檢查會用到，原本在下方 alignment 區才定義）
    ns = pre["n_stocks"]
    nd = pre["n_days"]

    # BUG #3 修正：時序方向驗證（assert margin_raw 軸 0 是從舊到新）
    # 檢查方式：比對 margin_meta 裡的 dates 和 pre["dates"] 最後一天
    _meta_dates = margin_meta.get("dates")
    _ohlcv_dates_str = [str(d.date() if hasattr(d, 'date') else d)[:10] for d in pre["dates"]]

    if _meta_dates is not None and len(_meta_dates) >= 2:
        _margin_dates_str = [str(md)[:10] for md in _meta_dates]
        _first_md = _margin_dates_str[0]
        _last_md = _margin_dates_str[-1]
        _pre_first = _ohlcv_dates_str[0]
        _pre_last = _ohlcv_dates_str[-1]
        print(f"[V34] margin dates: {_first_md} ~ {_last_md}  |  OHLCV dates: {_pre_first} ~ {_pre_last}")

        if _first_md > _last_md:
            raise RuntimeError(f"[V34] margin_raw 時序顛倒（{_first_md} > {_last_md}），先重跑 preprocess_margin.py 修正")

        # BUG #12 (2026-04-25)：交易日對齊精確性檢查
        # FinMind 和 TWSE 的交易日可能有差（除權息日、開盤異常日）
        # 比對最後 nd 天的 margin dates 和 OHLCV dates 逐天對應
        _m_last_nd = _margin_dates_str[-nd:] if len(_margin_dates_str) >= nd else _margin_dates_str
        _mismatch = 0
        _mismatch_examples = []
        for _i in range(min(len(_m_last_nd), nd)):
            _ohlcv_day = _ohlcv_dates_str[-(nd - _i)] if (nd - _i - 1) < len(_ohlcv_dates_str) else None
            _margin_day = _m_last_nd[_i]
            if _ohlcv_day and _ohlcv_day != _margin_day:
                _mismatch += 1
                if len(_mismatch_examples) < 3:
                    _mismatch_examples.append(f"idx={_i}: OHLCV={_ohlcv_day} vs margin={_margin_day}")
        if _mismatch == 0:
            print(f"[V34] ✅ 交易日 1:1 對齊（{min(len(_m_last_nd), nd)} 天全部一致）")
        else:
            _pct = _mismatch / min(len(_m_last_nd), nd) * 100
            print(f"[V34] ⚠️ 交易日錯位 {_mismatch}/{min(len(_m_last_nd), nd)} 天 ({_pct:.1f}%)")
            for _ex in _mismatch_examples:
                print(f"[V34]    {_ex}")
            if _pct > 5.0:
                raise RuntimeError(
                    f"[V34] 交易日錯位 {_pct:.1f}% 超過 5% 閾值，資料嚴重錯位。\n"
                    f"可能原因：FinMind 和 TWSE 交易日定義不同、margin_raw 天數 ({len(_margin_dates_str)}) "
                    f"vs OHLCV 天數 ({len(_ohlcv_dates_str)}) 不符"
                )
    else:
        print(f"[V34] ⚠️ margin_meta 沒有 dates 欄位，無法驗證時序方向和日期對齊")

    # 把 margin 重排成跟 pre["tickers"] 一致的順序
    # BUG #4 修正：missing stock 填 -999（sentinel）→ kernel/cpu_replay 用 > -900 guard 過濾
    # (ns/nd 已在上方定義)
    margin_idx_map = {t: i for i, t in enumerate(margin_tickers)}
    aligned = np.full((ns, nd, 5), -999.0, dtype=np.float32)  # -999 = sentinel
    missing = 0
    for si, t in enumerate(pre["tickers"]):
        mi = margin_idx_map.get(t)
        if mi is None:
            missing += 1
            continue  # 保持 -1
        # margin_raw shape (days_m, stocks_m, 5)；取最後 nd 天對齊
        m_days = margin_raw.shape[0]
        if m_days >= nd:
            aligned[si, :, :] = margin_raw[-nd:, mi, :]
        else:
            aligned[si, -m_days:, :] = margin_raw[:, mi, :]

    # BUG #5 修正：把 warmup 期（前 60 天）填 -1，避免「沒資料的 0」混進 scoring
    aligned[:, :60, :] = -999.0

    # BUG #14/#15 已在 preprocess_margin.py 修正：無效情境（NaN/除零）填 -1，不再靠 zero_mask 後處理
    # Legacy zero_mask 已移除（會誤殺合法的「當天 0 交易」）
    _sentinel_cnt = int((aligned <= -900.0).sum())

    print(f"[V34] aligned margin to GPU layout (stocks, days, 5): {aligned.shape}  "
          f"missing tickers {missing}/{ns}  sentinel(-1) 總數 {_sentinel_cnt:,}")

    # Sanity check：每維度的有效值範圍（sentinel -999 排除在外，只看 > -900）
    _dim_names = ["margin_heat (0-1 比例)", "margin_accel (%)", "short_ratio (%)", "offset_rate (%)", "margin_diverge (%)"]
    # BUG #14/#15 後：無效已填 -1，有效值範圍就是 clip 的上下限
    _expect_ranges = [(0.0, 1.0), (-100.0, 200.0), (0.0, 500.0), (0.0, 200.0), (-200.0, 200.0)]
    for _di, (_name, (_lo, _hi)) in enumerate(zip(_dim_names, _expect_ranges)):
        _valid = aligned[:, :, _di][aligned[:, :, _di] > -900.0]
        if len(_valid) > 0:
            _min, _max, _mean = float(_valid.min()), float(_valid.max()), float(_valid.mean())
            _ok = "✅" if _lo <= _min and _max <= _hi else "⚠️"
            print(f"[V34] dim[{_di}] {_name}: min={_min:.3f} max={_max:.3f} mean={_mean:.3f}  {_ok}")
            if _max > _hi * 1.5 or _min < _lo - _hi * 0.5:
                print(f"[V34]    ⚠️ dim[{_di}] 超出預期範圍 [{_lo}, {_hi}]，檢查 preprocess_margin.py")
        else:
            print(f"[V34] dim[{_di}] {_name}: 全部 missing/warmup")

    # BUG #11 稽查（2026-04-25）：memory layout 驗證
    # aligned 是 numpy row-major (C-contiguous)，cupy 預設也 row-major
    # kernel 用 (si * n_days + day) * 5 索引，要求 strides = (n_days*5*4, 5*4, 4) bytes
    assert aligned.flags["C_CONTIGUOUS"], "[V34] aligned 不是 C-contiguous！kernel 索引會錯"
    assert aligned.strides == (nd * 5 * 4, 5 * 4, 4), (
        f"[V34] aligned strides 異常 {aligned.strides}，預期 {(nd*5*4, 5*4, 4)}"
    )
    print(f"[V34] ✅ numpy memory layout: C-contiguous, strides={aligned.strides}")

    # 丟上 GPU，存為 module-level 讓 kernel wrapper 取用
    _v34_margin_gpu = cp.asarray(aligned)
    # cupy 應該繼承 numpy 的 layout，但 double check
    assert _v34_margin_gpu.flags["C_CONTIGUOUS"], "[V34] cupy tensor 不是 C-contiguous！kernel 索引會錯"
    assert _v34_margin_gpu.strides == (nd * 5 * 4, 5 * 4, 4), (
        f"[V34] cupy strides 異常 {_v34_margin_gpu.strides}"
    )
    print(f"[V34] ✅ cupy memory layout: C-contiguous, strides={_v34_margin_gpu.strides}")
    print(f"[V34] margin tensor on GPU: {_v34_margin_gpu.nbytes/1024/1024:.1f} MB")

    # BUG #2 修正：cpu_replay 需要 numpy 版（不能用 cupy，cpu_replay 是 Python 迴圈）
    pre["margin_tensor_np"] = aligned  # numpy, for cpu_replay
    pre["margin_tensor"] = _v34_margin_gpu  # cupy, for kernel 備查
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

    BUG #9 修正：加 args 結構檢查，base 改 kernel 簽名會立即炸出來
    """
    _EXPECTED_TAIL_LEN = 6  # d_params, n_params, d_results, BATCH, train_start, train_end

    def __init__(self, real_kernel, kernel_src):
        self.real = real_kernel
        self.code = kernel_src

    def __call__(self, grid, block, args):
        if _v34_margin_gpu is None:
            raise RuntimeError("[V34] margin tensor 未載入，請先跑 precompute()")
        args = tuple(args)
        # 基本結構檢查：最後 6 個 args 應該是 (d_params, n_params_int, d_results, BATCH_int, train_start_int, train_end_int)
        if len(args) < self._EXPECTED_TAIL_LEN + 30:  # base 有 40+ 個 indicator tensor args
            raise RuntimeError(
                f"[V34] kernel args 長度 {len(args)} 太短，base.CUDA_KERNEL 簽名可能已改。"
                f"檢查 base.main() 裡 CUDA_KERNEL((grid,), (BLOCK,), (...)) 呼叫。"
            )
        # 插入 margin_tensor 在 d_params 前面
        new_args = args[:-self._EXPECTED_TAIL_LEN] + (_v34_margin_gpu,) + args[-self._EXPECTED_TAIL_LEN:]
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
    print(f"[V34] 💡 SEED (89.90) 沒有 margin params → margin 9 slot 全部 default 0 = margin scoring 關閉")
    print(f"[V34] 💡 這代表 baseline 分數 = 89.90 純 OHLCV 表現，margin 只給「新策略」加分用")
    print(f"[V34] 啟動 base.main()...\n")
    base.main()
