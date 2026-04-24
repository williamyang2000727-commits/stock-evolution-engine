"""
V35 Regime Commander sanity test — 89.90 在 BULL/BEAR/CHOP 下表現真的差異顯著嗎？
用法：C:\\stock-evolution> python sanity_test_regime.py

邏輯：
  1. 讀 Windows cache + GPU Gist 當前策略 params（V35 要打的是「當前 SEED」，不一定是 89.90）
  2. 跑 cpu_replay 產生所有完成交易
  3. 對每筆交易按「買入日所在 regime」分組（BULL / BEAR / CHOP）
  4. 算每組的 勝率 / avg return / 筆數

Go/no-go 裁決：
  🟢 BEAR 勝率比整體低 ≥ 5% AND BEAR 筆數 ≥ 10  →  V35 有攻擊面，值得燒 24h
  🟡 BEAR 勝率比整體低 3-5%                      →  邊際值得試
  🔴 三個 regime 差 < 3%                         →  SEED 其實已 regime-aware，V35 白搭，pivot

這是 V31 教訓的補救（沒 sanity test 就上 GPU = 24h 白燒）
對應記憶：v34_margin_gambit_2026_04_25.md 第 340 行
"""
import os, sys, json, pickle
import urllib.request
import numpy as np

# 把 stock-evolution 加進 sys.path，import base gpu_cupy_evolve
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if os.path.isdir(_USER_SE) and _USER_SE not in sys.path:
    sys.path.insert(0, _USER_SE)

import gpu_cupy_evolve as base

GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
GIST_URL = f"https://api.github.com/gists/{GPU_GIST_ID}"

# Regime 偵測參數（對齊 V35 precompute 的 default）
MA20_LEN = 20
MA60_LEN = 60


def compute_regime(market_close: np.ndarray, ma_short: int, ma_long: int) -> np.ndarray:
    """同 V35 _compute_regime_array，float32 array {0,1,2}"""
    n = len(market_close)
    ms = np.zeros(n, dtype=np.float32)
    ml = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i >= ma_short:
            ms[i] = market_close[max(0, i - ma_short + 1):i + 1].mean()
        if i >= ma_long:
            ml[i] = market_close[max(0, i - ma_long + 1):i + 1].mean()
    reg = np.full(n, 2.0, dtype=np.float32)
    for i in range(max(ma_short, ma_long), n):
        if ms[i] > ml[i] and market_close[i] > ms[i]:
            reg[i] = 0.0  # BULL
        elif ms[i] < ml[i] and market_close[i] < ms[i]:
            reg[i] = 1.0  # BEAR
    return reg


def fetch_gist_strategy():
    """讀 GPU Gist 當前策略 params"""
    req = urllib.request.Request(GIST_URL)
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    content = d["files"]["best_strategy.json"]["content"]
    strategy = json.loads(content)
    params = strategy.get("params", strategy)  # 容錯：可能直接在頂層
    score = strategy.get("score", "N/A")
    source = strategy.get("source", "unknown")
    return params, score, source


def main():
    print("=== V35 Regime Commander sanity test ===")
    print(f"問題：SEED（當前 Gist 策略）在 BULL/BEAR/CHOP 下表現差異夠大嗎？")
    print(f"標準：BEAR 勝率比整體低 ≥ 5% → V35 有攻擊面 → 值得燒 24h GPU\n")

    # Step 1: 讀 Gist 策略
    print("[1/4] 讀 GPU Gist 當前策略...")
    params, score, source = fetch_gist_strategy()
    print(f"  score = {score}")
    print(f"  source = {source}")
    print(f"  params 欄位數 = {len(params)}")

    # Step 2: precompute（mirror base.main() 的過濾邏輯）
    print("\n[2/4] 載入 cache + 過濾...")
    cache_path = os.path.join(_USER_SE, "stock_data_cache.pkl")
    raw = pickle.load(open(cache_path, "rb"))
    print(f"  raw cache: {len(raw)} 檔")

    # 動態選擇 TARGET_DAYS（複製 base.main() line 1696-1708 邏輯）
    _lens = [len(v) for v in raw.values()]
    _n_1500 = sum(1 for l in _lens if l >= 1500)
    _n_1200 = sum(1 for l in _lens if l >= 1200)
    _n_900 = sum(1 for l in _lens if l >= 900)
    if _n_1500 >= 500:
        TARGET_DAYS = 1500
    elif _n_1200 >= 800:
        TARGET_DAYS = 1200
    else:
        TARGET_DAYS = 900
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    print(f"  過濾後: {len(data)} 檔 × {TARGET_DAYS} 天（1500-q {_n_1500} / 1200-q {_n_1200} / 900-q {_n_900}）")

    print("\n  base.precompute 執行中...")
    pre = base.precompute(data)
    print(f"  precompute done: {pre['n_stocks']} 檔 × {pre['n_days']} 天")

    # Step 3: 算 regime array（用跟 V35 一致的邏輯）
    print(f"\n[3/4] 算 regime array（ma20={MA20_LEN}, ma60={MA60_LEN}）...")
    market_close = pre["close"].mean(axis=0)
    regime = compute_regime(market_close, MA20_LEN, MA60_LEN)
    n = pre["n_days"]
    bull_n = int((regime == 0).sum())
    bear_n = int((regime == 1).sum())
    chop_n = int((regime == 2).sum())
    print(f"  🟢 BULL: {bull_n} 天 ({bull_n/n*100:.1f}%)")
    print(f"  🔴 BEAR: {bear_n} 天 ({bear_n/n*100:.1f}%)")
    print(f"  🟡 CHOP: {chop_n} 天 ({chop_n/n*100:.1f}%)")

    # 建 date string → day index map
    date_to_day = {str(d.date()): i for i, d in enumerate(pre["dates"])}

    # Step 4: cpu_replay 跑 SEED 策略
    print(f"\n[4/4] cpu_replay 跑當前 SEED...")
    trades = base.cpu_replay(pre, params)
    print(f"  產出 {len(trades)} 筆 trades（含持有中）")

    # Step 5: 按 regime 分組
    # 排除「持有中」交易（sell_date 空字串或 reason=持有中）
    completed = []
    for t in trades:
        sd = t.get("sell_date", "")
        reason = t.get("reason", "")
        if sd and reason != "持有中":
            completed.append(t)
    print(f"  完成交易 {len(completed)} 筆")

    regime_names = ["BULL", "BEAR", "CHOP"]
    buckets = {0: [], 1: [], 2: []}  # regime index → list of trade returns
    for t in completed:
        bd_str = t.get("buy_date", "")
        if bd_str not in date_to_day:
            continue
        day = date_to_day[bd_str]
        r = int(regime[day])
        ret = float(t.get("return", 0))
        buckets[r].append(ret)

    # Step 6: 算各組指標
    all_rets = [float(t.get("return", 0)) for t in completed]
    overall_wr = (np.array(all_rets) > 0).mean() * 100 if all_rets else 0.0
    overall_avg = np.mean(all_rets) if all_rets else 0.0
    overall_total = sum(all_rets)

    print(f"\n=== 結果（SEED 在各 regime 下的表現）===")
    print(f"{'regime':<8s}{'筆數':>6s}{'勝率':>9s}{'avg':>9s}{'total':>11s}{'vs整體':>10s}")
    print(f"整體     {len(all_rets):>6d}{overall_wr:>8.1f}%{overall_avg:>8.1f}%{overall_total:>10.1f}%{'—':>10s}")
    print("-" * 55)

    bear_wr = None
    bull_wr = None
    chop_wr = None
    for r in [0, 1, 2]:
        arr = np.array(buckets[r])
        name = regime_names[r]
        if len(arr) == 0:
            print(f"{name:<8s}{0:>6d}{'N/A':>9s}{'N/A':>9s}{'N/A':>11s}{'N/A':>10s}")
            continue
        wr = (arr > 0).mean() * 100
        avg = arr.mean()
        total = arr.sum()
        diff = wr - overall_wr
        diff_str = f"{diff:+.1f}%"
        print(f"{name:<8s}{len(arr):>6d}{wr:>8.1f}%{avg:>8.1f}%{total:>10.1f}%{diff_str:>10s}")
        if r == 0: bull_wr = wr
        elif r == 1: bear_wr = wr
        else: chop_wr = wr

    # Step 7: Go/no-go 裁決
    print(f"\n=== V35 Go/no-go 裁決 ===")
    if bear_wr is None or len(buckets[1]) < 10:
        print(f"🟡 BEAR 樣本不足（{len(buckets[1])} 筆），無法判斷")
        print(f"   建議：(a) 擴大 cache 期間，或 (b) 調整 ma60 定義讓 BEAR 期更長")
        return

    bear_gap = overall_wr - bear_wr
    bull_gap = (bull_wr - overall_wr) if bull_wr is not None else 0.0

    print(f"BEAR 勝率比整體低 {bear_gap:+.1f}%")
    if bull_wr is not None:
        print(f"BULL 勝率比整體高 {bull_gap:+.1f}%")

    if bear_gap >= 5.0 and len(buckets[1]) >= 10:
        print(f"\n🟢 有攻擊面！BEAR 勝率 {bear_wr:.1f}% 明顯低於整體 {overall_wr:.1f}%")
        print(f"   V35 用 bear_buy_th_delta=+2 擋掉 BEAR 期低品質訊號 = 預期可拉勝率")
        print(f"   → 值得燒 24h GPU 跑 V35")
    elif bear_gap >= 3.0:
        print(f"\n🟡 邊際值得試（BEAR 差 {bear_gap:.1f}%）")
        print(f"   V35 可能有 +2-3% 勝率提升，但風險是 GPU 選不到有效 delta")
        print(f"   → 可選擇跑 V35 或直接 pivot 到 ensemble")
    else:
        print(f"\n🔴 BEAR 差距只 {bear_gap:.1f}% < 3% 門檻")
        print(f"   SEED 在各 regime 下表現已接近一致 → 代表指標（ADX / week52 / BB）已隱含 regime")
        print(f"   V35 硬加 regime gate 會被 GPU 學成 delta=0（= 等於不啟用）")
        print(f"   → **不要**燒 24h GPU，改做 ensemble 89.90 + 103（絕招 6）")


if __name__ == "__main__":
    main()
