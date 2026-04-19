"""2D 漸進搜尋：鎖定當前策略，每次改 2 個關鍵參數找組合最佳。
1D (coord_descent) 已證 89.90 收斂。2D 搜尋看能否找到跨 2 維的 local optimum 突破。

用法：
  python search_2d.py                    # 預設反向 WF
  $env:GPU_WF_MODE="forward"; python search_2d.py   # 正向 WF
"""
import sys, os, json, time, types, urllib.request
from itertools import combinations

mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

from gpu_cupy_evolve import PARAMS_SPACE, precompute, cpu_replay, download_data

# Top 8 關鍵參數（影響策略行為最大）
TOP_PARAMS = [
    "stop_loss",         # 6 options
    "take_profit",       # 8 options
    "trailing_stop",     # 8 options
    "breakeven_trigger", # 6 options
    "lock_trigger",      # 5 options
    "lock_floor",        # 6 options
    "hold_days",         # 7 options
    "buy_threshold",     # 6 options
]

GPU_GIST = "c1bef892d33589baef2142ce250d18c2"
TOKEN = os.environ.get("GH_TOKEN", "")
if not TOKEN:
    try:
        import subprocess
        TOKEN = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
    except: pass


def score_params(params, pre):
    """scoring 公式跟 coord_descent 一致（跟 kernel 相似但 Python 版）"""
    import math as _m
    trades = cpu_replay(pre, params)
    trades = [t for t in trades if not _m.isnan(t.get("return",0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    n = len(completed)
    if n < 40 or n > 200: return -1

    total = sum(t.get("return",0) for t in completed)
    avg = total / n
    if avg < 10: return -1

    # train/test 分類（支援正向/反向 WF）
    _tr_start_str = str(pre["dates"][pre["train_start"]].date())
    _tr_end_str = str(pre["dates"][pre["train_end"]-1].date()) if pre["train_end"] < pre["n_days"] else str(pre["dates"][-1].date())
    tr = [t for t in completed if _tr_start_str <= t.get("buy_date","") <= _tr_end_str]
    te = [t for t in completed if t.get("buy_date","") < _tr_start_str or t.get("buy_date","") > _tr_end_str]
    if not tr or len(te) < 5: return -1
    tr_tot = sum(t.get("return",0) for t in tr)
    te_tot = sum(t.get("return",0) for t in te)
    if te_tot <= 0: return -1

    tr_y = (pre["train_end"] - pre["train_start"]) / 250.0
    te_days = (pre["n_days"] - 60) - (pre["train_end"] - pre["train_start"])
    te_y = te_days / 250.0
    tr_ann = tr_tot / tr_y if tr_y > 0.5 else tr_tot
    te_ann = te_tot / te_y if te_y > 0.3 else te_tot
    if tr_ann < 326 or te_ann < 277: return -1
    if tr_ann > 0 and te_ann < tr_ann * 0.4: return -1

    tr_wr = sum(1 for t in tr if t.get("return",0) > 0) / len(tr) * 100
    te_wr = sum(1 for t in te if t.get("return",0) > 0) / len(te) * 100
    if tr_wr < 65 or te_wr < 60: return -1

    tr_avg = tr_tot / len(tr)
    mdd = 0; run = 0
    for t in tr:
        r = t.get("return",0)
        run = run + r if r < 0 else 0
        if run < mdd: mdd = run
    if mdd < -50: return -1

    s_wr = max(0, min(80, (tr_wr - 50) * 2.0))
    s_return = max(0, min(20, min(tr_ann, te_ann, 400) * 0.05))
    s_avg = max(0, min(30, (tr_avg - 15) * 2.0))
    recent_start_str = str(pre["dates"][max(pre["n_days"]-500, pre["train_start"])].date())
    rec = [t for t in tr if t.get("buy_date","") >= recent_start_str]
    s_recent = 0
    if len(rec) >= 5:
        rec_wr = sum(1 for t in rec if t.get("return",0) > 0) / len(rec) * 100
        s_recent = max(0, min(15, (rec_wr - 60) * 0.5))
    abs_dd = abs(mdd) if mdd < 0 else 1
    calmar = tr_ann / abs_dd if abs_dd > 1 else 0
    s_calmar = max(0, min(10, (calmar - 2) * 1.5)) if calmar > 2 else 0
    wf_ratio = min(1.2, te_ann / tr_ann) if tr_ann > 1 else 1
    s_wf = wf_ratio * 15

    return s_wr + s_return + s_avg + s_recent + s_calmar + s_wf


def main():
    # 載入 GPU Gist
    req = urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST}",
                                  headers={"Authorization": f"token {TOKEN}"} if TOKEN else {})
    gd = json.load(urllib.request.urlopen(req))
    gf = gd["files"]["best_strategy.json"]
    content = urllib.request.urlopen(gf["raw_url"]).read().decode() if gf.get("truncated") else gf["content"]
    strategy = json.loads(content)
    params = dict(strategy["params"])
    print(f"[起點] GPU Gist 策略 score={strategy.get('score')}")

    # 載入資料
    print("[資料] 下載 + precompute...")
    raw = download_data()
    _lens = [len(v) for v in raw.values()]
    TARGET_DAYS = 1500 if sum(1 for l in _lens if l >= 1500) >= 500 else 900
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    print(f"  {len(data)} 檔 × {TARGET_DAYS} 天")
    pre = precompute(data)

    # 基準分數
    print("[基準] cpu_replay + 評分...")
    base_score = score_params(params, pre)
    print(f"  基準 score = {base_score:.2f}\n")
    if base_score < 0:
        print("⚠️ 基準策略過不了 gate")
        return

    # 2D 掃描
    pairs = list(combinations(TOP_PARAMS, 2))
    total_tests = sum(len(PARAMS_SPACE[p1]) * len(PARAMS_SPACE[p2]) for p1, p2 in pairs)
    print(f"[2D 掃描] {len(pairs)} 個參數對，共 {total_tests} 次 cpu_replay，估計 30-60 分鐘\n")

    improvements = []
    pair_i = 0
    t0 = time.time()
    for p1, p2 in pairs:
        pair_i += 1
        opts1 = PARAMS_SPACE[p1]
        opts2 = PARAMS_SPACE[p2]
        pair_best_sc = base_score
        pair_best = (params.get(p1), params.get(p2))
        for v1 in opts1:
            for v2 in opts2:
                if v1 == params.get(p1) and v2 == params.get(p2): continue
                tp = dict(params)
                tp[p1] = v1; tp[p2] = v2
                sc = score_params(tp, pre)
                if sc > pair_best_sc:
                    pair_best_sc = sc
                    pair_best = (v1, v2)
        elapsed = time.time() - t0
        marker = "⭐" if pair_best_sc > base_score else "  "
        if pair_best_sc > base_score:
            improvements.append((pair_best_sc, p1, p2, pair_best[0], pair_best[1]))
            print(f"  {marker} [{pair_i:2d}/{len(pairs)}] {p1:20s} × {p2:20s}  → {pair_best[0]},{pair_best[1]}  score {base_score:.2f} → {pair_best_sc:.2f}  ({elapsed/60:.1f}分)")
        else:
            print(f"  {marker} [{pair_i:2d}/{len(pairs)}] {p1:20s} × {p2:20s}  無改進  ({elapsed/60:.1f}分)")

    # 結果
    print(f"\n{'='*70}\n  2D 掃描結果\n{'='*70}")
    if not improvements:
        print("  ❌ 2D 無任何改進，89.90 也是 2D local optimum")
        print("  建議：接受 89.90 為最終策略，或嘗試 3D（時間 > 10 小時）")
        return
    improvements.sort(reverse=True)
    print(f"  找到 {len(improvements)} 個 2D 改進（依分數排序）:\n")
    for sc, p1, p2, v1, v2 in improvements[:10]:
        print(f"    {p1}={v1}, {p2}={v2}  → score {sc:.2f}")

    # 應用最佳
    best_sc, p1, p2, v1, v2 = improvements[0]
    params[p1] = v1
    params[p2] = v2
    print(f"\n  應用最佳：{p1}={v1}, {p2}={v2}  final score {best_sc:.2f}")

    # 存 pending
    trades = cpu_replay(pre, params)
    import math as _m
    completed = [t for t in trades if not _m.isnan(t.get("return",0)) and t.get("reason") != "持有中"]
    pending = {
        "score": round(best_sc, 4),
        "source": "search_2d",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backtest": {
            "total_trades": len(completed),
            "total_return": round(sum(t.get("return",0) for t in completed), 2),
            "avg_return": round(sum(t.get("return",0) for t in completed) / len(completed), 2) if completed else 0,
            "win_rate": round(sum(1 for t in completed if t.get("return",0)>0) / len(completed) * 100, 1) if completed else 0,
        },
        "params": params,
        "trade_details": trades,
    }
    out = "pending_push_2d.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    print(f"\n存到 {out}")
    print(f"若推：mv {out} pending_push.json && python push_pending.py")


if __name__ == "__main__":
    main()
