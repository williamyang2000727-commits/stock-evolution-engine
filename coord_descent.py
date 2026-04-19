"""Coordinate Descent：鎖定 89.90 參數，逐個參數找最佳值，迭代找局部最佳。
GPU 隨機爬山可能跳過「89.90 附近精細微調區」，這個工具系統性掃描該區。

用法：
  python coord_descent.py              # 從當前 GPU Gist 策略開始
  python coord_descent.py --iter 2     # 迭代 2 輪（每輪掃所有 54 參數）
"""
import sys, os, json, time, types, urllib.request, pickle

# Mock CuPy
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

from gpu_cupy_evolve import (
    PARAM_ORDER, PARAMS_SPACE, MA_FAST_OPTS, MA_SLOW_OPTS, MOM_DAYS_OPTS,
    precompute, cpu_replay, download_data
)

GPU_GIST = "c1bef892d33589baef2142ce250d18c2"
TOKEN = os.environ.get("GH_TOKEN", "")
if not TOKEN:
    try:
        import subprocess
        TOKEN = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
    except: pass


def score_params(params, pre):
    """用新 scoring 公式評分（跟 kernel 一致）"""
    import math as _m
    trades = cpu_replay(pre, params)
    trades = [t for t in trades if not _m.isnan(t.get("return",0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    n = len(completed)
    if n < 40 or n > 200: return -1, None

    total = sum(t.get("return",0) for t in completed)
    avg = total / n
    if avg < 10: return -1, None

    # 分 train/test（反向 WF）
    _tsd = pre["dates"][pre["train_start"]]
    _ts_str = str(_tsd.date())
    tr = [t for t in completed if t.get("buy_date","") >= _ts_str]
    te = [t for t in completed if t.get("buy_date","") < _ts_str]
    if not tr or len(te) < 5: return -1, None
    tr_tot = sum(t.get("return",0) for t in tr)
    te_tot = sum(t.get("return",0) for t in te)
    if te_tot <= 0: return -1, None

    tr_y = (pre["train_end"] - pre["train_start"]) / 250.0
    te_y = (pre["train_start"] - 60) / 250.0
    tr_ann = tr_tot / tr_y if tr_y > 0.5 else tr_tot
    te_ann = te_tot / te_y if te_y > 0.3 else te_tot
    if tr_ann < 326 or te_ann < 277: return -1, None
    if tr_ann > 0 and te_ann < tr_ann * 0.4: return -1, None

    tr_wr = sum(1 for t in tr if t.get("return",0) > 0) / len(tr) * 100
    te_wr = sum(1 for t in te if t.get("return",0) > 0) / len(te) * 100
    if tr_wr < 65 or te_wr < 60: return -1, None

    tr_avg = tr_tot / len(tr)
    # MaxDD train
    mdd = 0; run = 0
    for t in tr:
        r = t.get("return",0)
        run = run + r if r < 0 else 0
        if run < mdd: mdd = run
    if mdd < -50: return -1, None

    # Scoring
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

    wf_ratio = te_ann / tr_ann if tr_ann > 1 else 1
    wf_ratio = min(1.2, wf_ratio)
    s_wf = wf_ratio * 15

    score = s_wr + s_return + s_avg + s_recent + s_calmar + s_wf
    info = {"n": n, "avg": avg, "wr": sum(1 for t in completed if t.get("return",0)>0)/n*100,
            "total": total, "tr_ann": tr_ann, "te_ann": te_ann, "calmar": calmar}
    return score, info


def main():
    iters = 1
    if "--iter" in sys.argv:
        iters = int(sys.argv[sys.argv.index("--iter") + 1])

    # 載入當前 GPU Gist 策略
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
    # 動態 TARGET_DAYS
    _lens = [len(v) for v in raw.values()]
    TARGET_DAYS = 1500 if sum(1 for l in _lens if l >= 1500) >= 500 else 900
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    print(f"  {len(data)} 檔 × {TARGET_DAYS} 天")
    pre = precompute(data)

    # 算起點分數
    print("[基準] 跑 cpu_replay 算起點 scoring...")
    base_score, base_info = score_params(params, pre)
    print(f"  基準 score = {base_score:.2f}")
    if base_info: print(f"  {base_info}")
    if base_score < 0:
        print("⚠️ 基準策略過不了 gate，無法 coord descent")
        return

    # 待優化的參數（54 個）
    opt_params = [k for k in PARAM_ORDER if len(PARAMS_SPACE.get(k, [])) > 1]
    print(f"\n[掃描] {len(opt_params)} 個參數，每輪試每個的所有 PARAMS_SPACE 選項")

    for it in range(iters):
        print(f"\n{'='*60}\n  迭代 {it+1} / {iters}\n{'='*60}")
        improved = False
        for pi, pk in enumerate(opt_params):
            opts = PARAMS_SPACE[pk]
            current = params.get(pk)
            best_opt = current
            best_sc = base_score
            n_better = 0
            for opt in opts:
                if opt == current: continue
                tp = dict(params)
                tp[pk] = opt
                sc, info = score_params(tp, pre)
                if sc > best_sc:
                    best_sc = sc
                    best_opt = opt
                    n_better += 1
            marker = "⭐" if best_opt != current else "  "
            print(f"  {marker} [{pi+1:2d}/{len(opt_params)}] {pk:25s}: {current} → {best_opt}  (score {base_score:.2f} → {best_sc:.2f}, {n_better} 選項更優)")
            if best_opt != current:
                params[pk] = best_opt
                base_score = best_sc
                improved = True

        # MA/MOM 也要掃
        for mkey, opts in [("ma_fast_w", MA_FAST_OPTS), ("ma_slow_w", MA_SLOW_OPTS), ("momentum_days", MOM_DAYS_OPTS)]:
            current = params.get(mkey)
            best_opt = current
            best_sc = base_score
            for opt in opts:
                if opt == current: continue
                tp = dict(params); tp[mkey] = opt
                # ma_fast < ma_slow 檢查
                if mkey == "ma_fast_w" and opt >= params.get("ma_slow_w", 15): continue
                if mkey == "ma_slow_w" and opt <= params.get("ma_fast_w", 3): continue
                sc, _ = score_params(tp, pre)
                if sc > best_sc:
                    best_sc = sc
                    best_opt = opt
            if best_opt != current:
                print(f"  ⭐ {mkey}: {current} → {best_opt}  (score → {best_sc:.2f})")
                params[mkey] = best_opt
                base_score = best_sc
                improved = True

        if not improved:
            print("\n  ✅ 已收斂（此輪無任何改進）")
            break

    # 最終結果
    print(f"\n{'='*60}\n  結果\n{'='*60}")
    final_score, final_info = score_params(params, pre)
    print(f"  最終 score = {final_score:.2f}")
    if final_info: print(f"  {final_info}")

    # 存成 pending 格式
    trades = cpu_replay(pre, params)
    import math as _m
    completed = [t for t in trades if not _m.isnan(t.get("return",0)) and t.get("reason") != "持有中"]
    holding = [t for t in trades if not _m.isnan(t.get("return",0)) and t.get("reason") == "持有中"]

    pending = {
        "score": round(final_score, 4),
        "source": "coord_descent",
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
    out = "pending_push_coord.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    print(f"\n存到 {out}")
    print(f"若要推上線：mv {out} pending_push.json && python push_pending.py")


if __name__ == "__main__":
    main()
