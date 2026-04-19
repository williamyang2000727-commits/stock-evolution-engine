"""漸進維度搜尋：1D → 2D → 3D → 4D ... 一路試。

Coord Descent 證 89.90 是 1D local optimum，這工具擴展到高維。
每找到 improvement 立即 apply + 存 pending，所以 ctrl+C 隨時停都保留進度。

用法：
  python progressive_search.py                   # 預設跑到 4D
  python progressive_search.py --max-dim 8       # 跑到 8D (時間爆炸)
  python progressive_search.py --top N           # 選 top N 參數 (預設 8)
  $env:GPU_WF_MODE="forward"; python progressive_search.py   # 正向 WF

時間估算（top 8 參數）：
  1D: ~ 2 分（50 次 cpu_replay）
  2D: ~ 1 小時（1,176 次）
  3D: ~ 18 小時（30,000 次）
  4D: ~ 10 天（378,000 次）  ← 跑完的話
  5D+: 不切實際（月以上）

隨時 ctrl+C，pending_push_progressive.json 保留當前最佳。
"""
import sys, os, json, time, types, urllib.request, signal
from itertools import combinations, product

mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp

from gpu_cupy_evolve import PARAMS_SPACE, precompute, cpu_replay, download_data

# Top 參數池（前 8 個影響最大，可加更多但時間爆炸）
ALL_TOP_PARAMS = [
    "stop_loss",         # 6 options
    "take_profit",       # 8 options
    "trailing_stop",     # 8 options
    "breakeven_trigger", # 6 options
    "lock_trigger",      # 5 options
    "lock_floor",        # 6 options
    "hold_days",         # 7 options
    "buy_threshold",     # 6 options
    "w_rsi", "rsi_th",   # 額外買入類（9, 10）
    "w_bb", "bb_th",     # 11, 12
    "w_mom", "mom_th",   # 13, 14
]

GPU_GIST = "c1bef892d33589baef2142ce250d18c2"
TOKEN = os.environ.get("GH_TOKEN", "")
if not TOKEN:
    try:
        import subprocess
        TOKEN = subprocess.check_output(["gh", "auth", "token"], timeout=5).decode().strip()
    except: pass

# 全局變數讓 signal handler 能存檔
_current_best = {"params": None, "score": -1, "info": None, "pre": None}
_interrupted = False


def score_params(params, pre):
    import math as _m
    trades = cpu_replay(pre, params)
    trades = [t for t in trades if not _m.isnan(t.get("return",0))]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    n = len(completed)
    if n < 40 or n > 200: return -1, None
    total = sum(t.get("return",0) for t in completed)
    avg = total / n
    if avg < 10: return -1, None
    _tr_start_str = str(pre["dates"][pre["train_start"]].date())
    _tr_end_str = str(pre["dates"][pre["train_end"]-1].date()) if pre["train_end"] < pre["n_days"] else str(pre["dates"][-1].date())
    tr = [t for t in completed if _tr_start_str <= t.get("buy_date","") <= _tr_end_str]
    te = [t for t in completed if t.get("buy_date","") < _tr_start_str or t.get("buy_date","") > _tr_end_str]
    if not tr or len(te) < 5: return -1, None
    tr_tot = sum(t.get("return",0) for t in tr)
    te_tot = sum(t.get("return",0) for t in te)
    if te_tot <= 0: return -1, None
    tr_y = (pre["train_end"] - pre["train_start"]) / 250.0
    te_days = (pre["n_days"] - 60) - (pre["train_end"] - pre["train_start"])
    te_y = te_days / 250.0
    tr_ann = tr_tot / tr_y if tr_y > 0.5 else tr_tot
    te_ann = te_tot / te_y if te_y > 0.3 else te_tot
    if tr_ann < 326 or te_ann < 277: return -1, None
    if tr_ann > 0 and te_ann < tr_ann * 0.4: return -1, None
    tr_wr = sum(1 for t in tr if t.get("return",0) > 0) / len(tr) * 100
    te_wr = sum(1 for t in te if t.get("return",0) > 0) / len(te) * 100
    if tr_wr < 65 or te_wr < 60: return -1, None
    tr_avg = tr_tot / len(tr)
    mdd = 0; run = 0
    for t in tr:
        r = t.get("return",0); run = run + r if r < 0 else 0
        if run < mdd: mdd = run
    if mdd < -50: return -1, None

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
    score = s_wr + s_return + s_avg + s_recent + s_calmar + s_wf
    info = {"n": n, "avg": round(avg,2), "wr": round(sum(1 for t in completed if t.get("return",0)>0)/n*100,1),
            "total": round(total,1), "tr_ann": round(tr_ann,1), "te_ann": round(te_ann,1),
            "calmar": round(calmar,2), "tr_wr": round(tr_wr,1), "te_wr": round(te_wr,1)}
    return score, info


def save_pending():
    """把當前 best 存到 pending_push_progressive.json（供 ctrl+C 後保留）"""
    if _current_best["params"] is None or _current_best["pre"] is None: return
    import math as _m
    trades = cpu_replay(_current_best["pre"], _current_best["params"])
    completed = [t for t in trades if not _m.isnan(t.get("return",0)) and t.get("reason") != "持有中"]
    pending = {
        "score": round(_current_best["score"], 4),
        "source": "progressive_search",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backtest": {
            "total_trades": len(completed),
            "total_return": round(sum(t.get("return",0) for t in completed), 2),
            "avg_return": round(sum(t.get("return",0) for t in completed) / len(completed), 2) if completed else 0,
            "win_rate": round(sum(1 for t in completed if t.get("return",0)>0) / len(completed) * 100, 1) if completed else 0,
        },
        "params": _current_best["params"],
        "trade_details": trades,
    }
    with open("pending_push_progressive.json", "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


def signal_handler(sig, frame):
    global _interrupted
    print("\n\n⚠️ 收到 ctrl+C，存檔後退出...")
    save_pending()
    print(f"已存 pending_push_progressive.json (score={_current_best['score']:.2f})")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def search_dim(params, pre, base_score, dim, top_params):
    """搜尋 dim 維度的所有參數組合"""
    combos = list(combinations(top_params, dim))
    total_tests = sum(
        1 for combo in combos
        for _ in product(*[PARAMS_SPACE[p] for p in combo])
    )
    print(f"\n{'='*70}")
    print(f"  {dim}D 搜尋：{len(combos)} 個 {dim}-元組，共 {total_tests:,} 次 cpu_replay")
    print(f"{'='*70}")

    tested = 0
    t0 = time.time()
    found_improvement = None
    for combo in combos:
        opts_list = [PARAMS_SPACE[p] for p in combo]
        for values in product(*opts_list):
            # 跳過全是當前值
            if all(values[i] == params.get(combo[i]) for i in range(dim)):
                continue
            tp = dict(params)
            for i, p in enumerate(combo):
                tp[p] = values[i]
            sc, info = score_params(tp, pre)
            tested += 1

            if sc > base_score:
                elapsed = (time.time() - t0) / 60
                vstr = ", ".join(f"{combo[i]}={values[i]}" for i in range(dim))
                print(f"  ⭐ [{tested:,}/{total_tests:,}] {vstr}  → score {base_score:.2f} → {sc:.2f}  ({elapsed:.1f}分)")
                # 立即 apply + 存檔
                for i, p in enumerate(combo):
                    params[p] = values[i]
                base_score = sc
                _current_best["params"] = dict(params)
                _current_best["score"] = sc
                _current_best["info"] = info
                save_pending()
                found_improvement = (combo, values, sc, info)
                # 繼續搜尋（base_score 已更新，可能找更好的）
            elif tested % 500 == 0:
                elapsed = (time.time() - t0) / 60
                rate = tested / (time.time() - t0) if time.time() - t0 > 0 else 0
                eta = (total_tests - tested) / rate / 60 if rate > 0 else 0
                print(f"     [{tested:,}/{total_tests:,}] {elapsed:.1f}分 速度 {rate:.1f}/s ETA {eta:.0f}分")

    elapsed = (time.time() - t0) / 60
    print(f"  {dim}D 完成：{tested:,} 次測試 ({elapsed:.1f}分)")
    if found_improvement:
        print(f"  ✅ {dim}D 找到改進，最終 score {base_score:.2f}")
    else:
        print(f"  ❌ {dim}D 無任何改進")
    return found_improvement is not None, params, base_score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dim", type=int, default=4, help="最大搜尋維度 (預設 4)")
    parser.add_argument("--top", type=int, default=8, help="top N 參數池 (預設 8)")
    args = parser.parse_args()

    top_params = ALL_TOP_PARAMS[:args.top]
    print(f"[設定] max_dim={args.max_dim} | top params={args.top}")
    print(f"       {top_params}")

    # 載入 Gist
    req = urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST}",
                                  headers={"Authorization": f"token {TOKEN}"} if TOKEN else {})
    gd = json.load(urllib.request.urlopen(req))
    gf = gd["files"]["best_strategy.json"]
    content = urllib.request.urlopen(gf["raw_url"]).read().decode() if gf.get("truncated") else gf["content"]
    strategy = json.loads(content)
    params = dict(strategy["params"])
    print(f"[起點] GPU Gist score={strategy.get('score')}")

    # 資料
    print("[資料] 下載 + precompute...")
    raw = download_data()
    _lens = [len(v) for v in raw.values()]
    TARGET_DAYS = 1500 if sum(1 for l in _lens if l >= 1500) >= 500 else 900
    data = {k: v.tail(TARGET_DAYS) for k, v in raw.items() if len(v) >= TARGET_DAYS}
    print(f"  {len(data)} 檔 × {TARGET_DAYS} 天")
    pre = precompute(data)

    # 基準分數
    base_score, base_info = score_params(params, pre)
    print(f"\n[基準] score = {base_score:.2f}")
    print(f"       {base_info}")
    if base_score < 0:
        print("⚠️ 起點過不了 gate")
        return

    _current_best["params"] = dict(params)
    _current_best["score"] = base_score
    _current_best["info"] = base_info
    _current_best["pre"] = pre
    save_pending()  # 初始存

    # 漸進維度
    total_improvements = 0
    for dim in range(1, args.max_dim + 1):
        improved, params, base_score = search_dim(params, pre, base_score, dim, top_params)
        if improved:
            total_improvements += 1

    # 最終
    print(f"\n{'='*70}\n  最終結果\n{'='*70}")
    final_score, final_info = score_params(params, pre)
    print(f"  初始 score: {_current_best.get('_init_score', '?')}")
    print(f"  最終 score: {final_score:.2f}")
    print(f"  {final_info}")
    print(f"  改進次數: {total_improvements}")
    print(f"\n存到 pending_push_progressive.json")
    print(f"若要推：mv pending_push_progressive.json pending_push.json && python push_pending.py")


if __name__ == "__main__":
    main()
