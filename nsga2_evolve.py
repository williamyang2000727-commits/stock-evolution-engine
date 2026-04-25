"""
V42 NSGA-II Multi-Objective Evolution — 5090 預備版（3060 也能跑）
用法：C:\\stock-evolution> python nsga2_evolve.py [--mode 3060|5090]

V1-V41 全敗共因：
  全部在「找單一 max score」框架內 → 89.90 是該框架 TRUE GLOBAL OPTIMUM
  即使 fine-tune 也只是更精準的 filter，沒解決結構問題

V42 NSGA-II 完全不同：
  不再找「單一最強策略」
  改找「Pareto frontier 多策略候選」
  4 個目標同時優化（互相 trade-off）：
    1. total return    (越大越好)
    2. win rate        (越大越好)
    3. -MaxDD          (越小越好，加負號變最大化)
    4. n_trades        (避免一年只買 2 筆)

跟 89.90 的差異：
  89.90: scalar score = s_wr*2 + s_avg*2 + s_return*0.05 + ... → 找 max
  V42:   找「沒有別的點同時在 4 個目標都贏」的 Pareto 點

預期突破：
  可能找到「total 1700%（比 89.90 略低 17%）+ MaxDD -10%（比 89.90 -19% 大幅改善）」這種 89.90 框架找不到的點

3060 vs 5090 設定：
  3060:  pop=64,  gen=30,  ~30-60min
  5090:  pop=256, gen=100, ~2-4 hours

學術依據：
  Deb et al. 2002 NSGA-II（25000+ citations）
  López de Prado 2018 AFML 提到 multi-objective optimization for trading
"""
import os, sys, json, pickle, time, argparse, random
import numpy as np

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def get_config(mode):
    if mode == "5090":
        return {"pop_size": 256, "n_gen": 100, "label": "5090 全力 (256 pop × 100 gen)"}
    elif mode == "3060":
        return {"pop_size": 64, "n_gen": 30, "label": "3060 縮版 (64 pop × 30 gen)"}
    else:
        raise ValueError(mode)


def random_individual():
    """從 PARAMS_SPACE 隨機抽一個 individual"""
    ind = {}
    for k, opts in base.PARAMS_SPACE.items():
        ind[k] = random.choice(opts)
    ind["ma_fast_w"] = random.choice(base.MA_FAST_OPTS)
    ind["ma_slow_w"] = random.choice(base.MA_SLOW_OPTS)
    ind["momentum_days"] = random.choice(base.MOM_DAYS_OPTS)
    return ind


def crossover(p1, p2):
    """uniform crossover"""
    child = {}
    for k in p1:
        child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return child


def mutate(ind, rate=0.15):
    for k in list(ind.keys()):
        if random.random() < rate:
            if k in base.PARAMS_SPACE:
                ind[k] = random.choice(base.PARAMS_SPACE[k])
            elif k == "ma_fast_w":
                ind[k] = random.choice(base.MA_FAST_OPTS)
            elif k == "ma_slow_w":
                ind[k] = random.choice(base.MA_SLOW_OPTS)
            elif k == "momentum_days":
                ind[k] = random.choice(base.MOM_DAYS_OPTS)
    return ind


def evaluate_with_kernel(individuals, pre, batch_size_max=4096):
    """
    用 base GPU kernel 一次評估整個 population
    回傳每個 ind 的 [score, n_trades, avg, total, wr]
    """
    import cupy as cp

    n_ind = len(individuals)
    PARAM_ORDER = base.PARAM_ORDER
    N_PARAMS = len(PARAM_ORDER)
    N_PARAMS_FULL = N_PARAMS + 3  # +ma_fast/slow/mom_days

    # build params_np
    params_np = np.zeros((n_ind, N_PARAMS_FULL), dtype=np.float32)
    for i, ind in enumerate(individuals):
        for j, key in enumerate(PARAM_ORDER):
            params_np[i, j] = float(ind.get(key, 0))
        params_np[i, N_PARAMS] = base.MA_FAST_MAP.get(int(ind.get("ma_fast_w", 5)), 1)
        params_np[i, N_PARAMS + 1] = base.MA_SLOW_MAP.get(int(ind.get("ma_slow_w", 20)), 1)
        params_np[i, N_PARAMS + 2] = base.MOM_MAP.get(int(ind.get("momentum_days", 5)), 1)

    ns = pre["n_stocks"]
    nd = pre["n_days"]

    d_close = cp.asarray(pre["close"])
    d_rsi = cp.asarray(pre["rsi"])
    d_bb = cp.asarray(pre["bb_pos"])
    d_vr = cp.asarray(pre["vol_ratio"])
    d_mh = cp.asarray(pre["macd_hist"])
    d_ml = cp.asarray(pre["macd_line"])
    d_kv = cp.asarray(pre["k_val"])
    d_dv = cp.asarray(pre["d_val"])
    d_mom3 = cp.asarray(pre["mom_d"][3])
    d_mom5 = cp.asarray(pre["mom_d"][5])
    d_mom10 = cp.asarray(pre["mom_d"][10])
    d_ig = cp.asarray(pre["is_green"])
    d_gp = cp.asarray(pre["gap"])
    d_nh = cp.asarray(pre["near_high"])
    d_wr = cp.asarray(pre["williams_r"])
    d_ma3 = cp.asarray(pre["ma_d"][3])
    d_ma5 = cp.asarray(pre["ma_d"][5])
    d_ma10 = cp.asarray(pre["ma_d"][10])
    d_ma15 = cp.asarray(pre["ma_d"][15])
    d_ma20 = cp.asarray(pre["ma_d"][20])
    d_ma30 = cp.asarray(pre["ma_d"][30])
    d_ma60 = cp.asarray(pre["ma60"])
    d_vp = cp.asarray(pre["vol_prev"])
    d_squeeze = cp.asarray(pre["squeeze_fire"])
    d_newhigh = cp.asarray(pre["new_high_60"])
    d_adx = cp.asarray(pre["adx"])
    d_bias = cp.asarray(pre["bias"])
    d_obv_rising = cp.asarray(pre["obv_rising"])
    d_atr_pct = cp.asarray(pre["atr_pct"])
    d_open_arr = cp.asarray(pre["open"])
    d_top100 = cp.asarray(pre["top100_mask"]) if pre.get("top100_mask") is not None else cp.zeros((ns, nd), dtype=cp.float32)
    d_market = cp.asarray(pre["market_bull"]) if pre.get("market_bull") is not None else cp.ones(nd, dtype=cp.float32)
    d_up_days = cp.asarray(pre["up_days"]) if pre.get("up_days") is not None else cp.zeros((ns, nd), dtype=cp.float32)
    d_week52 = cp.asarray(pre["week52_pos"]) if pre.get("week52_pos") is not None else cp.zeros((ns, nd), dtype=cp.float32)
    d_vol_up_days = cp.asarray(pre["vol_up_days"]) if pre.get("vol_up_days") is not None else cp.zeros((ns, nd), dtype=cp.float32)
    d_mom_accel = cp.asarray(pre["mom_accel"]) if pre.get("mom_accel") is not None else cp.zeros((ns, nd), dtype=cp.float32)

    BLOCK = 256
    BATCH = n_ind
    d_params = cp.asarray(params_np)
    d_results = cp.zeros((BATCH * 5,), dtype=cp.float32)
    grid = (BATCH + BLOCK - 1) // BLOCK

    base.CUDA_KERNEL((grid,), (BLOCK,), (
        np.int32(ns), np.int32(nd),
        d_close, d_rsi, d_bb, d_vr, d_mh, d_ml,
        d_kv, d_dv, d_mom3, d_mom5, d_mom10,
        d_ig, d_gp, d_nh, d_wr,
        d_ma3, d_ma5, d_ma10, d_ma15, d_ma20, d_ma30, d_ma60,
        d_vp, d_squeeze, d_newhigh, d_adx, d_bias, d_obv_rising, d_atr_pct,
        d_open_arr, d_top100, d_market,
        d_up_days, d_week52, d_vol_up_days, d_mom_accel,
        d_params, np.int32(N_PARAMS_FULL),
        d_results, np.int32(BATCH),
        np.int32(pre["train_start"]),
        np.int32(pre["train_end"])
    ))

    results = d_results.get().reshape(BATCH, 5)
    return results  # [score, n_trades, avg, total, wr]


def compute_dd_from_trades(trades):
    """從 trades 算 MaxDD（equity curve peak to trough）"""
    if not trades:
        return 0.0
    rets = sorted([t for t in trades if t.get("sell_date")], key=lambda t: t.get("sell_date", ""))
    if not rets:
        return 0.0
    cum = np.cumsum([t.get("return", 0) for t in rets])
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)
    return float(dd.min())  # negative


def compute_sharpe(trades):
    if not trades:
        return 0.0
    rets = np.array([t.get("return", 0) for t in trades if t.get("sell_date")])
    if len(rets) == 0 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std())


def evaluate_objectives_full(ind, pre):
    """
    對單一 ind 跑 cpu_replay 拿 4 個 objectives:
      total / wr / -MaxDD (negate 變最大化) / n_trades
    """
    trades = base.cpu_replay(pre, ind)
    completed = [t for t in trades if t.get("sell_date") and t.get("reason") != "持有中"]
    n = len(completed)
    if n < 15 or n > 200:
        return None  # gate fail
    rets = np.array([t.get("return", 0) for t in completed])
    total = float(rets.sum())
    wr = float((rets > 0).mean() * 100)
    dd = compute_dd_from_trades(completed)
    sharpe = compute_sharpe(completed)
    if not (np.isfinite(total) and np.isfinite(wr)):
        return None
    return {
        "total": total, "wr": wr, "dd": dd, "n_trades": n,
        "sharpe": sharpe, "neg_dd": -dd,  # neg_dd 越大越好
    }


def dominates(a, b):
    """a dominates b iff: a 在所有 obj >= b, AND 在至少一個 obj > b"""
    objs = ["total", "wr", "neg_dd", "n_trades"]
    geq = all(a[o] >= b[o] for o in objs)
    gt = any(a[o] > b[o] for o in objs)
    return geq and gt


def fast_non_dominated_sort(pop_objs):
    """NSGA-II Pareto rank assignment"""
    n = len(pop_objs)
    fronts = [[]]
    S = [[] for _ in range(n)]
    n_dom = [0] * n
    rank = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if dominates(pop_objs[i], pop_objs[j]):
                S[i].append(j)
            elif dominates(pop_objs[j], pop_objs[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            rank[i] = 0
            fronts[0].append(i)
    f_idx = 0
    while fronts[f_idx]:
        nxt = []
        for i in fronts[f_idx]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    rank[j] = f_idx + 1
                    nxt.append(j)
        f_idx += 1
        fronts.append(nxt)
    fronts.pop()  # remove empty
    return fronts, rank


def crowding_distance(front_idx, pop_objs):
    """NSGA-II crowding distance（保多樣性）"""
    n = len(front_idx)
    if n == 0: return {}
    if n == 1: return {front_idx[0]: float("inf")}
    if n == 2: return {front_idx[0]: float("inf"), front_idx[1]: float("inf")}
    dist = {i: 0.0 for i in front_idx}
    objs = ["total", "wr", "neg_dd", "n_trades"]
    for o in objs:
        sorted_idx = sorted(front_idx, key=lambda i: pop_objs[i][o])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        v_min = pop_objs[sorted_idx[0]][o]
        v_max = pop_objs[sorted_idx[-1]][o]
        denom = v_max - v_min
        if denom == 0: continue
        for k in range(1, n - 1):
            dist[sorted_idx[k]] += (pop_objs[sorted_idx[k+1]][o] - pop_objs[sorted_idx[k-1]][o]) / denom
    return dist


def nsga2_select(pop, pop_objs, k):
    """選 k 個個體（rank → crowding distance）"""
    fronts, rank = fast_non_dominated_sort(pop_objs)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            cd = crowding_distance(front, pop_objs)
            front_sorted = sorted(front, key=lambda i: -cd.get(i, 0))
            selected.extend(front_sorted[:k - len(selected)])
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="3060", choices=["3060", "5090"])
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print("=" * 80)
    print(f"V42 NSGA-II Multi-Objective Evolution — {cfg['label']}")
    print("=" * 80)

    # === 1. 載 89.90 + cache ===
    print(f"\n[1/4] 載 89.90 + cache...")
    seed_params = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)

    # === 2. SEED objectives ===
    print(f"\n[2/4] 算 89.90 SEED 4 objectives (作為 reference)...")
    seed_obj = evaluate_objectives_full(seed_params, pre)
    if seed_obj is None:
        print(f"  ⚠️ SEED gate fail")
        seed_total, seed_wr, seed_dd, seed_n = 0, 0, 0, 0
    else:
        seed_total = seed_obj["total"]
        seed_wr = seed_obj["wr"]
        seed_dd = seed_obj["dd"]
        seed_n = seed_obj["n_trades"]
        print(f"  89.90: total {seed_total:+.0f}%, wr {seed_wr:.1f}%, MaxDD {seed_dd:.1f}%, n {seed_n}")

    # === 3. NSGA-II main loop ===
    print(f"\n[3/4] NSGA-II evolve: pop={cfg['pop_size']}, gen={cfg['n_gen']}")

    POP = cfg["pop_size"]
    N_GEN = cfg["n_gen"]

    # init population (含 89.90 seed 為 ind 0)
    print(f"  初始化 population...")
    pop = [dict(seed_params)]
    for _ in range(POP - 1):
        pop.append(random_individual())

    # 評估初始 pop（用 cpu_replay 算 4 obj）
    print(f"  初始 pop 跑 cpu_replay 算 4 obj（會慢，每個 ind 1-2 秒）...")
    pop_objs = []
    valid_pop = []
    t_init = time.time()
    for i, ind in enumerate(pop):
        obj = evaluate_objectives_full(ind, pre)
        if obj is not None:
            pop_objs.append(obj)
            valid_pop.append(ind)
        if (i + 1) % 16 == 0:
            elapsed = time.time() - t_init
            print(f"    {i+1}/{POP} ({elapsed/60:.1f}min, valid {len(valid_pop)})")

    pop = valid_pop
    pop_objs = list(pop_objs)
    print(f"  Init valid: {len(pop)}/{POP} ({(time.time()-t_init)/60:.1f}min)")

    if len(pop) < 8:
        print(f"❌ valid pop 太少")
        return

    pareto_history = []
    t_evolve_start = time.time()

    for gen in range(N_GEN):
        gen_start = time.time()

        # 產生 offspring（pop_size 個）
        offspring = []
        offspring_objs = []
        attempts = 0
        while len(offspring) < POP and attempts < POP * 3:
            attempts += 1
            # tournament select
            i1, i2 = random.sample(range(len(pop)), 2)
            j1, j2 = random.sample(range(len(pop)), 2)
            # 用 rank 比較選 parent
            fronts, rank = fast_non_dominated_sort(pop_objs)
            p1 = pop[i1] if rank[i1] < rank[i2] else pop[i2]
            p2 = pop[j1] if rank[j1] < rank[j2] else pop[j2]
            child = crossover(p1, p2)
            child = mutate(child, rate=0.15)
            obj = evaluate_objectives_full(child, pre)
            if obj is not None:
                offspring.append(child)
                offspring_objs.append(obj)

        # 合 parent + offspring
        combined = pop + offspring
        combined_objs = pop_objs + offspring_objs

        # NSGA-II 選 POP 個
        selected = nsga2_select(combined, combined_objs, POP)
        pop = [combined[i] for i in selected]
        pop_objs = [combined_objs[i] for i in selected]

        # 抓 Pareto front (rank 0)
        fronts, rank = fast_non_dominated_sort(pop_objs)
        front0 = [pop_objs[i] for i in fronts[0]]
        front0_inds = [pop[i] for i in fronts[0]]

        # 印 stats
        totals = [o["total"] for o in pop_objs]
        wrs = [o["wr"] for o in pop_objs]
        dds = [o["dd"] for o in pop_objs]
        elapsed = time.time() - t_evolve_start
        gen_time = time.time() - gen_start
        print(f"  Gen {gen+1}/{N_GEN} ({elapsed/60:.1f}min, +{gen_time/60:.1f}min): "
              f"Pareto={len(fronts[0])}, "
              f"max total={max(totals):.0f}%, max wr={max(wrs):.1f}%, "
              f"min DD={max(dds):.1f}%（最小回撤越接近0越好）")

        # 找有沒有 dominate 89.90 的
        if seed_obj is not None:
            dominators = [o for o in front0
                          if o["total"] >= seed_total and o["wr"] >= seed_wr
                          and o["dd"] >= seed_dd and o["n_trades"] >= seed_n
                          and (o["total"] > seed_total or o["wr"] > seed_wr
                               or o["dd"] > seed_dd or o["n_trades"] > seed_n)]
            if dominators:
                print(f"    🔥 {len(dominators)} 個策略 dominate 89.90！")

        pareto_history.append({
            "gen": gen + 1, "n_pareto": len(fronts[0]),
            "max_total": max(totals), "max_wr": max(wrs),
            "min_dd": max(dds), "elapsed_min": elapsed / 60,
        })

    # === 4. 輸出 Pareto frontier ===
    print()
    print("=" * 80)
    print("📊 V42 NSGA-II Pareto Frontier")
    print("=" * 80)

    fronts, rank = fast_non_dominated_sort(pop_objs)
    front0_idx = fronts[0]
    pareto_inds = [pop[i] for i in front0_idx]
    pareto_objs = [pop_objs[i] for i in front0_idx]

    print(f"\n  Pareto front: {len(pareto_inds)} 個策略")
    print(f"\n  {'Rank':<5} {'total':<10} {'wr':<8} {'MaxDD':<10} {'n':<6} {'Sharpe':<8} {'vs 89.90'}")
    print(f"  {'-' * 70}")

    # sort by total desc
    sorted_pareto = sorted(zip(pareto_inds, pareto_objs), key=lambda x: -x[1]["total"])
    for i, (ind, o) in enumerate(sorted_pareto[:30]):  # top 30
        vs_seed = []
        if seed_obj:
            if o["total"] > seed_total: vs_seed.append("total↑")
            if o["wr"] > seed_wr: vs_seed.append("wr↑")
            if o["dd"] > seed_dd: vs_seed.append("DD↑")
            if o["n_trades"] > seed_n: vs_seed.append("n↑")
        vs_str = "/".join(vs_seed) if vs_seed else "-"
        print(f"  {i+1:<5} {o['total']:<+10.0f} {o['wr']:<8.1f} {o['dd']:<10.1f} "
              f"{o['n_trades']:<6} {o['sharpe']:<8.3f} {vs_str}")

    # 找 dominate 89.90 的
    dominators = []
    if seed_obj is not None:
        for ind, o in zip(pareto_inds, pareto_objs):
            if (o["total"] >= seed_total and o["wr"] >= seed_wr
                and o["dd"] >= seed_dd and o["n_trades"] >= seed_n
                and (o["total"] > seed_total or o["wr"] > seed_wr
                     or o["dd"] > seed_dd or o["n_trades"] > seed_n)):
                dominators.append((ind, o))

    print(f"\n  Dominate 89.90: {len(dominators)} 個策略")
    if dominators:
        print(f"  🔥🔥🔥 找到 dominate 89.90 的策略！89.90 不再是 global optimum")
        for i, (ind, o) in enumerate(dominators[:5]):
            print(f"\n  Dominator {i+1}: total {o['total']:+.0f}%, wr {o['wr']:.1f}%, "
                  f"DD {o['dd']:.1f}%, n {o['n_trades']}, Sharpe {o['sharpe']:.3f}")
    else:
        print(f"  🔴 沒有策略同時在 4 個目標都贏 89.90 → 89.90 在 4-obj 空間也是 Pareto-optimal")
        print(f"     但可能有「換得失」型 Pareto 點：")
        # 找 DD 改善大的（接受 total 損失）
        better_dd = [t for t in zip(pareto_inds, pareto_objs)
                     if seed_obj and t[1]["dd"] > seed_dd + 5]
        if better_dd:
            best_dd = max(better_dd, key=lambda t: t[1]["dd"])
            print(f"     最佳 DD 換 total: DD {best_dd[1]['dd']:.1f}% (vs 89.90 {seed_dd:.1f}%) "
                  f"total {best_dd[1]['total']:+.0f}% (vs 89.90 {seed_total:+.0f}%)")
        better_wr = [t for t in zip(pareto_inds, pareto_objs)
                     if seed_obj and t[1]["wr"] > seed_wr + 3]
        if better_wr:
            best_wr = max(better_wr, key=lambda t: t[1]["wr"])
            print(f"     最佳 wr 換 total: wr {best_wr[1]['wr']:.1f}% (vs 89.90 {seed_wr:.1f}%) "
                  f"total {best_wr[1]['total']:+.0f}% (vs 89.90 {seed_total:+.0f}%)")

    # 存
    out = os.path.join(USER_SE, f"nsga2_results_{args.mode}.json")
    with open(out, "w") as f:
        json.dump({
            "mode": args.mode, "config": cfg,
            "seed_obj": seed_obj,
            "n_pareto": len(pareto_inds),
            "n_dominators": len(dominators),
            "pareto_top30": [{"params": ind, "obj": o}
                              for ind, o in sorted_pareto[:30]],
            "history": pareto_history,
            "elapsed_min": (time.time() - t_evolve_start) / 60,
        }, f, indent=2, default=str)
    print(f"\n結果存到 {out}")
    print(f"總耗時: {(time.time()-t_evolve_start)/60:.1f} 分鐘")


if __name__ == "__main__":
    main()
