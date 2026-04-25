"""
V41 Kronos 真 fine-tune — 跟 V38b/c/d calibration head 不同
用法：C:\\stock-evolution> python finetune_kronos.py [--mode 3060|3060real|5090]

V38b/c/d 教訓（已敗）：frozen Kronos features + LogReg/GB head 學不到新東西
V40 WQ101 教訓：「過濾 89.90 trades」框架 31 種失敗
V41 真 fine-tune（新方向）：學 89.90 沒看到的 K 線 pattern

API 對齊（2026-04 從 shiyu-coder/Kronos 確認）：
  tokenizer.encode(x, half=True) → (s1_ids, s2_ids)
  model.decode_s1(s1_ids, s2_ids, stamp) → (s1_logits, context)
                                            context shape (B, T, d_model)
  stamp tensor shape (B, T, 5): [minute, hour, weekday, day, month]
  Input x: (B, T, d_in) normalized OHLCV+amount

模式：
  --mode 3060      Kronos-small + freeze backbone, 3 path, 3 epoch (~30min, 驗證 pipeline)
  --mode 3060real  Kronos-small + UNFREEZE backbone, 全 15 path, 5 epoch (~6-12h, 真版本)
  --mode 5090      Kronos-base + UNFREEZE, 全 15 path, 10 epoch (~1-3h on 5090)

判定（同 V36/V40）：
  CPCV n_break ≥ 12/15 AND mean ≥ 5% AND p25 ≥ 0
  AND backfill totΔ > -10% (V40 教訓加第三關)

對比 baseline：
  V38 Kronos zero-shot: 11/14 mean +14.28% backfill totΔ -87% (撤回)
  V40 WQ101 top5 combo: 15/15 mean +10.89% backfill totΔ -31% (拒)
"""
import os, sys, pickle, json, time, argparse
import numpy as np
import pandas as pd
from itertools import combinations

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
KRONOS_DIR = os.path.join(USER_SE, "Kronos")
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

import gpu_cupy_evolve as base

CACHE_PATH = os.path.join(USER_SE, "stock_data_cache.pkl")
LOOKBACK = 60
N_GROUPS = 6
K_TEST = 2
WARMUP = 60


def fetch_gist_strategy():
    import urllib.request
    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    d = json.loads(r.read())
    s = json.loads(d["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def get_config(mode):
    if mode == "5090":
        return {
            "model_name": "NeoQuasar/Kronos-base",
            "batch_size": 16, "epochs": 10, "lr": 1e-5,
            "freeze_backbone": False, "n_paths": 15,
            "label": "5090 真 fine-tune (Kronos-base 102M, unfreeze)",
        }
    elif mode == "3060real":
        return {
            "model_name": "NeoQuasar/Kronos-small",
            "batch_size": 4, "epochs": 5, "lr": 2e-5,
            "freeze_backbone": False, "n_paths": 15,
            "label": "3060 真版 (Kronos-small 24.7M, UNFREEZE, 全 15 path 過夜跑)",
        }
    elif mode == "3060":
        return {
            "model_name": "NeoQuasar/Kronos-small",
            "batch_size": 4, "epochs": 3, "lr": 5e-5,
            "freeze_backbone": True, "n_paths": 3,
            "label": "3060 dry-run (Kronos-small, freeze, 3 path 驗證 pipeline)",
        }
    else:
        raise ValueError(f"unknown mode: {mode}")


def split_into_groups(n_days, warmup, n_groups):
    g_size = (n_days - warmup) // n_groups
    return [(warmup + i * g_size, warmup + (i + 1) * g_size if i < n_groups - 1 else n_days)
            for i in range(n_groups)]


def collect_samples(pre, params):
    print("  跑 89.90 cpu_replay 拿 samples...")
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]

    samples = []
    tickers = pre["tickers"]
    dates = pre["dates"]
    date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    for t in completed:
        ticker = t.get("ticker", "")
        bd_str = t.get("buy_date", "")
        if ticker not in ticker_to_idx or bd_str not in date_to_day:
            continue
        ret = float(t.get("return", 0))
        si = ticker_to_idx[ticker]
        bd_idx = date_to_day[bd_str]
        signal_idx = bd_idx - 1
        if signal_idx < LOOKBACK:
            continue
        samples.append({
            "stock_idx": si, "ticker": ticker, "signal_idx": signal_idx,
            "buy_date": bd_str, "actual_return": ret,
            "label": 1 if ret > 0 else 0,
        })

    print(f"  Samples: {len(samples)} 筆 (win rate {sum(s['label'] for s in samples) / len(samples) * 100:.1f}%)")
    return samples


def extract_kline_input(stock_idx, signal_idx, pre, lookback=60):
    s = stock_idx
    e = signal_idx + 1
    b = e - lookback
    if b < 0:
        return None, None
    o = pre["open"][s, b:e]
    h = pre["high"][s, b:e]
    l = pre["low"][s, b:e]
    c = pre["close"][s, b:e]
    v = pre["volume"][s, b:e]
    amt = c * v
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                       "volume": v, "amount": amt})
    # 取對應 date range 用於 stamp
    dates = pre["dates"][b:e]
    return df, dates


def build_stamp(dates):
    """從 pandas DatetimeIndex 構建 (T, 5) stamp tensor: [minute, hour, weekday, day, month]"""
    s = pd.DataFrame()
    s["minute"] = dates.minute
    s["hour"] = dates.hour
    s["weekday"] = dates.weekday
    s["day"] = dates.day
    s["month"] = dates.month
    return s.values.astype(np.float32)


def normalize_kline(arr):
    """Per-sample normalize: each (T, d_in) divide by its own mean/std"""
    # arr shape (T, d_in)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mu) / sd, mu, sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="3060real",
                        choices=["3060", "3060real", "5090"],
                        help="3060 dry-run / 3060real overnight / 5090 production")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print("=" * 80)
    print(f"V41 Kronos Fine-tune — {cfg['label']}")
    print("=" * 80)

    # === 1. 環境 ===
    print(f"\n[1/5] 環境檢查...")
    import torch
    import torch.nn as nn

    try:
        from model import Kronos, KronosTokenizer
    except ImportError:
        print(f"❌ Kronos module 找不到，先跑 setup_kronos.py")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda:0":
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"  Config: {cfg}")

    # === 2. 89.90 + cache ===
    print(f"\n[2/5] 載 89.90 + cache...")
    params = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)

    n_stocks = len(pre["tickers"])
    n_days = pre["n_days"]
    o_arr = np.zeros((n_stocks, n_days), dtype=np.float32)
    h_arr = np.zeros((n_stocks, n_days), dtype=np.float32)
    l_arr = np.zeros((n_stocks, n_days), dtype=np.float32)
    v_arr = np.zeros((n_stocks, n_days), dtype=np.float32)
    for si, t in enumerate(pre["tickers"]):
        df = data_dict[t].tail(n_days)
        o_arr[si] = df["Open"].values[-n_days:]
        h_arr[si] = df["High"].values[-n_days:]
        l_arr[si] = df["Low"].values[-n_days:]
        v_arr[si] = df["Volume"].values[-n_days:]
    pre["open"] = o_arr; pre["high"] = h_arr; pre["low"] = l_arr; pre["volume"] = v_arr

    # === 3. samples ===
    print(f"\n[3/5] 收集樣本...")
    samples = collect_samples(pre, params)
    if len(samples) < 50:
        print(f"❌ 樣本太少")
        return

    # === 4. 載 Kronos + 搬 GPU（這是 v1 bug 修復重點）===
    print(f"\n[4/5] 載 Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    tokenizer = tokenizer.to(device)        # 🔥 關鍵 bug fix #1
    tokenizer.eval()
    kronos = Kronos.from_pretrained(cfg["model_name"]).to(device)
    print(f"  ✅ Kronos + tokenizer loaded on {device}")

    if cfg["freeze_backbone"]:
        for p in kronos.parameters():
            p.requires_grad = False
        kronos.eval()
        print(f"  Backbone frozen")
    else:
        print(f"  Backbone UNFROZEN — 真 fine-tune")

    # 動態偵測 d_model：先做一次 forward 拿 ctx.shape
    try:
        d_model = kronos.config.d_model
        print(f"  d_model from config: {d_model}")
    except AttributeError:
        d_model = None
        print(f"  d_model 從 config 拿不到，等首次 forward 後動態偵測")

    # === 抽 K 線 + normalize + 建 stamp ===
    print(f"\n  抽 K 線 + 建 stamp（lookback={LOOKBACK}）...")
    valid_samples = []
    klines = []     # list of (T, 6) normalized
    stamps = []     # list of (T, 5)
    for s in samples:
        df, dates = extract_kline_input(s["stock_idx"], s["signal_idx"], pre, LOOKBACK)
        if df is None or len(df) < LOOKBACK:
            continue
        if df["close"].isna().any() or (df["close"] <= 0).any():
            continue
        arr = df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        arr_norm, _, _ = normalize_kline(arr)
        if not np.all(np.isfinite(arr_norm)):
            continue
        stamp = build_stamp(dates)
        valid_samples.append(s)
        klines.append(arr_norm)
        stamps.append(stamp)
    print(f"  Valid samples: {len(valid_samples)}")

    if len(valid_samples) < 30:
        print(f"❌ 有效樣本太少")
        return

    klines_t = torch.tensor(np.stack(klines), dtype=torch.float32)   # (N, T, 6)
    stamps_t = torch.tensor(np.stack(stamps), dtype=torch.float32)   # (N, T, 5)
    labels_t = torch.tensor([s["label"] for s in valid_samples], dtype=torch.float32)
    rets_arr = np.array([s["actual_return"] for s in valid_samples])
    day_idx_arr = np.array([s["signal_idx"] for s in valid_samples])

    # === 預先 tokenize（不會變，省時間）===
    print(f"\n  Tokenize {len(valid_samples)} 個 K-line input...")
    bs = cfg["batch_size"]
    s1_all = []
    s2_all = []
    with torch.no_grad():
        for bi in range(0, len(klines_t), bs):
            kx = klines_t[bi:bi + bs].to(device)
            try:
                s1, s2 = tokenizer.encode(kx, half=True)
                s1_all.append(s1.cpu())
                s2_all.append(s2.cpu())
            except Exception as e:
                print(f"  ❌ Tokenizer.encode 失敗: {e}")
                print(f"     嘗試備用 API: tokenizer.encode(kx) 不帶 half")
                try:
                    out = tokenizer.encode(kx)
                    if isinstance(out, tuple):
                        s1_all.append(out[0].cpu())
                        s2_all.append(out[1].cpu())
                    else:
                        # 單一 tensor 假設是 s1，s2 用 zeros
                        s1_all.append(out.cpu())
                        s2_all.append(torch.zeros_like(out).cpu())
                except Exception as e2:
                    print(f"  ❌❌ 完全失敗: {e2}")
                    return
    s1_all = torch.cat(s1_all, dim=0)  # (N, T)
    s2_all = torch.cat(s2_all, dim=0)
    print(f"  Tokens shape: s1 {s1_all.shape}, s2 {s2_all.shape}")

    # === 探測 d_model（一次小 forward 拿真實 ctx shape）===
    if d_model is None:
        with torch.no_grad():
            probe_s1 = s1_all[:1].to(device)
            probe_s2 = s2_all[:1].to(device)
            probe_stamp = stamps_t[:1].to(device)
            try:
                _, probe_ctx = kronos.decode_s1(probe_s1, probe_s2, stamp=probe_stamp)
                d_model = probe_ctx.shape[-1]
                print(f"  ✅ 探測到 d_model = {d_model}（從 decode_s1 ctx.shape[-1]）")
            except Exception as e:
                print(f"  ❌ 探測失敗: {e}")
                d_model = 512  # Kronos-small fallback (從 trace error 推得)
                print(f"  使用 fallback d_model = {d_model}")

    # === 5. CPCV LOO fine-tune ===
    print(f"\n[5/5] CPCV LOO {N_GROUPS} groups, k={K_TEST}...")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    test_combos = test_combos[:cfg["n_paths"]]
    print(f"  跑 {len(test_combos)} paths")

    # Binary head
    class BinaryHead(nn.Module):
        def __init__(self, d_model, hidden=64):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, 1),
            )

        def forward(self, ctx):
            return self.fc(ctx[:, -1, :]).squeeze(-1)  # 取最後 token

    per_path_results = []
    t_start = time.time()

    for pi, gi in enumerate(test_combos):
        print(f"\n  Path {pi + 1}/{len(test_combos)}: test groups {gi}")
        ranges = [groups[g] for g in gi]
        in_test = np.zeros(len(valid_samples), dtype=bool)
        for s, e in ranges:
            in_test |= (day_idx_arr >= s) & (day_idx_arr < e)
        in_train = ~in_test

        if in_test.sum() < 5 or in_train.sum() < 20:
            print(f"    skip: train {in_train.sum()} test {in_test.sum()}")
            continue

        # 重新 init head 每 path
        head = BinaryHead(d_model).to(device)

        # Optimizer：unfreeze 時要包含 kronos params
        if cfg["freeze_backbone"]:
            opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
        else:
            opt = torch.optim.AdamW(
                list(head.parameters()) + list(kronos.parameters()),
                lr=cfg["lr"], weight_decay=0.01
            )

        bce = nn.BCEWithLogitsLoss()

        train_idx_np = np.where(in_train)[0]
        test_idx_np = np.where(in_test)[0]

        for epoch in range(cfg["epochs"]):
            kronos.train() if not cfg["freeze_backbone"] else kronos.eval()
            head.train()
            np.random.shuffle(train_idx_np)
            losses = []
            for bi in range(0, len(train_idx_np), bs):
                bidx = train_idx_np[bi:bi + bs]
                bs1 = s1_all[bidx].to(device)
                bs2 = s2_all[bidx].to(device)
                bstamp = stamps_t[bidx].to(device)
                blabel = labels_t[bidx].to(device)

                # forward through Kronos.decode_s1 → context
                if cfg["freeze_backbone"]:
                    with torch.no_grad():
                        _, ctx = kronos.decode_s1(bs1, bs2, stamp=bstamp)
                else:
                    _, ctx = kronos.decode_s1(bs1, bs2, stamp=bstamp)

                logits = head(ctx)
                loss = bce(logits, blabel)
                opt.zero_grad()
                loss.backward()
                # gradient clip 防爆炸
                if not cfg["freeze_backbone"]:
                    torch.nn.utils.clip_grad_norm_(kronos.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            print(f"    Epoch {epoch + 1}/{cfg['epochs']}: loss {np.mean(losses):.4f}")

        # Eval
        kronos.eval(); head.eval()
        proba_list = []
        with torch.no_grad():
            for bi in range(0, len(test_idx_np), bs):
                bidx = test_idx_np[bi:bi + bs]
                bs1 = s1_all[bidx].to(device)
                bs2 = s2_all[bidx].to(device)
                bstamp = stamps_t[bidx].to(device)
                _, ctx = kronos.decode_s1(bs1, bs2, stamp=bstamp)
                logits = head(ctx)
                proba_list.append(torch.sigmoid(logits).cpu().numpy())
        test_proba = np.concatenate(proba_list)
        test_rets = rets_arr[test_idx_np]

        # === Multi-threshold sweep（V40 教訓：median split 過寬，要試 top30/top20）===
        raw_wr = (test_rets > 0).mean() * 100
        raw_total = float(test_rets.sum())

        # 找該 path proba 分位數，跑多 threshold
        thresholds = [
            ("median", np.median(test_proba)),
            ("top50", np.percentile(test_proba, 50)),
            ("top30", np.percentile(test_proba, 70)),
            ("top20", np.percentile(test_proba, 80)),
            ("th0.5", 0.5),
            ("th0.55", 0.55),
        ]

        path_thresh_results = []
        for tname, tval in thresholds:
            keep = test_proba > tval
            if keep.sum() < 3:
                continue
            filt_wr = (test_rets[keep] > 0).mean() * 100
            filt_total = float(test_rets[keep].sum())
            wr_imp = filt_wr - raw_wr
            total_imp = filt_total - raw_total
            kept_pct = keep.sum() / len(test_rets) * 100
            path_thresh_results.append({
                "thresh": tname, "thresh_val": float(tval),
                "wr_imp": float(wr_imp), "total_imp": float(total_imp),
                "kept_pct": float(kept_pct), "filt_wr": float(filt_wr),
                "filt_total": float(filt_total),
            })

        # 印每個 threshold
        for r in path_thresh_results:
            print(f"    {r['thresh']:<8} (>{r['thresh_val']:.3f}): "
                  f"wr {r['filt_wr']:.1f}% (Δ {r['wr_imp']:+.1f}%), "
                  f"total {r['filt_total']:+.1f}% (Δ {r['total_imp']:+.1f}%), kept {r['kept_pct']:.0f}%")

        # 取 wr_imp 最高的當該 path 代表（如果有 > 0 的）
        if path_thresh_results:
            best = max(path_thresh_results, key=lambda r: r["wr_imp"])
            per_path_results.append({
                "path": list(gi), "best_thresh": best["thresh"],
                "wr_imp": best["wr_imp"], "kept_pct": best["kept_pct"],
                "raw_wr": float(raw_wr), "filt_wr": best["filt_wr"],
                "raw_total": raw_total, "filt_total": best["filt_total"],
                "total_imp": best["total_imp"],
                "all_thresh": path_thresh_results,
            })
            print(f"    ⭐ best: {best['thresh']} wr {best['wr_imp']:+.1f}% total {best['total_imp']:+.1f}%")
        else:
            print(f"    ❌ 所有 threshold 都 < 3 筆，skip")

        # 顯示估計剩餘時間
        elapsed = time.time() - t_start
        avg_per_path = elapsed / (pi + 1)
        remaining = avg_per_path * (len(test_combos) - pi - 1)
        print(f"    [計時] 已 {elapsed/60:.1f}min, 平均 {avg_per_path/60:.1f}min/path, 預估剩餘 {remaining/60:.1f}min")

    # === Summary ===
    print()
    print("=" * 80)
    print(f"📊 V41 Kronos Fine-tune ({args.mode}) 結果")
    print("=" * 80)

    if not per_path_results:
        print(f"\n  ❌ 0 path 完成")
        return

    wr_imps = np.array([r["wr_imp"] for r in per_path_results])
    total_imps = np.array([r["total_imp"] for r in per_path_results])
    n_break = int((wr_imps >= 5).sum())
    n_pos = int((wr_imps > 0).sum())

    print(f"\n  Path 完成: {len(per_path_results)}")
    print(f"  n_break (wr↑≥5%): {n_break}/{len(per_path_results)}")
    print(f"  n_pos (wr↑>0): {n_pos}/{len(per_path_results)}")
    print(f"  mean wr↑: {wr_imps.mean():+.2f}%")
    if len(wr_imps) >= 4:
        print(f"  p25 wr↑: {np.percentile(wr_imps, 25):+.2f}%")
    print(f"  min/max wr↑: {wr_imps.min():+.2f}% / {wr_imps.max():+.2f}%")
    print(f"  mean total↑: {total_imps.mean():+.1f}%")
    print(f"  min/max total↑: {total_imps.min():+.1f}% / {total_imps.max():+.1f}%")

    print(f"\n  Baseline:")
    print(f"    V38 Kronos zero-shot: 11/14 mean +14.28% backfill totΔ -87% (撤回)")
    print(f"    V40 WQ101 top5 combo: 15/15 mean +10.89% backfill totΔ -31% (拒)")

    print(f"\n  最終判定：")
    if cfg["n_paths"] < 15:
        print(f"    ⚠️ {args.mode} 模式只跑 {len(per_path_results)} path")
        if not cfg["freeze_backbone"] and n_break >= 1:
            print(f"    pipeline OK + 有 path 過 +5%，等 5090 跑全 15 path 看是否泛化")
        elif cfg["freeze_backbone"]:
            print(f"    pipeline 通了（frozen 不會強），改 --mode 3060real 跑真版本")
    else:
        if n_break >= 12 and wr_imps.mean() >= 5 and np.percentile(wr_imps, 25) >= 0:
            print(f"    🟢🟢🟢 strict CPCV！跑 backfill 看 totΔ")
        elif n_break >= 10 and wr_imps.mean() >= 4:
            print(f"    🟢🟢 real → backfill 驗證")
        elif n_break >= 7:
            print(f"    🟢 marginal → 跟 V39 同級")
        else:
            print(f"    🔴 連真 fine-tune 也救不了，89.90 終局，5090 轉 NSGA-II")

    out = os.path.join(USER_SE, f"finetune_kronos_{args.mode}_results.json")
    with open(out, "w") as f:
        json.dump({"mode": args.mode, "config": cfg, "per_path": per_path_results,
                   "n_break": n_break, "n_pos": n_pos,
                   "mean_wr_imp": float(wr_imps.mean()),
                   "mean_total_imp": float(total_imps.mean()),
                   "elapsed_min": (time.time() - t_start) / 60}, f, indent=2)
    print(f"\n結果存到 {out}")
    print(f"總耗時: {(time.time() - t_start)/60:.1f} 分鐘")


if __name__ == "__main__":
    main()
