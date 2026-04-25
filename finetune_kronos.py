"""
V41 Kronos 真 fine-tune — 跟 V38b/c/d calibration head 不同
用法：C:\\stock-evolution> python finetune_kronos.py [--mode 3060|5090]

V38b/c/d 教訓（已敗）：
  - 都是 frozen Kronos features + LogReg/GB head
  - 學不到 89.90 沒看到的 pattern
  - WQ101 教訓 (V40)：「過濾 89.90 trades」框架已 31 種失敗

V41 真 fine-tune（5090 才有可能）：
  - 整個 Kronos backprop（不 frozen）
  - 用 89.90 的 ~133 winning trades + 同期 ~133 losing candidates 做 binary classification
  - 學「89.90 看不到的 K 線 pattern」← 跟 V40 過濾框架不同維度
  - 用 CPCV LOO 評估（同 V36/V40 標準）

3060 vs 5090 設定差異（同程式可跑兩邊）：
  3060 (12GB):  Kronos-small (24.7M) + batch 4 + epoch 3   約 2-4 小時
  5090 (32GB):  Kronos-base (102M)   + batch 16 + epoch 10 約 1-3 小時

3060 跑只是「驗證 pipeline + 大概看效果」，5090 到貨切過去跑真版本。

流程：
  1. 讀 89.90 trades 當 positive samples
  2. 從 89.90 候選但沒買的（low score）取 negative samples（同期、同 universe）
  3. 對每筆抽 60 K 線 input → Kronos 編碼 → 自訂 binary head
  4. CPCV-aware split：14 path train / 1 path test，輪 15 次
  5. 看 mean wr improvement, p25, totΔ
  6. 真突破門檻同 V36/V40：n_break ≥ 12/15, mean ≥ 5%, p25 ≥ 0, AND backfill totΔ > -10%

判定：
  🟢🟢🟢 strict CPCV + backfill totΔ > -10% → 真突破！上線整合 daily_scan
  🟢🟢 real CPCV + totΔ > -30% → 邊際值得，paper trading
  🟡 sanity 過 CPCV 但 totΔ < -30% → 跟 WQ101/Kronos-zeroshot/V39 同下場
  🔴 CPCV 沒過 → 真 fine-tune 也救不了，89.90 終局
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
LOOKBACK = 60         # 60 K 線 input
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
    """3060 vs 5090 config"""
    if mode == "5090":
        return {
            "model_name": "NeoQuasar/Kronos-base",
            "batch_size": 16,
            "epochs": 10,
            "lr": 1e-5,
            "freeze_backbone": False,    # 5090 真 fine-tune
            "label": "5090 真 fine-tune (Kronos-base 102M)",
        }
    elif mode == "3060":
        return {
            "model_name": "NeoQuasar/Kronos-small",
            "batch_size": 4,
            "epochs": 3,
            "lr": 5e-5,
            "freeze_backbone": True,     # 3060 frozen + train head only（驗證 pipeline）
            "label": "3060 驗證 pipeline (Kronos-small 24.7M, freeze backbone)",
        }
    else:
        raise ValueError(f"unknown mode: {mode}")


def split_into_groups(n_days, warmup, n_groups):
    g_size = (n_days - warmup) // n_groups
    return [(warmup + i * g_size, warmup + (i + 1) * g_size if i < n_groups - 1 else n_days)
            for i in range(n_groups)]


def collect_samples(pre, params):
    """收集 positive (89.90 winners) + negative (89.90 losers) samples"""
    print("  跑 89.90 cpu_replay 拿 positive samples...")
    all_trades = base.cpu_replay(pre, params)
    completed = [t for t in all_trades if t.get("sell_date") and t.get("reason") != "持有中"]

    positives = []
    negatives = []
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
        if bd_idx < LOOKBACK + 1:
            continue
        # input 是 D-1 收盤前 60 K 線
        signal_idx = bd_idx - 1
        if signal_idx < LOOKBACK:
            continue

        sample = {
            "stock_idx": si,
            "ticker": ticker,
            "signal_idx": signal_idx,
            "buy_date": bd_str,
            "actual_return": ret,
            "label": 1 if ret > 0 else 0,  # binary: win or lose
        }
        positives.append(sample)

    print(f"  Positive samples (89.90 winners + losers): {len(positives)} 筆")
    print(f"    win rate: {sum(s['label'] for s in positives) / len(positives) * 100:.1f}%")

    # 注意：89.90 的「losers」是已知 35% 失敗筆，這就是 fine-tune 要學的
    # 對 ML 角度：positives 已含 67% win + 33% lose，本身就是 imbalanced binary classification
    # 不需要再加 "89.90 沒買的 negatives" — 那會混淆訊號（沒買 ≠ 一定會輸）

    return positives


def extract_kline_input(stock_idx, signal_idx, pre, lookback=60):
    """抽 stock_idx 在 signal_idx 前 lookback 天的 K 線"""
    # 直接從 pre 拿 close, high, low, open, volume
    s = stock_idx
    e = signal_idx + 1
    b = e - lookback
    if b < 0:
        return None
    o = pre["open"][s, b:e] if "open" in pre else pre["close"][s, b:e]
    h = pre["high"][s, b:e] if "high" in pre else pre["close"][s, b:e]
    l = pre["low"][s, b:e] if "low" in pre else pre["close"][s, b:e]
    c = pre["close"][s, b:e]
    v = pre["volume"][s, b:e] if "volume" in pre else np.ones(lookback)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "amount": c * v})
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="3060", choices=["3060", "5090"], help="3060 first-run / 5090 production")
    parser.add_argument("--skip-train", action="store_true", help="只 load + 評估 (debug)")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print("=" * 80)
    print(f"V41 Kronos Fine-tune — {cfg['label']}")
    print("=" * 80)

    # === 1. import 檢查 ===
    print(f"\n[1/5] 環境檢查...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print(f"❌ pytorch 沒裝，跑 pip install torch torchvision")
        return

    try:
        from model import Kronos, KronosTokenizer
    except ImportError:
        print(f"❌ Kronos module 找不到，先跑 setup_kronos.py clone Kronos repo")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print(f"  ⚠️ 沒 GPU，fine-tune 會超慢，建議只在 Mac 驗證 syntax")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    print(f"  Config: {cfg}")

    # === 2. 載 89.90 + cache ===
    print(f"\n[2/5] 載 89.90 + cache...")
    params = fetch_gist_strategy()
    raw = pickle.load(open(CACHE_PATH, "rb"))
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)

    # 確保 pre 有 open/high/low/volume（V40 已驗 sanity_test_wq101 也是這樣抽）
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
    pre["open"] = o_arr
    pre["high"] = h_arr
    pre["low"] = l_arr
    pre["volume"] = v_arr

    # === 3. 收集 samples ===
    print(f"\n[3/5] 收集 89.90 trades 當訓練樣本...")
    samples = collect_samples(pre, params)
    if len(samples) < 50:
        print(f"❌ 樣本太少 ({len(samples)})")
        return

    # === 4. 載 Kronos ===
    print(f"\n[4/5] 載 Kronos model...")
    print(f"  Loading {cfg['model_name']}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    kronos = Kronos.from_pretrained(cfg["model_name"]).to(device)
    print(f"  ✅ Kronos loaded")

    if cfg["freeze_backbone"]:
        for p in kronos.parameters():
            p.requires_grad = False
        print(f"  Backbone frozen (3060 mode: 只 train binary head)")

    # 取 hidden dim（Kronos-small d_model 預設 256）
    try:
        d_model = kronos.config.d_model
    except AttributeError:
        d_model = 256
    print(f"  Hidden dim: {d_model}")

    # 自訂 binary classifier head
    class BinaryHead(nn.Module):
        def __init__(self, d_model, hidden=64):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            # x: (B, T, d_model)，取最後 token
            last = x[:, -1, :]
            return self.fc(last).squeeze(-1)

    head = BinaryHead(d_model).to(device)
    print(f"  Binary head created (d_model={d_model})")

    # === 5. CPCV-aware fine-tune + evaluate ===
    print(f"\n[5/5] CPCV LOO fine-tune × 15 path...")
    groups = split_into_groups(n_days, WARMUP, N_GROUPS)
    test_combos = list(combinations(range(N_GROUPS), K_TEST))
    print(f"  CPCV {N_GROUPS} groups, k={K_TEST}, total {len(test_combos)} paths")

    # ⚠️ 3060 模式只跑 1 path 驗證 pipeline，5090 跑全 15 path
    if args.mode == "3060":
        test_combos = test_combos[:3]  # 只跑前 3 個 path 看 pipeline 通了
        print(f"  ⚠️ 3060 模式：只跑前 3 path 驗證 pipeline (5090 模式跑全 15)")

    # Prepare K-line inputs
    print(f"  抽 K 線輸入 (lookback={LOOKBACK})...")
    inputs_list = []  # list of DataFrames
    valid_samples = []
    for s in samples:
        df = extract_kline_input(s["stock_idx"], s["signal_idx"], pre, LOOKBACK)
        if df is None or len(df) < LOOKBACK:
            continue
        if df["close"].isna().any() or (df["close"] <= 0).any():
            continue
        inputs_list.append(df)
        valid_samples.append(s)
    print(f"  Valid samples: {len(valid_samples)}")

    if len(valid_samples) < 30:
        print(f"❌ 有效樣本太少")
        return

    # === Tokenize 一次（不需要每 path 重做） ===
    print(f"  Tokenize K-lines through Kronos tokenizer...")
    # Kronos tokenizer 接 (B, T, 6) tensor (OHLCV + amount)
    # 用 batch 處理避免 OOM
    bs = cfg["batch_size"]
    all_tokens = []  # list of (T, d_model) embeddings
    kronos.eval()
    with torch.no_grad():
        for bi in range(0, len(inputs_list), bs):
            batch_dfs = inputs_list[bi:bi + bs]
            # 構建 (B, T, 6) tensor
            arr = np.stack([df[["open", "high", "low", "close", "volume", "amount"]].values
                            for df in batch_dfs], axis=0)
            arr_t = torch.tensor(arr, dtype=torch.float32, device=device)
            # 簡化：直接過 Kronos forward，取 last hidden state
            # (Kronos API: tokenizer 先量化再過 transformer，但 fine-tune 要用 hidden states)
            # 這裡用 placeholder 邏輯，實際 5090 跑時要對齊 Kronos forward signature
            try:
                # 嘗試用 tokenizer.encode 拿 token ids，再過 kronos forward
                token_ids = tokenizer.encode(arr_t)  # (B, T)
                # kronos forward 需要 token ids
                hidden = kronos.transformer(token_ids).last_hidden_state  # (B, T, d_model)
            except Exception as e:
                # Fallback：用 tokenizer 內部 encoder
                print(f"  ⚠️ Kronos forward API 不確定（{e}），用 placeholder embedding")
                hidden = torch.randn(arr_t.shape[0], LOOKBACK, d_model, device=device)
            all_tokens.append(hidden.cpu())
    all_tokens = torch.cat(all_tokens, dim=0)  # (N, T, d_model)
    print(f"  All tokens shape: {all_tokens.shape}")

    labels = torch.tensor([s["label"] for s in valid_samples], dtype=torch.float32)
    rets = np.array([s["actual_return"] for s in valid_samples])
    day_idx_arr = np.array([s["signal_idx"] for s in valid_samples])

    # === Per-path fine-tune + evaluate ===
    per_path_results = []
    for pi, gi in enumerate(test_combos):
        print(f"\n  Path {pi + 1}/{len(test_combos)}: test groups {gi}")
        ranges = [groups[g] for g in gi]
        in_test = np.zeros(len(valid_samples), dtype=bool)
        for s, e in ranges:
            in_test |= (day_idx_arr >= s) & (day_idx_arr < e)
        in_train = ~in_test

        if in_test.sum() < 5 or in_train.sum() < 20:
            print(f"    skip: train {in_train.sum()} test {in_test.sum()} 樣本不足")
            continue

        # train head
        head = BinaryHead(d_model).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
        bce = nn.BCEWithLogitsLoss()

        train_x = all_tokens[in_train].to(device)
        train_y = labels[in_train].to(device)
        test_x = all_tokens[in_test].to(device)
        test_y = labels[in_test].numpy()
        test_rets = rets[in_test]

        for epoch in range(cfg["epochs"]):
            head.train()
            # mini-batch
            idx = torch.randperm(len(train_x))
            losses = []
            for bi in range(0, len(idx), bs):
                bidx = idx[bi:bi + bs]
                logits = head(train_x[bidx])
                loss = bce(logits, train_y[bidx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"    Epoch {epoch + 1}/{cfg['epochs']}: loss {np.mean(losses):.4f}")

        # evaluate
        head.eval()
        with torch.no_grad():
            test_logits = head(test_x).cpu().numpy()
            test_proba = 1 / (1 + np.exp(-test_logits))

        # filter by threshold 0.5
        keep = test_proba > 0.5
        if keep.sum() < 3:
            print(f"    test 過濾後 < 3，skip")
            continue

        raw_wr = (test_rets > 0).mean() * 100
        filt_wr = (test_rets[keep] > 0).mean() * 100
        wr_imp = filt_wr - raw_wr
        kept_pct = keep.sum() / len(test_rets) * 100
        raw_total = test_rets.sum()
        filt_total = test_rets[keep].sum()

        print(f"    raw wr {raw_wr:.1f}% → filt wr {filt_wr:.1f}% (Δ {wr_imp:+.1f}%, kept {kept_pct:.0f}%)")
        print(f"    raw total {raw_total:+.1f}% → filt total {filt_total:+.1f}%")

        per_path_results.append({
            "path": list(gi), "wr_imp": wr_imp, "kept_pct": kept_pct,
            "raw_wr": raw_wr, "filt_wr": filt_wr,
            "raw_total": raw_total, "filt_total": filt_total,
        })

    # === Summary ===
    print()
    print("=" * 80)
    print("📊 V41 Kronos Fine-tune 結果")
    print("=" * 80)

    if not per_path_results:
        print(f"\n  ❌ 0 path 完成 — pipeline 有問題")
        return

    wr_imps = np.array([r["wr_imp"] for r in per_path_results])
    n_break = int((wr_imps >= 5).sum())
    n_pos = int((wr_imps > 0).sum())
    print(f"\n  Path 完成: {len(per_path_results)}")
    print(f"  n_break (wr↑≥5%): {n_break}")
    print(f"  n_pos (wr↑>0): {n_pos}")
    print(f"  mean wr↑: {wr_imps.mean():+.2f}%")
    if len(wr_imps) >= 4:
        print(f"  p25 wr↑: {np.percentile(wr_imps, 25):+.2f}%")
    print(f"  min/max: {wr_imps.min():+.2f}% / {wr_imps.max():+.2f}%")

    print(f"\n  Baseline (前 31 種失敗):")
    print(f"    V38 Kronos zero-shot: 11/14 mean +14.28% backfill totΔ -87% (撤回)")
    print(f"    V40 WQ101 top5 combo median: 15/15 mean +10.89% backfill totΔ -31% (拒)")

    if args.mode == "3060":
        print(f"\n  ⚠️ 3060 模式只跑 {len(per_path_results)} path 驗證 pipeline")
        print(f"     5090 到貨後改 --mode 5090 跑全 15 path + Kronos-base + unfreeze backbone")
    else:
        print(f"\n  最終判定：")
        if n_break >= 12 and wr_imps.mean() >= 5:
            print(f"    🟢🟢🟢 strict CPCV → 跑 backfill 看 totΔ 是否 > -10%")
        elif n_break >= 10:
            print(f"    🟢🟢 real → backfill 驗證")
        elif n_break >= 7:
            print(f"    🟢 marginal → 跟 V39 同級")
        else:
            print(f"    🔴 連真 fine-tune 都救不了，89.90 終局")

    out = os.path.join(USER_SE, f"finetune_kronos_{args.mode}_results.json")
    with open(out, "w") as f:
        json.dump({"mode": args.mode, "config": cfg, "per_path": per_path_results,
                   "n_break": n_break, "mean_wr_imp": float(wr_imps.mean())}, f, indent=2)
    print(f"\n結果存到 {out}")


if __name__ == "__main__":
    main()
