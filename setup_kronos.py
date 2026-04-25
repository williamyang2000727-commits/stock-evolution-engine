"""
V38 Kronos Foundation Model — 環境準備 + Smoke test
用法：C:\\stock-evolution> python setup_kronos.py

做的事：
1. 檢查 dependencies (torch, transformers, huggingface_hub, einops)
2. clone Kronos GitHub repo（如果還沒）
3. 從 Hugging Face 下載 NeoQuasar/Kronos-small（24.7M，較快）+ Kronos-Tokenizer-base
4. Smoke test：對 1101.TW 的最近 60 K 線跑一次 prediction
5. 如果 OK，存到 KRONOS_DIR

注意：
- 預設用 Kronos-small（24.7M）+ Tokenizer-base，平衡速度跟效能
- Kronos-base 太大（102M），CPU 上慢
- 如果用戶有 GPU，可以改成 Kronos-base
"""
import os, sys, subprocess
from datetime import datetime

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
KRONOS_DIR = os.path.join(USER_SE, "Kronos")
LOG = os.path.join(USER_SE, "kronos_setup.log")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def check_dep(module_name, install_name=None):
    install_name = install_name or module_name
    try:
        __import__(module_name)
        return True
    except ImportError:
        log(f"  ❌ {module_name} 沒裝，需 pip install {install_name}")
        return False


def main():
    log("=" * 60)
    log("V38 Kronos Foundation Model — 環境準備")
    log("=" * 60)

    # === 1. 檢查 dependencies ===
    log(f"\n[1/4] 檢查 dependencies...")
    deps = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface_hub"),
        ("einops", "einops"),
    ]
    missing = []
    for mod, pkg in deps:
        if check_dep(mod, pkg):
            log(f"  ✅ {mod}")
        else:
            missing.append(pkg)

    if missing:
        log(f"\n  ⚠️ 缺少 packages: {missing}")
        log(f"  請手動跑：pip install {' '.join(missing)}")
        log(f"  跑完再執行此 script")
        return

    # === 2. Clone Kronos repo ===
    log(f"\n[2/4] Clone Kronos repo...")
    if os.path.isdir(KRONOS_DIR):
        log(f"  ✅ {KRONOS_DIR} 已存在（skip clone）")
    else:
        try:
            log(f"  git clone https://github.com/shiyu-coder/Kronos.git ...")
            subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git", KRONOS_DIR],
                          check=True, capture_output=True, text=True)
            log(f"  ✅ clone 完成")
        except subprocess.CalledProcessError as e:
            log(f"  ❌ clone 失敗：{e.stderr}")
            log(f"  請手動跑：cd {USER_SE} && git clone https://github.com/shiyu-coder/Kronos.git")
            return
        except FileNotFoundError:
            log(f"  ❌ git 沒裝。請先裝 Git for Windows: https://git-scm.com/download/win")
            return

    # 加 Kronos 路徑到 sys.path
    sys.path.insert(0, KRONOS_DIR)

    # === 3. Smoke test 載入 model ===
    log(f"\n[3/4] 從 Hugging Face 載入 model（首次會下載 ~150 MB）...")
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        log(f"  ✅ import model 成功")

        log(f"  載入 Kronos-Tokenizer-base...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        log(f"  ✅ tokenizer loaded")

        log(f"  載入 Kronos-small（24.7M params）...")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        log(f"  ✅ model loaded")

        # Detect GPU
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            log(f"  🎯 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            log(f"  ⚠️ 沒 GPU，使用 CPU（會慢一些）")

        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        log(f"  ✅ predictor instantiated")

    except Exception as e:
        log(f"  ❌ 載入失敗：{e}")
        import traceback
        log(traceback.format_exc())
        return

    # === 4. Smoke test：對 1101.TW 跑 prediction ===
    log(f"\n[4/4] Smoke test：對 1101.TW 預測未來 5 天...")
    try:
        import pickle
        import pandas as pd
        cache_path = os.path.join(USER_SE, "stock_data_cache.pkl")
        cache = pickle.load(open(cache_path, "rb"))

        if "1101.TW" not in cache:
            log(f"  ❌ 1101.TW 不在 cache")
            return

        df = cache["1101.TW"].tail(100).copy()
        # Kronos 要 lowercase columns
        df.columns = [c.lower() for c in df.columns]
        # 確認有 OHLC + volume
        required = ["open", "high", "low", "close", "volume"]
        for c in required:
            if c not in df.columns:
                log(f"  ❌ cache 缺欄位 {c}")
                return

        x_df = df[required].iloc[:60].reset_index(drop=True)
        x_timestamp = df.index[:60].to_series().reset_index(drop=True)
        # Future timestamps（連續 5 天）
        last_date = df.index[59]
        y_timestamp = pd.Series(pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq="B"))

        log(f"  Input: 60 K 線 (1101.TW {x_timestamp.iloc[0].date()} ~ {x_timestamp.iloc[-1].date()})")
        log(f"  Output: 預測未來 5 天 ({y_timestamp.iloc[0].date()} ~ {y_timestamp.iloc[-1].date()})")
        log(f"  跑預測（首次需要 model warmup，~10-30 秒）...")

        import time
        t0 = time.time()
        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=5,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        elapsed = time.time() - t0
        log(f"  ✅ 預測完成（{elapsed:.1f} 秒）")
        log(f"")
        log(f"  歷史最後一天 close = {x_df['close'].iloc[-1]:.2f}")
        log(f"  預測未來 5 天 close: {pred['close'].tolist()}")
        log(f"  → 次日 close vs 今日：{pred['close'].iloc[0]:.2f} vs {x_df['close'].iloc[-1]:.2f} ({(pred['close'].iloc[0]/x_df['close'].iloc[-1]-1)*100:+.2f}%)")
    except Exception as e:
        log(f"  ❌ smoke test 失敗：{e}")
        import traceback
        log(traceback.format_exc())
        return

    log(f"\n✅ V38 Kronos 環境準備完成！")
    log(f"   Kronos 路徑：{KRONOS_DIR}")
    log(f"   Device：{device}")
    log(f"   下一步：python sanity_test_kronos.py")


if __name__ == "__main__":
    main()
