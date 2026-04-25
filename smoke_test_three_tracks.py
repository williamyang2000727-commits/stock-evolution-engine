"""
Smoke test 三軌完整流程（dry-run，不寫 log）
用法：C:\\stock-evolution> python smoke_test_three_tracks.py

驗證：
  1. paper_trade_tracker.py 的 cmd_scan 邏輯能跑完
  2. Track A (89.90) → Track B (V38) → Track C (V38d) 三段都執行
  3. 印出每段結果，但不寫 paper_trade_log.json

跟 cmd_scan 差別：
  - 沒有「今天已掃過 skip」檢查
  - 不寫 log（只 print）
"""
import os, sys, pickle, json
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
KRONOS_DIR = os.path.join(USER_SE, "Kronos")
if KRONOS_DIR not in sys.path: sys.path.insert(0, KRONOS_DIR)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)


def main():
    print("=" * 60)
    print("三軌 Smoke Test（dry-run，不寫 log）")
    print("=" * 60)

    # 模擬 paper_trade_tracker.cmd_scan 的流程
    print(f"\n[1/4] 89.90 找今日候選...")
    from paper_trade_tracker import get_today_89_90_pick
    pick = get_today_89_90_pick()
    if not pick:
        print(f"  ❌ 89.90 沒選股，三軌測試無法繼續")
        return
    print(f"  Track A: 89.90 選 {pick['ticker']} ({pick['name']}) buy_date={pick['buy_date']} @{pick.get('buy_price')}")

    print(f"\n[2/4] Kronos filter...")
    from kronos_filter import KronosFilter
    cache = pickle.load(open(os.path.join(USER_SE, "stock_data_cache.pkl"), "rb"))
    if pick["ticker"] not in cache:
        print(f"  ❌ {pick['ticker']} 不在 cache")
        return
    ohlcv = cache[pick["ticker"]]
    all_closes = []
    for k, v in cache.items():
        if len(v) >= 100:
            all_closes.append(v["Close"].tail(100).values)
    market_close = np.array(all_closes).mean(axis=0) if all_closes else None

    kf = KronosFilter()
    decision = kf.should_buy(ohlcv, market_close_history=market_close)
    print(f"  Track B (V38): buy={decision['buy']}")
    print(f"    pred_next={decision['pred_next_pct']:+.2f}% (th {decision['threshold_next']})")
    print(f"    pred_5d  ={decision['pred_5d_pct']:+.2f}% (th {decision['threshold_5d']:.2f})")
    print(f"    reason: {decision['reason']}")

    print(f"\n[3/4] V38d ML head 二次過濾...")
    try:
        import gpu_cupy_evolve as base
        from metalabel_features import extract_features_at
        from v38d_filter import V38dFilter

        _lens = [len(v) for v in cache.values()]
        if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
        elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
        else: TARGET = 900
        data_dict = {k: v.tail(TARGET) for k, v in cache.items() if len(v) >= TARGET}
        pre = base.precompute(data_dict)

        dates = pre["dates"]
        date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}
        bd_str = pick["buy_date"]
        if isinstance(bd_str, pd.Timestamp):
            bd_str = bd_str.strftime("%Y-%m-%d")
        elif not isinstance(bd_str, str):
            bd_str = str(bd_str)[:10]
        day_idx = date_to_day.get(bd_str)

        if day_idx is None:
            print(f"  ❌ Track C: buy_date {bd_str} 找不到 day_idx")
        else:
            features_19d = extract_features_at(pre, pick["ticker"], day_idx)
            if features_19d is None:
                print(f"  ❌ Track C: features_19d = None（NaN/Inf）")
            else:
                v38d = V38dFilter()
                d_v38d = v38d.should_buy(
                    ohlcv, market_close, today=None,
                    kronos_pred_next=decision["pred_next_pct"],
                    kronos_pred_5d=decision["pred_5d_pct"],
                    features_19d=features_19d,
                )
                print(f"  Track C (V38d): buy={d_v38d['buy']}")
                print(f"    v38_pass={d_v38d['v38_pass']}")
                print(f"    ml_proba={d_v38d.get('ml_proba')}")
                print(f"    reason: {d_v38d['reason']}")
    except Exception as e:
        import traceback
        print(f"  ❌ Track C exception: {e}")
        print(traceback.format_exc())
        return

    print(f"\n[4/4] 三軌全部執行完畢 ✅")
    print(f"  Track A: {pick['ticker']}（89.90 候選）")
    print(f"  Track B (V38): {'✅ buy' if decision['buy'] else '❌ skip'}")
    print(f"  Track C (V38d): 看上面結果")
    print()
    print(f"如果三軌都印出來 → 4/27 排程跑會成功記錄三軌")


if __name__ == "__main__":
    main()
