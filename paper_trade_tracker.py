"""
V38 Paper Trading Tracker — 雙軌記錄 89.90 vs V38 (89.90 + Kronos filter)

用途：1-2 週 forward test，比較
  Track A: 89.90 不過濾選股（current 上線策略）
  Track B: 89.90 + Kronos filter 過濾後選股（V38 候選）

每天執行：
  1. 跑 89.90 daily_scan 拿候選 X
  2. 對 X 跑 Kronos filter
  3. 記錄到 paper_trade_log.json：
     {
       "date": "2026-04-26",
       "track_A_buy": "2330.TW",        # 89.90 想買的
       "track_B_decision": True/False,  # V38 同意嗎
       "kronos_pred_next_pct": +1.2,
       "kronos_pred_5d_pct": +3.4,
       "actual_next_close": null,       # 隔天填
       "actual_5d_close": null,         # 5 天後填
     }

每週 review：
  - track A 跟 B 的 wr 比較
  - track B skip 的天，A 真實 return 是賺是賠
  - 如果 B 真的擋掉爛筆 → 上線

用法：
  C:\\stock-evolution> python paper_trade_tracker.py [scan|review|fill]
    scan   : 今天執行（跑 89.90 + Kronos，記錄到 log）
    review : 看當前 log 統計（A vs B wr / total）
    fill   : 用最新 cache 填過去 trades 的 actual return
"""
import os, sys, json, pickle
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
KRONOS_DIR = os.path.join(USER_SE, "Kronos")
if KRONOS_DIR not in sys.path: sys.path.insert(0, KRONOS_DIR)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if USER_SE not in sys.path: sys.path.insert(0, USER_SE)

LOG_PATH = os.path.join(USER_SE, "paper_trade_log.json")


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"trades": []}


def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def get_today_89_90_pick():
    """
    拿 89.90 今天會選的股票（用 cache + cpu_replay 模擬到最後一天）
    Returns: dict 或 None
    """
    import gpu_cupy_evolve as base

    GPU_GIST_ID = "c1bef892d33589baef2142ce250d18c2"
    r = urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/gists/{GPU_GIST_ID}"), timeout=30)
    s = json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])
    params = s.get("params", s)

    cache = pickle.load(open(os.path.join(USER_SE, "stock_data_cache.pkl"), "rb"))
    _lens = [len(v) for v in cache.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data = {k: v.tail(TARGET) for k, v in cache.items() if len(v) >= TARGET}
    pre = base.precompute(data)

    # 跑 cpu_replay 找出最後一天 buy 的股票
    all_trades = base.cpu_replay(pre, params)
    # 取最近 5 天的 trades（包括 hold）
    last_date = pre["dates"][-1]
    recent = []
    for t in all_trades:
        bd_str = t.get("buy_date", "")
        if bd_str:
            bd = pd.to_datetime(bd_str)
            if (last_date - bd).days <= 5:
                recent.append(t)

    if recent:
        latest = max(recent, key=lambda x: x.get("buy_date", ""))
        return {
            "ticker": latest.get("ticker"),
            "name": latest.get("name", ""),
            "buy_date": latest.get("buy_date"),
            "buy_price": latest.get("buy_price"),
        }
    return None


def cmd_scan():
    """每天跑：拿 89.90 候選 + Kronos filter，記錄"""
    print("=" * 60)
    print("Paper Trade Tracker — Today's Scan")
    print("=" * 60)

    log = load_log()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 檢查今天有沒有跑過
    for t in log["trades"]:
        if t.get("scan_date") == today_str:
            print(f"⚠️ {today_str} 已掃過，skip")
            return

    print(f"\n[1/3] 89.90 找今日候選...")
    pick = get_today_89_90_pick()
    if not pick:
        print(f"  89.90 今天沒選股")
        log["trades"].append({
            "scan_date": today_str,
            "track_A_buy": None,
            "track_B_decision": None,
            "note": "89.90 no pick",
        })
        save_log(log)
        return

    print(f"  89.90 選: {pick['ticker']} ({pick['name']}) buy_date={pick['buy_date']} @{pick.get('buy_price')}")

    print(f"\n[2/3] Kronos filter 對該股檢查...")
    from kronos_filter import KronosFilter
    cache = pickle.load(open(os.path.join(USER_SE, "stock_data_cache.pkl"), "rb"))
    if pick["ticker"] not in cache:
        print(f"  ❌ {pick['ticker']} 不在 cache")
        return
    ohlcv = cache[pick["ticker"]]

    # 大盤 proxy
    all_closes = []
    for k, v in cache.items():
        if len(v) >= 100:
            all_closes.append(v["Close"].tail(100).values)
    market_close = np.array(all_closes).mean(axis=0) if all_closes else None

    kf = KronosFilter()
    decision = kf.should_buy(ohlcv, market_close_history=market_close)

    print(f"\n[3/4] V38 決策結果...")
    print(f"  buy = {decision['buy']}")
    print(f"  pred next-day = {decision['pred_next_pct']:+.2f}% (th {decision['threshold_next']})")
    print(f"  pred 5-day    = {decision['pred_5d_pct']:+.2f}% (th {decision['threshold_5d']:.2f})")
    print(f"  reason: {decision['reason']}")

    # === Track C: V38d (V38 + ML head) ===
    print(f"\n[4/4] V38d ML head 二次過濾...")
    track_c_decision = None
    track_c_proba = None
    track_c_reason = None
    try:
        # 先抽該股當天的 19 features
        import gpu_cupy_evolve as base
        from metalabel_features import extract_features_at
        from v38d_filter import V38dFilter, KRONOS_NEXT_TH

        # 跑 89.90 precompute 拿 features
        _lens = [len(v) for v in cache.values()]
        if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
        elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
        else: TARGET = 900
        data_dict = {k: v.tail(TARGET) for k, v in cache.items() if len(v) >= TARGET}
        pre = base.precompute(data_dict)

        # 找該股 buy_date 對應 day_idx
        dates = pre["dates"]
        date_to_day = {str(d.date() if hasattr(d, 'date') else d)[:10]: i for i, d in enumerate(dates)}
        bd_str = pick["buy_date"]
        if isinstance(bd_str, pd.Timestamp):
            bd_str = bd_str.strftime("%Y-%m-%d")
        elif not isinstance(bd_str, str):
            bd_str = str(bd_str)[:10]
        day_idx = date_to_day.get(bd_str)
        if day_idx is None:
            track_c_decision = False
            track_c_reason = f"找不到 buy_date {bd_str} 在 pre.dates"
        else:
            features_19d = extract_features_at(pre, pick["ticker"], day_idx)
            if features_19d is None:
                track_c_decision = False
                track_c_reason = "features_19d = None（NaN/Inf）"
            else:
                v38d = V38dFilter()
                d_v38d = v38d.should_buy(
                    ohlcv, market_close, today=None,
                    kronos_pred_next=decision["pred_next_pct"],
                    kronos_pred_5d=decision["pred_5d_pct"],
                    features_19d=features_19d,
                )
                track_c_decision = d_v38d["buy"]
                track_c_proba = d_v38d.get("ml_proba")
                track_c_reason = d_v38d["reason"]
                print(f"  V38d buy = {track_c_decision}")
                print(f"  ml_proba = {track_c_proba}")
                print(f"  reason: {track_c_reason}")
    except Exception as e:
        print(f"  ⚠️ V38d failed: {e}")
        track_c_reason = f"exception: {e}"

    # 寫 log
    entry = {
        "scan_date": today_str,
        "track_A_buy": pick["ticker"],
        "track_A_name": pick.get("name", ""),
        "track_A_buy_date": pick["buy_date"],
        "track_A_buy_price": pick.get("buy_price"),
        "track_B_decision": decision["buy"],
        "kronos_pred_next_pct": decision["pred_next_pct"],
        "kronos_pred_5d_pct": decision["pred_5d_pct"],
        "kronos_threshold_next": decision["threshold_next"],
        "kronos_threshold_5d": decision["threshold_5d"],
        "kronos_reason": decision["reason"],
        "track_C_decision": track_c_decision,
        "track_C_ml_proba": track_c_proba,
        "track_C_reason": track_c_reason,
        "actual_5d_close": None,  # fill 階段填
        "actual_5d_return_pct": None,
    }
    log["trades"].append(entry)
    save_log(log)
    print(f"\n✅ 記錄到 {LOG_PATH}")


def cmd_fill():
    """用最新 cache 填過去未填的 actual return"""
    print("=" * 60)
    print("Paper Trade Tracker — Fill actual returns")
    print("=" * 60)

    log = load_log()
    cache = pickle.load(open(os.path.join(USER_SE, "stock_data_cache.pkl"), "rb"))

    n_filled = 0
    for t in log["trades"]:
        if t.get("actual_5d_close") is not None:
            continue
        ticker = t.get("track_A_buy")
        buy_date_str = t.get("track_A_buy_date")
        buy_price = t.get("track_A_buy_price")
        if not ticker or not buy_date_str or not buy_price:
            continue
        if ticker not in cache:
            continue
        df = cache[ticker]
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        bd = pd.to_datetime(buy_date_str)
        # 找 buy_date 後 5 個交易日
        mask = df.index > bd
        future = df[mask].head(5)
        if len(future) < 5:
            continue
        close_5d = float(future["Close"].iloc[-1])
        ret_pct = (close_5d / float(buy_price) - 1) * 100
        t["actual_5d_close"] = close_5d
        t["actual_5d_return_pct"] = ret_pct
        n_filled += 1

    save_log(log)
    print(f"填了 {n_filled} 筆")


def cmd_review():
    """看當前 log 統計"""
    print("=" * 60)
    print("Paper Trade Tracker — Review")
    print("=" * 60)

    log = load_log()
    n_total = len(log["trades"])
    n_filled = sum(1 for t in log["trades"] if t.get("actual_5d_return_pct") is not None)
    print(f"\n總紀錄 {n_total} 筆，已填 actual return {n_filled}")

    if n_filled < 1:
        print("還沒夠資料 review，繼續累積")
        return

    # Track A：所有 89.90 買的
    a_rets = [t["actual_5d_return_pct"] for t in log["trades"] if t.get("actual_5d_return_pct") is not None]
    a_wr = sum(1 for r in a_rets if r > 0) / len(a_rets) * 100 if a_rets else 0
    a_total = sum(a_rets)
    a_avg = np.mean(a_rets) if a_rets else 0

    # Track B：Kronos buy=True 的（V38 會買的）
    b_rets = [t["actual_5d_return_pct"] for t in log["trades"]
              if t.get("track_B_decision") == True and t.get("actual_5d_return_pct") is not None]
    b_wr = sum(1 for r in b_rets if r > 0) / len(b_rets) * 100 if b_rets else 0
    b_total = sum(b_rets)
    b_avg = np.mean(b_rets) if b_rets else 0

    # Track B 拒絕的（V38 skip 的，看 89.90 會虧多少）
    skip_rets = [t["actual_5d_return_pct"] for t in log["trades"]
                if t.get("track_B_decision") == False and t.get("actual_5d_return_pct") is not None]
    skip_wr = sum(1 for r in skip_rets if r > 0) / len(skip_rets) * 100 if skip_rets else 0
    skip_avg = np.mean(skip_rets) if skip_rets else 0

    # Track C：V38d (V38 + ML head)
    c_rets = [t["actual_5d_return_pct"] for t in log["trades"]
              if t.get("track_C_decision") == True and t.get("actual_5d_return_pct") is not None]
    c_wr = sum(1 for r in c_rets if r > 0) / len(c_rets) * 100 if c_rets else 0
    c_total = sum(c_rets)
    c_avg = np.mean(c_rets) if c_rets else 0

    print(f"\n=== Track A: 89.90 全買 ===")
    print(f"  n = {len(a_rets)}, wr = {a_wr:.1f}%, avg = {a_avg:+.2f}%, total = {a_total:+.2f}%")

    print(f"\n=== Track B: 89.90 + Kronos filter（V38）===")
    print(f"  n = {len(b_rets)}, wr = {b_wr:.1f}%, avg = {b_avg:+.2f}%, total = {b_total:+.2f}%")

    print(f"\n=== Track C: V38d (V38 + ML head) ===")
    print(f"  n = {len(c_rets)}, wr = {c_wr:.1f}%, avg = {c_avg:+.2f}%, total = {c_total:+.2f}%")

    print(f"\n=== Track B 擋下的 ===")
    print(f"  n = {len(skip_rets)}, wr = {skip_wr:.1f}%, avg = {skip_avg:+.2f}%")
    print(f"  （這些是 V38 認為會虧的，看實際真的虧嗎）")

    print(f"\n=== 對比 ===")
    if len(b_rets) >= 5 and len(a_rets) >= 5:
        wr_diff = b_wr - a_wr
        avg_diff = b_avg - a_avg
        print(f"  V38 wr 比 89.90 高 {wr_diff:+.1f}%")
        print(f"  V38 avg 比 89.90 高 {avg_diff:+.2f}%")
        if wr_diff > 5 and len(b_rets) >= 10:
            print(f"  🟢 V38 顯著勝出，可考慮上線")
        elif wr_diff > 0:
            print(f"  🟡 V38 略好，繼續累積資料")
        else:
            print(f"  🔴 V38 沒贏，可能要重新評估")
    else:
        print(f"  資料不夠（B {len(b_rets)} A {len(a_rets)}），繼續累積")

    if len(c_rets) >= 3:
        cb_wr_diff = c_wr - b_wr if len(b_rets) >= 1 else None
        cb_avg_diff = c_avg - b_avg if len(b_rets) >= 1 else None
        print(f"\n=== V38d vs V38 對比 ===")
        if cb_wr_diff is not None:
            print(f"  V38d wr 比 V38 {cb_wr_diff:+.1f}%")
            print(f"  V38d avg 比 V38 {cb_avg_diff:+.2f}%")
        if len(c_rets) >= 10 and cb_wr_diff and cb_wr_diff > 5:
            print(f"  🟢 V38d 顯著勝 V38（n={len(c_rets)} >= 10）")
        elif len(c_rets) < 10:
            print(f"  📝 V38d 樣本數 {len(c_rets)}/10，繼續累積")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "scan":
        cmd_scan()
    elif cmd == "fill":
        cmd_fill()
    elif cmd == "review":
        cmd_review()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Usage: python paper_trade_tracker.py [scan|review|fill]")


if __name__ == "__main__":
    main()
