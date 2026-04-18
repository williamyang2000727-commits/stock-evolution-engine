"""在推 pending_push.json 到 Gist 之前，用 cpu_replay 驗證一次。

用法：python verify_pending.py

跟 realistic_test.py 一樣是用 cpu_replay + 當前 stock_data_cache.pkl，
但讀的是 pending_push.json（GPU 剛存的）而不是 best_strategy.json（Gist 同步的）。

輸出：驗證結果是否跟 Telegram 推播一致。
"""
import json, os, sys, types
mock_cp = types.ModuleType('cupy')
mock_cp.RawKernel = lambda *a, **k: None
sys.modules['cupy'] = mock_cp
from gpu_cupy_evolve import precompute, cpu_replay, download_data

HERE = os.path.dirname(os.path.abspath(__file__))
PENDING = os.path.join(HERE, "pending_push.json")

if not os.path.exists(PENDING):
    print("❌ 沒有 pending_push.json")
    sys.exit(1)

pending = json.load(open(PENDING, encoding="utf-8"))
p = pending["params"]
claimed = pending.get("backtest", {})
print(f"Pending 宣稱：score={pending.get('score')}, {claimed.get('total_trades')}筆 總{claimed.get('total_return')}% 勝率{claimed.get('win_rate')}%")

data = download_data()
TARGET_DAYS = 900
data = {k: v.tail(TARGET_DAYS) for k, v in data.items() if len(v) >= TARGET_DAYS}
pre = precompute(data)
print(f"資料：{pre['n_stocks']} 檔 × {pre['n_days']} 天")
print(f"期間：{pre['dates'][0].date()} ~ {pre['dates'][-1].date()}")
print(f"WF 切點：day {pre['train_end']}（train {pre['train_end']-60} 天 / test {pre['n_days']-pre['train_end']} 天）")

trades = cpu_replay(pre, p)
import math
trades = [t for t in trades if not math.isnan(t.get("return",0))]
completed = [t for t in trades if t.get("reason") != "持有中"]
holding = [t for t in trades if t.get("reason") == "持有中"]

n = len(completed)
if n == 0:
    print("❌ 無交易")
    sys.exit()

total = sum(t["return"] for t in completed)
avg = total / n
wins = sum(1 for t in completed if t["return"] > 0)
wr = wins/n*100

# WF 分析（反向 WF：train 在新，test 在舊）
train_start_date = pre["dates"][pre["train_start"]]
train_t = [t for t in completed if t["buy_date"] >= str(train_start_date.date())]
test_t = [t for t in completed if t["buy_date"] < str(train_start_date.date())]
train_total = sum(t["return"] for t in train_t)
test_total = sum(t["return"] for t in test_t)
_tr_days = pre["train_end"] - pre["train_start"]
_ts_days = pre["train_start"] - 60
train_ann = train_total / (_tr_days/250.0) if _tr_days > 125 else train_total
test_ann = test_total / (_ts_days/250.0) if _ts_days > 75 else test_total
wf_ratio = test_ann/train_ann*100 if train_ann > 0 else 0

# MaxDD
run_dd = 0; max_dd = 0
for t in completed:
    r = t["return"]
    if r < 0: run_dd += r
    else: run_dd = 0
    if run_dd < max_dd: max_dd = run_dd

print(f"\n{'='*60}")
print(f"  ✅ cpu_replay 驗證結果")
print(f"{'='*60}")
print(f"  筆數       : {n}  (Telegram 宣稱 {claimed.get('total_trades')})")
print(f"  總報酬     : {total:.2f}%  (宣稱 {claimed.get('total_return')}%)")
print(f"  平均/筆    : {avg:.2f}%  (宣稱 {claimed.get('avg_return')}%)")
print(f"  勝率       : {wr:.1f}%  (宣稱 {claimed.get('win_rate')}%)")
print(f"  MaxDD      : {max_dd:.1f}%")
print(f"  持有中     : {len(holding)} 檔")
print(f"\n  WF 驗證：")
print(f"    train    : {len(train_t)}筆 {train_total:.0f}% (年化 {train_ann:.0f}%)")
print(f"    test     : {len(test_t)}筆 {test_total:.0f}% (年化 {test_ann:.0f}%)")
print(f"    WF 比例  : {wf_ratio:.0f}%  (需 ≥ 50%)")

# 分年
years = {}
for t in completed:
    y = t["buy_date"][:4]
    if y not in years: years[y] = []
    years[y].append(t)
print(f"\n  分年績效：")
for y in sorted(years.keys()):
    ts = years[y]; yn=len(ts); yt=sum(t["return"] for t in ts); ya=yt/yn; yw=sum(1 for t in ts if t["return"]>0)
    print(f"    {y}: {yn}筆 avg{ya:.1f}% 總{yt:.0f}% 勝率{yw/yn*100:.0f}%")

print(f"\n  {'✅ 數字一致' if abs(n - claimed.get('total_trades',0)) <= 1 and abs(total - claimed.get('total_return',0)) < 1 else '⚠️ 數字不一致！'}")
print(f"\n  審核 OK → python push_pending.py")
