r"""
測試 cpu_replay 對 cache 端點滑動的敏感度

用同一份 cache，分別取 tail(1500) / tail(1499) / tail(1501) 跑 cpu_replay
看末尾「持有中」會不會不一樣
"""
import os, sys, json, types, urllib.request, pickle
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "stock-evolution"))
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import numpy as np
from gpu_cupy_evolve import precompute, cpu_replay, download_data


def fetch_strategy():
    r = urllib.request.urlopen(
        urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"),
        timeout=30,
    )
    return json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])


data = download_data()
strategy = fetch_strategy()
p = strategy.get("params", strategy)


def run_with_target(target_days):
    """用 tail(target_days) 跑一次，回末尾持有中 + 統計"""
    data_t = {k: v.tail(target_days) for k, v in data.items() if len(v) >= target_days}
    pre = precompute(data_t)
    trades = cpu_replay(pre, p)
    holding = [t for t in trades if t.get("reason") == "持有中"]
    completed = [t for t in trades if t.get("reason") != "持有中"]
    return {
        "n_completed": len(completed),
        "total_return": sum(t.get("return", 0) for t in completed),
        "holdings": [(h.get("ticker"), h.get("name"), h.get("buy_date"), h.get("buy_price")) for h in holding],
    }


print("=" * 70)
print("測試 cache 端點滑動對「持有中」的影響")
print("=" * 70)

# 測 5 個不同 target
targets = [1450, 1480, 1499, 1500, 1501, 1520, 1550]
results = {}
for t in targets:
    print(f"\n--- TARGET_DAYS = {t} ---")
    r = run_with_target(t)
    results[t] = r
    print(f"  完成 {r['n_completed']} 筆, 總報酬 {r['total_return']:+.1f}%")
    print(f"  持有中:")
    for h in r["holdings"]:
        print(f"    {h[1]} ({h[0]}) buy {h[2]} @ {h[3]}")

# 比對結果
print("\n" + "=" * 70)
print("【穩定性分析】")
print("=" * 70)
all_holdings = [tuple(sorted(t[0] for t in r["holdings"])) for r in results.values()]
unique_holding_sets = set(all_holdings)
if len(unique_holding_sets) == 1:
    print(f"✅ 所有 {len(targets)} 種 target_days 持有中**完全一致**")
    print(f"   → cpu_replay 對 cache 端點滑動不敏感（穩定）")
else:
    print(f"❌ 出現 {len(unique_holding_sets)} 種不同的持有組合：")
    for t, r in results.items():
        tks = sorted(h[0] for h in r["holdings"])
        print(f"   target={t}: {tks}")
    print(f"\n   → cpu_replay 對 cache 端點敏感，有 bug 風險")
    print(f"   → 每天 rebuild_tab3 跑出來持股可能不同 → Web 顯示亂跳")
