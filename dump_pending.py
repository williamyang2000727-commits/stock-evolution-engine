"""印 pending_push.json 的交易明細（依買入日排序）。

用法：python dump_pending.py
"""
import json, os, sys

PENDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_push.json")

if not os.path.exists(PENDING):
    print("pending_push.json 不存在")
    sys.exit(1)

d = json.load(open(PENDING, encoding="utf-8"))
trades = sorted(d.get("trade_details", []), key=lambda t: t.get("buy_date", ""))
if not trades:
    print("無交易")
    sys.exit(0)

# 分年統計
years = {}
for t in trades:
    if t.get("reason") == "持有中":
        continue
    y = t.get("buy_date", "????")[:4]
    years.setdefault(y, {"n": 0, "ret": 0, "win": 0, "loss": 0})
    years[y]["n"] += 1
    years[y]["ret"] += t.get("return", 0)
    if t.get("return", 0) > 0:
        years[y]["win"] += 1
    else:
        years[y]["loss"] += 1

print("=" * 95)
print(f"  pending 策略交易明細（共 {len(trades)} 筆，依買入日排序）")
print("=" * 95)
print(f"  {'買入日':10} {'賣出日':10} {'代號':10} {'名稱':7} {'買價':>7} {'賣價':>7} {'報酬':>7} {'天數':>4} 出場原因")
print("-" * 95)
for t in trades:
    bd = t.get("buy_date", "")
    sd = t.get("sell_date") or "持有中"
    ticker = t.get("ticker", "")
    name = t.get("name", "")
    bp = t.get("buy_price", 0)
    sp = t.get("sell_price", 0)
    ret = t.get("return", 0)
    days = t.get("days", 0)
    reason = t.get("reason", "")
    marker = "⭐" if ret > 30 else ("🔥" if ret > 15 else ("❌" if ret < -10 else "  "))
    print(f"  {bd:10} {sd:10} {ticker:10} {name:7} {bp:>7.2f} {sp:>7.2f} {ret:>+6.1f}% {days:>3}天 {reason} {marker}")

print("-" * 95)
print()
print("  分年統計：")
for y, s in sorted(years.items()):
    avg = s["ret"] / s["n"] if s["n"] else 0
    wr = s["win"] / s["n"] * 100 if s["n"] else 0
    print(f"    {y}：{s['n']:>2} 筆  avg {avg:>+5.1f}%  總 {s['ret']:>+5.0f}%  勝 {s['win']}/{s['loss']+s['win']} = {wr:.0f}%")

# 出場原因分布
print()
print("  出場原因分布：")
reasons = {}
completed = [t for t in trades if t.get("reason") != "持有中"]
for t in completed:
    r = t.get("reason", "?")
    reasons.setdefault(r, {"n": 0, "ret": 0, "win": 0})
    reasons[r]["n"] += 1
    reasons[r]["ret"] += t.get("return", 0)
    if t.get("return", 0) > 0:
        reasons[r]["win"] += 1
for r, s in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
    avg = s["ret"] / s["n"]
    wr = s["win"] / s["n"] * 100
    print(f"    {r:20}：{s['n']:>2} 筆  avg {avg:>+5.1f}%  勝率 {wr:.0f}%")

# 最大贏家/輸家 top 5
print()
completed_sorted = sorted(completed, key=lambda t: -t.get("return", 0))
print("  最大贏家 Top 5：")
for t in completed_sorted[:5]:
    print(f"    {t.get('buy_date')} {t.get('ticker'):10} {t.get('name',''):7} {t.get('return',0):>+6.1f}% ({t.get('days',0)}天, {t.get('reason','')})")
print("  最大輸家 Top 5：")
for t in completed_sorted[-5:]:
    print(f"    {t.get('buy_date')} {t.get('ticker'):10} {t.get('name',''):7} {t.get('return',0):>+6.1f}% ({t.get('days',0)}天, {t.get('reason','')})")

# 持倉
holding = [t for t in trades if t.get("reason") == "持有中"]
if holding:
    print()
    print(f"  持倉中 {len(holding)} 檔：")
    for t in holding:
        print(f"    {t.get('buy_date')} {t.get('ticker'):10} {t.get('name',''):7} 買{t.get('buy_price')} 現{t.get('sell_price')} {t.get('return',0):>+6.1f}% ({t.get('days',0)}天)")
