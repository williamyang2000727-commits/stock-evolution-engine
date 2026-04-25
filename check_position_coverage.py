"""
驗證 89.90 持倉時序：到底有沒有空倉天 / 只有 1 檔的天
William 說「不可能有空倉的時候 也基本不會只有一檔」 — 確認看看
"""
import os, sys, json, pickle
import pandas as pd
import numpy as np

USER_SE = os.path.join(os.path.expanduser("~"), "stock-evolution")
if not os.path.isdir(USER_SE):
    USER_SE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, USER_SE)
import gpu_cupy_evolve as base


def fetch_gist_strategy():
    import urllib.request
    r = urllib.request.urlopen(urllib.request.Request("https://api.github.com/gists/c1bef892d33589baef2142ce250d18c2"), timeout=30)
    s = json.loads(json.loads(r.read())["files"]["best_strategy.json"]["content"])
    return s.get("params", s)


def main():
    print("=" * 70)
    print("89.90 持倉時序驗證")
    print("=" * 70)
    raw = pickle.load(open(os.path.join(USER_SE, "stock_data_cache.pkl"), "rb"))
    params = fetch_gist_strategy()
    _lens = [len(v) for v in raw.values()]
    if sum(1 for l in _lens if l >= 1500) >= 500: TARGET = 1500
    elif sum(1 for l in _lens if l >= 1200) >= 800: TARGET = 1200
    else: TARGET = 900
    data_dict = {k: v.tail(TARGET) for k, v in raw.items() if len(v) >= TARGET}
    pre = base.precompute(data_dict)
    trades = base.cpu_replay(pre, params)
    completed = [t for t in trades if t.get("sell_date")]
    print(f"\n  共 {len(completed)} 筆完成交易")
    print(f"  max_pos = {params.get('max_pos', 2)}")

    # 用 pre['dates'] 重建時序
    dates = pre.get("dates")
    if dates is None:
        # 抓一個 ticker 的 index
        for k in data_dict:
            dates = data_dict[k].index
            break
    dates_norm = pd.DatetimeIndex(dates).tz_localize(None) if hasattr(dates[0], "tz") and dates[0].tz else pd.DatetimeIndex(dates)
    n_days = len(dates_norm)
    print(f"  期間：{dates_norm[0].date()} ~ {dates_norm[-1].date()} ({n_days} 交易日)")

    # 對每一交易日算當天有幾個持倉
    pos_count = np.zeros(n_days, dtype=int)
    date_to_idx = {d.normalize(): i for i, d in enumerate(dates_norm)}

    for t in completed:
        bd = pd.Timestamp(t["buy_date"]).normalize()
        sd = pd.Timestamp(t["sell_date"]).normalize()
        i_b = date_to_idx.get(bd)
        i_s = date_to_idx.get(sd)
        if i_b is None or i_s is None:
            continue
        # 持倉期間：buy_date 到 sell_date 前一天（sell 當天賣掉算空）
        for i in range(i_b, i_s):
            pos_count[i] += 1

    # 統計
    n_zero = (pos_count == 0).sum()
    n_one = (pos_count == 1).sum()
    n_two = (pos_count == 2).sum()
    n_more = (pos_count >= 3).sum()
    total = n_days
    print(f"\n{'─' * 70}")
    print("【每日持倉檔數分布】")
    print(f"{'─' * 70}")
    print(f"  0 檔（空倉）  ：{n_zero:4d} 天 ({n_zero/total*100:5.1f}%)")
    print(f"  1 檔         ：{n_one:4d} 天 ({n_one/total*100:5.1f}%)")
    print(f"  2 檔（滿倉） ：{n_two:4d} 天 ({n_two/total*100:5.1f}%)")
    if n_more:
        print(f"  3+ 檔（異常）：{n_more:4d} 天 ({n_more/total*100:5.1f}%)")

    # 找最長空倉/單檔
    print(f"\n{'─' * 70}")
    print("【最長空倉期 + 單檔期】")
    print(f"{'─' * 70}")

    def longest_run(arr, target):
        max_len, max_end = 0, -1
        cur_len = 0
        for i, v in enumerate(arr):
            if v == target:
                cur_len += 1
                if cur_len > max_len:
                    max_len = cur_len; max_end = i
            else:
                cur_len = 0
        return max_len, max_end

    z_len, z_end = longest_run(pos_count, 0)
    o_len, o_end = longest_run(pos_count, 1)
    if z_len > 0:
        z_start = z_end - z_len + 1
        print(f"  最長空倉：{z_len} 天連續（{dates_norm[z_start].date()} ~ {dates_norm[z_end].date()}）")
    else:
        print(f"  最長空倉：0 天 ✅ 從未空倉")
    if o_len > 0:
        o_start = o_end - o_len + 1
        print(f"  最長單檔：{o_len} 天連續（{dates_norm[o_start].date()} ~ {dates_norm[o_end].date()}）")
    else:
        print(f"  最長單檔：0 天 ✅ 從未只剩 1 檔")

    print(f"\n{'─' * 70}")
    print("【結論】")
    print(f"{'─' * 70}")
    if n_zero == 0 and n_one == 0:
        print(f"  ✅✅✅ William 說的對：從來沒空倉、也從來沒只有 1 檔")
        print(f"     滿倉率 100%（{n_two}/{total} 天）")
    elif n_zero == 0:
        print(f"  ✅ 從來沒空倉，但有 {n_one} 天 ({n_one/total*100:.1f}%) 只有 1 檔")
        print(f"     主要是換股當天（賣 1 檔還沒買到下一檔）")
    elif n_one < total * 0.05 and n_zero < total * 0.05:
        print(f"  ✅ 基本上 William 講對：空倉 {n_zero/total*100:.1f}%、單檔 {n_one/total*100:.1f}%（都 < 5%）")
        print(f"     滿倉率 {n_two/total*100:.1f}%")
    else:
        print(f"  🟡 我前面說錯：空倉 {n_zero/total*100:.1f}%、單檔 {n_one/total*100:.1f}%、滿倉 {n_two/total*100:.1f}%")

    # 加值：如果是換股，buy 跟 sell 應該同一天 → 計算「先賣後買」邏輯有沒有讓單檔天減少
    print(f"\n{'─' * 70}")
    print("【補充：因為 89.90 是「先賣後買」，理論上換股當天會短暫單檔】")
    print(f"  但同日的「sell + buy」會把那一天算成兩檔過渡 — 實際取決於回測 simulator")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
