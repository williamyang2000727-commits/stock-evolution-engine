"""印 pending_push.json（或 .pushed）的策略完整買賣條件
用法：python show_params.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 先找 pending_push.json，沒有就找 .pushed
candidates = [os.path.join(HERE, "pending_push.json"),
              os.path.join(HERE, "pending_push.json.pushed")]
path = next((p for p in candidates if os.path.exists(p)), None)
if not path:
    print("❌ 找不到 pending_push.json 或 .pushed"); sys.exit(1)

print(f"[讀] {os.path.basename(path)}\n")
d = json.load(open(path, encoding="utf-8"))
p = d["params"]
score = d.get("score", "?")
bt = d.get("backtest", {})
print(f"策略分數: {score}")
if bt:
    print(f"回測: {bt.get('n_trades')} 筆 | avg {bt.get('avg_return'):.1f}% | 總 {bt.get('total_return'):.0f}% | 勝率 {bt.get('win_rate'):.0f}%")
print()

# ========== 買入條件 ==========
print("=" * 60)
print("買入條件（收盤分數 >= 門檻才買）")
print("=" * 60)
print(f"買入總分門檻：>= {p.get('buy_threshold')} 分（滿分約 40）")
print(f"MA 設定：fast={p.get('ma_fast_w')} | slow={p.get('ma_slow_w')} | momentum_days={p.get('momentum_days')}")
print()

indicators = [
    ("w_rsi",       "RSI 超買",       "rsi_th",        "≥"),
    ("w_bb",        "布林通道位置",     "bb_th",         "≥"),
    ("w_vol",       "爆量（vol/MA20）",  "vol_th",        "≥"),
    ("w_ma",        "MA 多頭排列",      None,            ""),
    ("w_macd",      "MACD",           "macd_mode",     "mode"),
    ("w_kd",        "KD 值",          "kd_th",         "≥"),
    ("w_wr",        "威廉指標 WR",    "wr_th",         "≥"),
    ("w_mom",       "動量 %",         "mom_th",        "≥"),
    ("w_near_high", "近新高 %",        "near_high_pct", "≤"),
    ("w_squeeze",   "Squeeze 擠壓發射", None,           ""),
    ("w_new_high",  "60 日新高",        None,           ""),
    ("w_adx",       "ADX 趨勢強度",    "adx_th",        "≥"),
    ("w_bias",      "BIAS 乖離 %",     "bias_max",      "≤"),
    ("w_obv",       "OBV 能量潮上升",   "obv_rising_days", "日"),
    ("w_atr",       "ATR 波動率 %",    "atr_min",       "≥"),
    ("w_up_days",   "連漲天數",        "up_days_min",   "≥"),
    ("w_week52",    "52 週位置 %",     "week52_min",    "≥"),
    ("w_vol_up_days","連量增天數",      "vol_up_days_min","≥"),
    ("w_mom_accel", "動量加速",         "mom_accel_min", "≥"),
    ("w_sector_flow","類股 RS 強度",    "sector_flow_topn","top"),
]

print("【加分指標】(w > 0 才計分)")
enabled = 0
for wkey, name, thkey, op in indicators:
    w = int(p.get(wkey, 0))
    if w > 0:
        enabled += 1
        if thkey:
            th = p.get(thkey, "?")
            th_str = f" {op} {th}" if op != "mode" else f" (mode={th})"
        else:
            th_str = ""
        print(f"  +{w} 分：{name}{th_str}")
if enabled == 0:
    print("  （全部關閉）")
print()

print("【單次旗標】(=1 才啟用)")
flags = [
    ("consecutive_green", "連 N 根紅 K"),
    ("gap_up",           "當日跳空 ≥ 1%"),
    ("above_ma60",       "收盤 ≥ MA60"),
    ("vol_gt_yesterday", "量 > 昨"),
]
any_flag = False
for f, n in flags:
    v = int(p.get(f, 0))
    if v > 0:
        any_flag = True
        print(f"  +{v}：{n}")
if not any_flag:
    print("  （全部關閉）")
print()

# ========== 賣出條件 ==========
print("=" * 60)
print("賣出條件（任一觸發就 D+1 開盤賣）")
print("=" * 60)

print(f"停損：{p.get('stop_loss')}%")
print(f"停利：+{p.get('take_profit')}%（啟用={p.get('use_take_profit', 1)}）")
t = p.get('trailing_stop', 0)
print(f"移動停利：{'關閉' if t == 0 else f'peak 回撤 {t}% 賣'}")
print(f"最長持有：{p.get('hold_days')} 交易日")

ub = p.get('use_breakeven', 0)
bt_trigger = p.get('breakeven_trigger', 0)
if ub:
    print(f"保本機制：啟用（peak 漲 ≥ {bt_trigger}% 後回到 0% 即賣）")
else:
    print("保本機制：關閉")

ma_mode = int(p.get('sell_below_ma', 0))
ma_names = {0: "關閉", 1: "跌破 MA5", 2: "跌破 MA20", 3: "跌破 MA60"}
print(f"跌破 MA 出場：{ma_names.get(ma_mode, '未知')}")

print()
print("【其他條件式出場】(use_X=1 才啟用)")
sell_flags = [
    ("use_rsi_sell",      "RSI 超買出場",  "rsi_sell"),
    ("use_macd_sell",     "MACD 死叉出場",  None),
    ("use_kd_sell",       "KD 死叉出場",    None),
    ("sell_vol_shrink",   "量縮出場",       None),
    ("use_stagnation_exit","停滯出場（N 天沒動）", "stagnation_days"),
    ("use_time_decay",    "漸進停利（每天要求 X%）", "ret_per_day"),
    ("use_profit_lock",   "鎖利出場",      "lock_trigger"),
    ("use_mom_exit",      "動量反轉出場",    "mom_exit_th"),
]
any_sell = False
for key, name, sub in sell_flags:
    v = int(p.get(key, 0)) if not isinstance(p.get(key), float) else float(p.get(key))
    if v:
        any_sell = True
        extra = f"（{sub}={p.get(sub)}）" if sub else ""
        print(f"  啟用：{name}{extra}")
if not any_sell:
    print("  （全部關閉 — 只靠上面 5 條核心賣出）")

print()
print("=" * 60)
print(f"其他設定")
print("=" * 60)
print(f"max_positions：{p.get('max_positions')} 檔（最多同時持倉）")
print(f"upgrade_margin：{p.get('upgrade_margin', 0)}（換股賣弱換強門檻，0=關閉）")
