"""day index 1495 是哪一天？"""
import os, sys, types, json, urllib.request
mock_cp = types.ModuleType("cupy")
mock_cp.RawKernel = lambda *a, **k: None
sys.modules["cupy"] = mock_cp
import pandas as pd
from gpu_cupy_evolve import precompute, download_data

data = download_data()
TARGET = 1500
data_t = {k: v.tail(TARGET) for k, v in data.items() if len(v) >= TARGET}
pre = precompute(data_t)
dates = pre["dates"]

# 印 1490 ~ 1499 是哪些日期
for i in range(1490, 1500):
    d = dates[i]
    d_str = pd.Timestamp(d).strftime("%Y-%m-%d %a")
    print(f"  index {i}: {d_str}")
