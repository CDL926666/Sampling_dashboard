#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# Step-2 : Trend / Changepoint / Outlier
# 读取 : ch4_sampling_result/agg.csv  +  ts_df.csv
# 输出 : ch4_sampling_result/agg_analysis.csv
# ----------------------------------------------------------------
from __future__ import annotations

import time, warnings
import pandas as pd
import numpy as np

# ── 第三方依赖 ──────────────────────────────────────────────
try:
    import pymannkendall as mk                      # 趋势检验
    _probe = mk.original_test(np.arange(10))
    HAVE_SLOPE = hasattr(_probe, "slope")
except Exception as e:                              # 无库或低版本
    warnings.warn(f"[Trend] pymannkendall unavailable ({e}); "
                  "trend_slope 将设为 NaN")
    mk, HAVE_SLOPE = None, False

from scipy.stats import median_abs_deviation, zscore
from statsmodels.tsa.seasonal import STL            # 去季节
import ruptures as rpt                              # 突变点

# ── 全局参数 ───────────────────────────────────────────────
OUTPUT_DIR   = "ch4_sampling_result"
MIN_OBS_TREND= 5
MIN_OBS_CPT  = 8
TREND_ALPHA  = 0.05

# ──────────────────────────────────────────────────────────
def tic(msg: str) -> float:
    t0 = time.perf_counter()
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg} …", flush=True)
    return t0

def toc(t0: float, msg: str) -> None:
    print(f"    -> {msg} done  {time.perf_counter() - t0:.2f} s", flush=True)

# ── STL 去季节 ────────────────────────────────────────────
def stl_detrend(y: np.ndarray, period: int) -> tuple[np.ndarray, float]:
    if period > 1 and len(y) >= period * 2:
        res = STL(y, period=period, robust=True).fit()
        seasonal = res.seasonal
        return y - seasonal, float(np.std(seasonal) / (np.std(y) + 1e-9))
    return y, np.nan

# ── 数据读取 ───────────────────────────────────────────────
agg   = pd.read_csv(f"{OUTPUT_DIR}/agg.csv",   low_memory=False)
ts_df = pd.read_csv(f"{OUTPUT_DIR}/ts_df.csv", low_memory=False)

t0 = tic("Step-2 : Trend / CPT / Outlier")

# ---- 占位列（dtype 明确） --------------------------------
agg["trend_slope"]    = np.nan
agg["trend_sig"]      = False
agg["changepoint_day"]= pd.Series([pd.NA]*len(agg), dtype="object")
agg["outlier_count"]  = 0
agg["mad_stat"]       = np.nan

# ── 主循环 ────────────────────────────────────────────────
for i, row in agg.iterrows():
    sub = ts_df[(ts_df.latitude == row.latitude) &
                (ts_df.longitude == row.longitude)]
    if sub.empty or pd.isna(sub.iloc[0].time_series):
        continue

    pairs = [s.split(",") for s in sub.iloc[0].time_series.split("|")]
    if len(pairs) < MIN_OBS_TREND:
        continue

    y      = np.array([float(v) for _, v in pairs])
    dates  = np.array([pd.to_datetime(d) for d, _ in pairs])

    # —— 去季节 —— #
    per    = min(12, len(y)//2) if len(y) >= 12 else max(2, len(y)//2)
    y_det, _ = stl_detrend(y, per)

    # —— 趋势 Mann-Kendall —— #
    if mk is not None:
        try:
            res = mk.original_test(y_det)
            agg.at[i, "trend_slope"] = res.slope if HAVE_SLOPE else np.nan
            agg.at[i, "trend_sig"]   = (res.trend != "no trend") and (res.p < TREND_ALPHA)
        except Exception as e:
            warnings.warn(f"[Trend] MK failed @grid {row.grid_id}: {e}")

    # —— 突变点 (ruptures-Pelt,RBF) —— #
    if len(y) >= MIN_OBS_CPT:
        try:
            cps = rpt.Pelt(model="rbf").fit(y).predict(pen=1)
            if len(cps) > 1:                           # 最后一位是 len(y)
                agg.at[i, "changepoint_day"] = dates[cps[0]-1].strftime("%Y-%m-%d")
        except Exception as e:
            warnings.warn(f"[CPT] RPT failed @grid {row.grid_id}: {e}")

    # —— 异常值统计 (MAD + Z-score) —— #
    try:
        med   = np.median(y)
        mad_v = median_abs_deviation(y, scale="normal")
        mad_z = 0.6745 * (y - med) / (mad_v + 1e-8)
        zs    = np.abs(zscore(y)) if len(y) > 1 else np.zeros_like(y)
        n_out = np.maximum(np.abs(mad_z) > 3.5, zs > 3).sum()
        agg.at[i, "outlier_count"] = int(n_out)
        agg.at[i, "mad_stat"]      = mad_v
    except Exception as e:
        warnings.warn(f"[Outlier] failed @grid {row.grid_id}: {e}")

# ---- 保存 ------------------------------------------------
agg["trend_slope"] = pd.to_numeric(agg["trend_slope"], errors="coerce")
agg["trend_sig"]   = agg["trend_sig"].astype(bool)

agg.to_csv(f"{OUTPUT_DIR}/agg_analysis.csv", index=False)

toc(t0, "Step-2")
print("Trend / CPT / Outlier → agg_analysis.csv", flush=True)
