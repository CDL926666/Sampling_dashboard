# Y:\Bishe_project\PY3\step2_analysis.py

import time
import pandas as pd
import numpy as np
import warnings

# 检查趋势分析相关依赖
try:
    import pymannkendall as mk
    _probe = mk.original_test(np.arange(10))
    HAVE_SLOPE = hasattr(_probe, "slope")
except Exception as e:
    warnings.warn(f"[Trend] pymannkendall unavailable ({e})")
    mk, HAVE_SLOPE = None, False

from scipy.stats import median_abs_deviation, zscore
from statsmodels.tsa.seasonal import STL
import ruptures as rpt

DEFAULT_OUTPUT_DIR = "ch4_sampling_result"
DEFAULT_MIN_OBS_TREND = 5
DEFAULT_MIN_OBS_CPT = 8
DEFAULT_TREND_ALPHA = 0.05

# STL去趋势
def stl_detrend(y, period):
    if period > 1 and len(y) >= period * 2:
        res = STL(y, period=period, robust=True).fit()
        seasonal = res.seasonal
        return y - seasonal, float(np.std(seasonal) / (np.std(y) + 1e-9))
    return y, np.nan

# 单格点趋势分析
def analyze_grid_trend(
    ts,
    min_obs_trend=DEFAULT_MIN_OBS_TREND,
    min_obs_cpt=DEFAULT_MIN_OBS_CPT,
    trend_alpha=DEFAULT_TREND_ALPHA
):
    result = {
        "trend_slope": np.nan,
        "trend_sig": False,
        "changepoint_day": pd.NA,
        "outlier_count": 0,
        "mad_stat": np.nan
    }
    if ts is None or pd.isna(ts):
        return result
    pairs = [s.split(",") for s in ts.split("|") if "," in s]
    if len(pairs) < min_obs_trend:
        return result
    y = np.array([float(v) for _, v in pairs])
    dates = np.array([pd.to_datetime(d) for d, _ in pairs])
    per = min(12, len(y)//2) if len(y) >= 12 else max(2, len(y)//2)
    y_det, _ = stl_detrend(y, per)
    if mk is not None:
        try:
            res = mk.original_test(y_det)
            result["trend_slope"] = res.slope if HAVE_SLOPE else np.nan
            result["trend_sig"] = (res.trend != "no trend") and (res.p < trend_alpha)
        except Exception:
            pass
    if len(y) >= min_obs_cpt:
        try:
            cps = rpt.Pelt(model="rbf").fit(y).predict(pen=1)
            if len(cps) > 1:
                result["changepoint_day"] = dates[cps[0]-1].strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        med = np.median(y)
        mad_v = median_abs_deviation(y, scale="normal")
        mad_z = 0.6745 * (y - med) / (mad_v + 1e-8)
        zs = np.abs(zscore(y)) if len(y) > 1 else np.zeros_like(y)
        n_out = np.maximum(np.abs(mad_z) > 3.5, zs > 3).sum()
        result["outlier_count"] = int(n_out)
        result["mad_stat"] = mad_v
    except Exception:
        pass
    return result

# 整体批量分析
def analyze_data(
    agg_df, ts_df,
    min_obs_trend=DEFAULT_MIN_OBS_TREND,
    min_obs_cpt=DEFAULT_MIN_OBS_CPT,
    trend_alpha=DEFAULT_TREND_ALPHA
):
    agg_df["trend_slope"] = np.nan
    agg_df["trend_sig"] = False
    agg_df["changepoint_day"] = pd.Series([pd.NA] * len(agg_df), dtype="object")
    agg_df["outlier_count"] = 0
    agg_df["mad_stat"] = np.nan
    for i, row in agg_df.iterrows():
        sub = ts_df[(ts_df.latitude == row.latitude) & (ts_df.longitude == row.longitude)]
        if sub.empty or pd.isna(sub.iloc[0].time_series):
            continue
        analysis = analyze_grid_trend(
            sub.iloc[0].time_series,
            min_obs_trend=min_obs_trend,
            min_obs_cpt=min_obs_cpt,
            trend_alpha=trend_alpha
        )
        agg_df.at[i, "trend_slope"] = analysis["trend_slope"]
        agg_df.at[i, "trend_sig"] = analysis["trend_sig"]
        agg_df.at[i, "changepoint_day"] = analysis["changepoint_day"]
        agg_df.at[i, "outlier_count"] = analysis["outlier_count"]
        agg_df.at[i, "mad_stat"] = analysis["mad_stat"]
    return agg_df

# 结果保存
def save_analysis(agg_df, output_dir=DEFAULT_OUTPUT_DIR):
    agg_df["trend_slope"] = pd.to_numeric(agg_df["trend_slope"], errors="coerce")
    agg_df["trend_sig"] = agg_df["trend_sig"].astype(bool)
    agg_df.to_csv(f"{output_dir}/agg_analysis.csv", index=False)

def main(
    output_dir=DEFAULT_OUTPUT_DIR,
    min_obs_trend=DEFAULT_MIN_OBS_TREND,
    min_obs_cpt=DEFAULT_MIN_OBS_CPT,
    trend_alpha=DEFAULT_TREND_ALPHA
):
    agg = pd.read_csv(f"{output_dir}/agg.csv", low_memory=False)
    ts_df = pd.read_csv(f"{output_dir}/ts_df.csv", low_memory=False)
    agg = analyze_data(
        agg, ts_df,
        min_obs_trend=min_obs_trend,
        min_obs_cpt=min_obs_cpt,
        trend_alpha=trend_alpha
    )
    save_analysis(agg, output_dir)
    print(f"Number of valid trend_slope outputs: {agg['trend_slope'].notnull().sum()}")

if __name__ == "__main__":
    main()
