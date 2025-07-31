# Y:\Bishe_project\PY3\step4_priority_report.py

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import inspect

try:
    import jenkspy
    HAS_JENKS = True
    _SIG = inspect.signature(jenkspy.jenks_breaks)
    _JENKS_KW = "n_classes" if "n_classes" in _SIG.parameters else "nb_class"
except ImportError:
    HAS_JENKS = False

DEFAULT_DIR_OUT = "ch4_sampling_result"
DEFAULT_TREND_ALPHA = 0.05
DEFAULT_NORM_COLS = {
    "std_ch4": 0.22,
    "trend_slope": 0.22,
    "outlier_count": 0.14,
    "space_sparse": 0.18,
    "mad_stat": 0.14,
    "data_quality_flag": 0.10
}

# 空间稀疏度计算
def calc_space_sparse(agg, grid_cols=("latitude", "longitude")):
    coords_df = agg[list(grid_cols)].dropna()
    sparse = pd.Series(0.0, index=agg.index)
    if len(coords_df) >= 2:
        try:
            nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(coords_df)
            dist, _ = nbrs.kneighbors(coords_df)
            sparse.loc[coords_df.index] = dist[:, 1]
        except Exception as e:
            warnings.warn(f"[NearestNeighbors] {e} — space_sparse=0")
    return sparse

# 各指标归一化
def normalize_columns(agg, norm_cols):
    for col, w in norm_cols.items():
        if col not in agg.columns:
            agg[col] = 0
        if col == "data_quality_flag":
            agg[f"{col}_norm"] = (agg[col] == "sparse").astype(float)
        else:
            lo, hi = agg[col].min(), agg[col].max()
            if hi > lo:
                norm_series = (agg[col] - lo) / (hi - lo + 1e-8)
            else:
                norm_series = pd.Series(0.0, index=agg.index)
            norm_series = pd.to_numeric(norm_series, errors="coerce").replace([np.inf, -np.inf], 0)
            agg[f"{col}_norm"] = norm_series
    return agg

# 计算优先级分数
def calc_priority_score(agg, norm_cols):
    agg["priority_score"] = sum(norm_cols[c] * agg[f"{c}_norm"] for c in norm_cols)
    return agg

# 优先级分级
def classify_priority(agg, breaks):
    def _label(x):
        return (
            "High Priority (Weekly Sampling)" if x >= breaks[3] else
            "Medium Priority (Monthly Sampling)" if x >= breaks[2] else
            "Low Priority (Quarterly Sampling)" if x >= breaks[1] else
            "Minimal Priority (Biannual)"
        )
    agg["sampling_recommendation"] = agg["priority_score"].apply(_label)
    return agg

# 导出csv文件
def export_csvs(agg, dir_out):
    valid = agg["priority_score"].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        warnings.warn("[export_csvs] No valid priority_score values, exporting empty histogram.")
        pd.DataFrame({"priority_score_bins": [], "count": []}).to_csv(f"{dir_out}/priority_score_histogram.csv", index=False)
        agg.to_csv(f"{dir_out}/ALL_ch4_sampling_suggestion.csv", index=False, encoding="utf-8")
        return
    hist, bins = np.histogram(valid, bins=30)
    pd.DataFrame({"priority_score_bins": bins[:-1], "count": hist}).to_csv(f"{dir_out}/priority_score_histogram.csv", index=False)
    agg.to_csv(f"{dir_out}/ALL_ch4_sampling_suggestion.csv", index=False, encoding="utf-8")

# 输出字段说明
def write_readme(agg, norm_cols, breaks, dir_out, trend_alpha=DEFAULT_TREND_ALPHA):
    with open(f"{dir_out}/ALL_ch4_sampling_suggestion_README.txt", "w", encoding="utf-8") as f:
        f.write(f"""# Field Description
grid_id                : Grid cell ID
latitude / longitude   : Grid center coordinates (deg)
mean_ch4 / std_ch4     : Statistics
obs_count              : Observation count
trend_slope / trend_sig: Trend indicators (p<{trend_alpha})
changepoint_day        : Change point date
outlier_count / mad_stat: Outlier indicators
data_quality_flag      : sparse / ok
blind_spot_flag        : blind_zone / ok
space_sparse           : Nearest neighbor distance (deg)
priority_score         : Weighted normalized score
sampling_recommendation: Recommended sampling frequency
# Weights: {norm_cols}
# Breaks : {breaks}
""")

# 输出汇总报告
def write_summary_report(agg, prio_lv, prio_cnt, breaks, dir_out, trend_alpha=DEFAULT_TREND_ALPHA):
    summary = [
        "==== CH4 Sampling Analysis Summary ====",
        f"Total valid grid cells : {len(agg)}",
        *[f"{lv} priority grids : {prio_cnt[lv]}" for lv in prio_lv],
        f"Significant trend grids (p<{trend_alpha}) : {int(agg['trend_sig'].sum())}"
    ]
    try:
        bz_df = pd.read_csv(f"{dir_out}/suggested_blind_spots.csv")
        bz_stat = pd.read_csv(f"{dir_out}/blind_spot_stats.csv").iloc[0].to_dict()
        summary.append(f"Blind zones requiring sampling : {len(bz_df)}")
        summary += [f"  · {k} : {v}" for k, v in bz_stat.items()]
    except Exception:
        summary.append("Blind zones requiring sampling : N/A")
    try:
        q25, q50, q75, q95 = np.nanquantile(agg["priority_score"], [0.25, 0.5, 0.75, 0.95])
        summary.append(f"Priority-score quartiles : {q25:.3f}, {q50:.3f}, {q75:.3f}, {q95:.3f}")
    except Exception:
        summary.append("Priority-score quartiles : N/A")
    summary.append("Key files : ALL_ch4_sampling_suggestion.csv, priority_score_histogram.csv")
    with open(f"{dir_out}/CH4_Quick_Analysis_Report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

def main(
    agg_path=f"{DEFAULT_DIR_OUT}/agg_with_blind.csv",
    dir_out=DEFAULT_DIR_OUT,
    trend_alpha=DEFAULT_TREND_ALPHA,
    norm_cols=DEFAULT_NORM_COLS
):
    agg = pd.read_csv(agg_path, low_memory=False)
    if agg.empty:
        warnings.warn("[main] Input agg table is empty. All outputs will be empty.")
        export_csvs(agg, dir_out)
        write_readme(agg, norm_cols, np.array([0, 0, 0, 0, 0]), dir_out, trend_alpha=trend_alpha)
        write_summary_report(agg, ["High", "Medium", "Low", "Minimal"], {lv: 0 for lv in ["High", "Medium", "Low", "Minimal"]}, np.array([0, 0, 0, 0, 0]), dir_out, trend_alpha=trend_alpha)
        return

    agg["space_sparse"] = calc_space_sparse(agg)
    agg = normalize_columns(agg, norm_cols)
    agg = calc_priority_score(agg, norm_cols)
    valid = agg["priority_score"].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        breaks = np.array([0, 0, 0, 0, 0])
    elif HAS_JENKS and valid.nunique() >= 4:
        breaks = jenkspy.jenks_breaks(valid, **{_JENKS_KW: 4})
    else:
        breaks = np.nanpercentile(valid, [0, 25, 50, 75, 100])
    agg = classify_priority(agg, breaks)
    prio_lv = ["High", "Medium", "Low", "Minimal"]
    prio_cnt = {lv: agg["sampling_recommendation"].str.contains(lv).sum() for lv in prio_lv}
    export_csvs(agg, dir_out)
    write_readme(agg, norm_cols, breaks, dir_out, trend_alpha=trend_alpha)
    write_summary_report(agg, prio_lv, prio_cnt, breaks, dir_out, trend_alpha=trend_alpha)
    print(f"✓ Priority report output. Table rows: {len(agg)}, Valid score: {len(valid)}")

if __name__ == "__main__":
    main()
