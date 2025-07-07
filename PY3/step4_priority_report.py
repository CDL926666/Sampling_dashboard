"""
Step 6–8 : Priority Scoring → Diagnostics → Text Report
输入  : ch4_sampling_result/agg_with_blind.csv
输出  : ALL_ch4_sampling_suggestion.csv
        priority_score_histogram.csv
        ALL_ch4_sampling_suggestion_README.txt
        CH4_Quick_Analysis_Report.txt
可选  : SAVE_PLOTS=True 时再额外生成 PNG
"""
from __future__ import annotations
import os, time, warnings, inspect
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ────────── jenkspy 兼容处理 ──────────
try:
    import jenkspy                               # type: ignore
    HAS_JENKS = True
    _SIG      = inspect.signature(jenkspy.jenks_breaks)
    _JENKS_KW = "n_classes" if "n_classes" in _SIG.parameters else "nb_class"
except ImportError:
    HAS_JENKS = False
    warnings.warn("jenkspy not installed ⇒ 使用四分位分档")

# ────────── 全局参数 ──────────
DIR_OUT     = "ch4_sampling_result"
TREND_ALPHA = 0.05
SAVE_PLOTS  = False               # 调试/汇报时设 True

# ────────── 小计时器 ──────────
def tic(msg): t0=time.perf_counter(); print(f"\n[{time.strftime('%H:%M:%S')}] {msg} …"); return t0
def toc(t0, msg): print(f"    → {msg} 完成 {time.perf_counter()-t0:.2f}s")

# ════════════════════  Step 6  ════════════════════
t6 = tic("Step 6: 计算 priority_score 与采样级别")

agg = pd.read_csv(f"{DIR_OUT}/agg_with_blind.csv", low_memory=False)

# ① 空间稀疏度（最近邻距离）
coords_df = agg[["latitude", "longitude"]].dropna()
sparse = pd.Series(0.0, index=agg.index)
if len(coords_df) >= 2:
    try:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(coords_df)
        dist, _ = nbrs.kneighbors(coords_df)
        sparse.loc[coords_df.index] = dist[:, 1]
    except Exception as e:
        warnings.warn(f"[NearestNeighbors] {e} — space_sparse=0")

agg["space_sparse"] = sparse

# ② 指标归一化 + 权重
NORM_COLS = {
    "std_ch4"        : 0.22,
    "trend_slope"    : 0.22,
    "outlier_count"  : 0.14,
    "space_sparse"   : 0.18,
    "mad_stat"       : 0.14,
    "data_quality_flag": 0.10
}
for col, w in NORM_COLS.items():
    if col not in agg.columns:
        agg[col] = 0
    if col == "data_quality_flag":
        agg[f"{col}_norm"] = (agg[col] == "sparse").astype(float)
    else:
        lo, hi = agg[col].min(), agg[col].max()
        if hi > lo:
            norm_series = (agg[col] - lo) / (hi - lo + 1e-8)
        else:                                        # ← 所有值相同
            norm_series = pd.Series(0.0, index=agg.index)      ### FIX
        norm_series = pd.to_numeric(norm_series, errors="coerce")\
                          .replace([np.inf, -np.inf], 0)
        agg[f"{col}_norm"] = norm_series                         ### FIX

agg["priority_score"] = sum(
    NORM_COLS[c] * agg[f"{c}_norm"] for c in NORM_COLS
)

# ③ Jenks 自然断点（兼容不同版本）；若失败退回四分位
valid = agg["priority_score"].dropna().replace([np.inf, -np.inf], np.nan).dropna()
if HAS_JENKS and valid.nunique() >= 4:
    breaks = jenkspy.jenks_breaks(valid, **{_JENKS_KW: 4})      # type: ignore
else:
    breaks = np.nanpercentile(valid, [0, 25, 50, 75, 100])

def _label(x: float) -> str:
    return ("High Priority (Weekly Sampling)"    if x >= breaks[3] else
            "Medium Priority (Monthly Sampling)" if x >= breaks[2] else
            "Low Priority (Quarterly Sampling)"  if x >= breaks[1] else
            "Minimal Priority (Biannual)")
agg["sampling_recommendation"] = agg["priority_score"].apply(_label)

# ④ 统计
prio_lv  = ["High", "Medium", "Low", "Minimal"]
prio_cnt = {lv: agg["sampling_recommendation"].str.contains(lv).sum() for lv in prio_lv}

# ⑤ 输出 CSV
hist, bins = np.histogram(agg["priority_score"], bins=30)
pd.DataFrame({"priority_score_bins": bins[:-1], "count": hist})\
  .to_csv(f"{DIR_OUT}/priority_score_histogram.csv", index=False)
agg.to_csv(f"{DIR_OUT}/ALL_ch4_sampling_suggestion.csv", index=False, encoding="utf-8")

toc(t6, "Step 6")

# ════════════════════  Step 7  ════════════════════
t7 = tic("Step 7: 诊断图 + README")

if SAVE_PLOTS:
    COLMAP = {"High":"#d73027", "Medium":"#fc8d59", "Low":"#91bfdb", "Minimal":"#4575b4"}
    def _save(fig, name): fig.tight_layout(); fig.savefig(f"{DIR_OUT}/{name}", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(7,4))
    plt.hist(agg["obs_count"], 30, color="skyblue", edgecolor="k")
    plt.title("Observation Count / Grid"); _save(fig, "obs_count_hist.png")

    fig = plt.figure(figsize=(9,5))
    plt.scatter(agg["longitude"], agg["latitude"], c=agg["priority_score"],
                cmap="YlOrRd", s=8); plt.colorbar(label="Priority Score")
    plt.title("Spatial Priority (score)"); _save(fig, "priority_map.png")

    fig = plt.figure(figsize=(9,5))
    for lv, c in COLMAP.items():
        sel = agg["sampling_recommendation"].str.contains(lv)
        plt.scatter(agg.loc[sel, "longitude"], agg.loc[sel, "latitude"],
                    c=c, s=6, label=lv, alpha=.8)
    plt.legend(); plt.title("Priority Classes"); _save(fig, "priority_class_map.png")

    fig = plt.figure(figsize=(6,4))
    plt.bar(prio_cnt.keys(), prio_cnt.values(), color=list(COLMAP.values()))
    plt.ylabel("Grids"); plt.title("Priority Level Counts"); _save(fig, "priority_level_bar.png")

with open(f"{DIR_OUT}/ALL_ch4_sampling_suggestion_README.txt", "w", encoding="utf-8") as f:
    f.write(f"""# 字段说明
grid_id               : 网格 ID
latitude / longitude  : 栅格中心坐标 (deg)
mean_ch4 / std_ch4    : 统计量
obs_count             : 观测次数
trend_slope / trend_sig : 趋势指标 (p<{TREND_ALPHA})
changepoint_day       : 突变日期
outlier_count / mad_stat : 异常指标
data_quality_flag     : sparse / ok
blind_spot_flag       : blind_zone / ok
space_sparse          : 最近邻距离 (deg)
priority_score        : 加权归一化得分
sampling_recommendation : 采样频率建议
# 权重 : {NORM_COLS}
# breaks : {breaks}
""")

toc(t7, "Step 7")

# ════════════════════  Step 8  ════════════════════
t8 = tic("Step 8: 文本报告")

summary = [
    "==== CH₄ Sampling Analysis Summary ====",
    f"Total valid grid cells : {len(agg)}",
    *[f"{lv} priority grids : {prio_cnt[lv]}" for lv in prio_lv],
    f"Significant trend grids (p<{TREND_ALPHA}) : {int(agg['trend_sig'].sum())}"
]

try:
    bz_df   = pd.read_csv(f"{DIR_OUT}/suggested_blind_spots.csv")
    bz_stat = pd.read_csv(f"{DIR_OUT}/blind_spot_stats.csv").iloc[0].to_dict()
    summary.append(f"Blind zones requiring sampling : {len(bz_df)}")
    summary += [f"  · {k} : {v}" for k, v in bz_stat.items()]
except Exception:
    summary.append("Blind zones requiring sampling : N/A")

q25, q50, q75, q95 = np.nanquantile(agg["priority_score"], [0.25, 0.5, 0.75, 0.95])
summary.append(f"Priority-score quartiles : {q25:.3f}, {q50:.3f}, {q75:.3f}, {q95:.3f}")
summary.append("Key files : ALL_ch4_sampling_suggestion.csv, priority_score_histogram.csv"
               + (", PNG diagnostics" if SAVE_PLOTS else ""))

print("\n".join(summary))
with open(f"{DIR_OUT}/CH4_Quick_Analysis_Report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

toc(t8, "Step 8")
print(f"\n✓ All outputs written to «{DIR_OUT}»\n")
