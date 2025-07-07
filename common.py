#!/usr/bin/env python3
# ======================================================================
#  common.py  ·  常量 / 数据加载 / 通用函数          v2025-07-06
# ======================================================================
from __future__ import annotations

import os, pathlib
import pandas as pd
import numpy as np
import streamlit as st

# ───────────────────────── 路径 & 文件名 ──────────────────────────
DIR_RESULTS   = pathlib.Path("ch4_sampling_result")   # 后端主结果目录
DIR_ENGINE    = pathlib.Path("sampling_engine")       # 抽样引擎输出目录
DIR_RESULTS.mkdir(exist_ok=True, parents=True)
DIR_ENGINE.mkdir(exist_ok=True,  parents=True)

# ★ 对旧代码的兼容别名（不要删） ------------------------------
DIR          : pathlib.Path = DIR_RESULTS            # 旧脚本期望的名字
OUT_SAMPLING : pathlib.Path = DIR_ENGINE             # tab_optimizer 里用
# ---------------------------------------------------------------

FILES: dict[str, str] = {
    "main"  : "ALL_ch4_sampling_suggestion.csv",
    "blind" : "suggested_blind_spots.csv",
    "hist"  : "priority_score_histogram.csv",
    "report": "CH4_Quick_Analysis_Report.txt",
    "readme": "ALL_ch4_sampling_suggestion_README.txt",
    "ts"    : "ts_df.csv",
}
REQUIRED: tuple[str, ...] = ("main", "ts")           # 缺一不可继续

# ───────────────────────── 视觉常量 ──────────────────────────
PRIO_LV: list[str] = [
    "High Priority (Weekly Sampling)",
    "Medium Priority (Monthly Sampling)",
    "Low Priority (Quarterly Sampling)",
    "Minimal Priority (Biannual)",
]
PRIO_COL: dict[str, str] = {
    PRIO_LV[0]: "#d73027",   # red
    PRIO_LV[1]: "#fc8d59",   # orange
    PRIO_LV[2]: "#91bfdb",   # light-blue
    PRIO_LV[3]: "#4575b4",   # dark-blue
}

# ───────────────────────── I/O wrappers ─────────────────────────
@st.cache_data(show_spinner=False)
def _safe_read_csv(path: os.PathLike | str) -> pd.DataFrame:
    """读 CSV，异常时返回空 DF 而不抛错。"""
    try:
        df = pd.read_csv(path, low_memory=False)
        # 添加健壮性提示
        if df.empty:
            st.warning(f"⚠️ 文件 {path} 读取成功，但内容为空。")
        else:
            st.info(f"✅ 读取 {path}: {df.shape[0]} 行, 列: {list(df.columns)}")
        return df
    except Exception as e:
        st.error(f"❌ 读取 {path} 失败: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame,
                        pd.DataFrame, pd.DataFrame]:
    """
    统一读取所有结果文件。\n
    输出顺序：`(main_df, blind_df, hist_df, ts_df)`\n
    若关键文件缺失，会在 Streamlit 前端报错并返回 4×空 DF。
    """
    dfs = {k: _safe_read_csv(DIR_RESULTS / v) for k, v in FILES.items()}

    missing = [k for k in REQUIRED if dfs[k].empty]
    if missing:
        st.error("❌ 以下关键结果文件缺失或为空，请先运行后端脚本："
                 + ", ".join(missing))
        return tuple(pd.DataFrame() for _ in range(4))
    # 增加调试输出
    for k, df in dfs.items():
        st.caption(f"【DEBUG】{k}: 形状={df.shape}, 字段={list(df.columns)}")
        if not df.empty:
            st.caption(df.head(2))

    return dfs["main"], dfs["blind"], dfs["hist"], dfs["ts"]

# ───────────────────────── 数据清洗 ─────────────────────────
def numeric_safe_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    将关键列安全转为数值型；**不会**修改原 DF，返回新对象。
    """
    if df is None or df.empty:
        st.warning("numeric_safe_cast: 输入为空表")
        return pd.DataFrame()

    out = df.copy()
    num_cols = [
        "latitude", "longitude", "mean_ch4", "std_ch4",
        "trend_slope", "priority_score", "obs_count", "outlier_count",
    ]
    # 自动修正数字类型
    for col in num_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # 提示缺失/异常列
    for col in num_cols:
        if col not in out.columns:
            st.warning(f"⚠️ 缺少字段: {col}")

    # std_ch4 若缺失/为零 → 用最小非零替代，避免 0 导致 size_ref 为 0
    if "std_ch4" in out.columns:
        min_std = out.loc[out["std_ch4"] > 0, "std_ch4"].min() or 1.0
        out["std_ch4"] = out["std_ch4"].fillna(min_std).clip(lower=1e-3)

    # 检查经纬度全空
    if "latitude" in out.columns and "longitude" in out.columns:
        null_lat = out["latitude"].isnull().sum()
        null_lon = out["longitude"].isnull().sum()
        st.caption(f"【DEBUG】latitude 缺失: {null_lat} / {len(out)}，longitude 缺失: {null_lon} / {len(out)}")
        if null_lat == len(out) or null_lon == len(out):
            st.error("❌ 所有经纬度都缺失，地图不会显示任何点。")

    return out
