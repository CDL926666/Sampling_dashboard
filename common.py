# Y:\Bishe_project\common.py

from __future__ import annotations

import os
import pathlib
import pandas as pd
import numpy as np
import streamlit as st

# 全局路径/文件名配置
BASE_DIR = pathlib.Path(os.getenv("CH4_BASE_DIR", ".")).resolve()
DIR_RESULTS = BASE_DIR / "ch4_sampling_result"
DIR_ENGINE  = BASE_DIR / "sampling_engine"
DIR_RESULTS.mkdir(exist_ok=True, parents=True)
DIR_ENGINE.mkdir(exist_ok=True, parents=True)

DIR: pathlib.Path = DIR_RESULTS
OUT_SAMPLING: pathlib.Path = DIR_ENGINE

FILES: dict[str, str] = {
    "main"  : "ALL_ch4_sampling_suggestion.csv",
    "blind" : "suggested_blind_spots.csv",
    "hist"  : "priority_score_histogram.csv",
    "report": "CH4_Quick_Analysis_Report.txt",
    "readme": "ALL_ch4_sampling_suggestion_README.txt",
    "ts"    : "ts_df.csv",
}
REQUIRED: tuple[str, ...] = ("main", "ts")

# 采样优先级和配色
PRIO_LV: list[str] = [
    "High Priority (Weekly Sampling)",
    "Medium Priority (Monthly Sampling)",
    "Low Priority (Quarterly Sampling)",
    "Minimal Priority (Biannual)",
]
PRIO_COL: dict[str, str] = {
    PRIO_LV[0]: "#ff0d00",
    PRIO_LV[1]: "#ffc320",
    PRIO_LV[2]: "#12a2fc",
    PRIO_LV[3]: "#3abda5",
}

ANALYSIS_CONFIG: dict[str, dict] = {
    "methane": {
        "field_mean": "mean_ch4",
        "field_std": "std_ch4",
        "lat_col": "latitude",
        "lon_col": "longitude",
        "valid_range": (0, np.inf),
    },
}

# 数据加载工具

@st.cache_data(show_spinner=False)
def _safe_read_csv(path: os.PathLike | str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            st.warning(f"File {path} read successfully, but is empty.")
        return df
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_df  = _safe_read_csv(DIR_RESULTS / FILES["main"])
    blind_df = _safe_read_csv(DIR_RESULTS / FILES["blind"])
    hist_df  = _safe_read_csv(DIR_RESULTS / FILES["hist"])
    ts_df    = _safe_read_csv(DIR_RESULTS / FILES["ts"])

    # 文本文件加载（只放入session_state，不参与返回）
    for key in ("report", "readme"):
        p = DIR_RESULTS / FILES[key]
        if p.exists():
            try:
                st.session_state[f"txt_{key}"] = p.read_text(encoding="utf-8")
            except Exception as e:
                st.warning(f"Read {p.name} failed: {e}")

    if main_df.empty or ts_df.empty:
        st.error("Key result files missing or empty, please run backend scripts first.")
    return main_df, blind_df, hist_df, ts_df

def numeric_safe_cast(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        st.warning("numeric_safe_cast: input is empty DataFrame")
        return pd.DataFrame()
    out = df.copy()
    num_cols = [
        "latitude", "longitude", "mean_ch4", "std_ch4",
        "trend_slope", "priority_score", "obs_count", "outlier_count",
    ]
    for col in num_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in num_cols:
        if col not in out.columns:
            st.warning(f"Missing field: {col}")

    if "std_ch4" in out.columns:
        min_std = out.loc[out["std_ch4"] > 0, "std_ch4"].min() or 1.0
        out["std_ch4"] = out["std_ch4"].fillna(min_std).clip(lower=1e-3)

    if {"latitude", "longitude"}.issubset(out.columns):
        if out["latitude"].isnull().all() or out["longitude"].isnull().all():
            st.error("All latitude/longitude missing, map will not show any points.")
    return out

# 通用工具函数

def haversine(lon1, lat1, lon2, lat2):
    try:
        if np.isnan([lon1, lat1, lon2, lat2]).any():  # type: ignore
            return None
        R = 6371000.0
        phi1, phi2 = map(np.radians, (lat1, lat2))
        dphi       = np.radians(lat2 - lat1)
        dlambda    = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))
    except Exception as e:
        st.warning(f"haversine error: {e}")
        return None

def format_number(num, precision=3):
    try:
        if num is None or np.isnan(num):
            return "-"
        if abs(num) < 1e-2 or abs(num) > 1e4:
            return f"{num:.{precision}e}"
        return f"{num:,.{precision}f}"
    except Exception:
        return str(num)
