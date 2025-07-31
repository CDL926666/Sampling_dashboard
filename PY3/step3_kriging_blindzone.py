# Y:\Bishe_project\PY3\step3_kriging_blindzone.py

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, Voronoi
import warnings

# kriging模块可选导入
try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None

DEFAULT_OUT_DIR = "ch4_sampling_result"
DEFAULT_GRID_SIZE = 1.0
DEFAULT_BLIND_DIST_FACTOR = 3.0
DEFAULT_KRIG_MIN_PTS = 50
DEFAULT_KRIG_MAX_PTS = 1000

# 克里金插值
def kriging_interpolation(
    df,
    value_col,
    grid_size=DEFAULT_GRID_SIZE,
    model_list=None,
    min_pts=DEFAULT_KRIG_MIN_PTS,
    max_pts=DEFAULT_KRIG_MAX_PTS
):
    if OrdinaryKriging is None:
        return None, (None, None), None
    if model_list is None:
        model_list = ["linear", "gaussian", "exponential", "spherical"]
    interp = df.dropna(subset=[value_col])
    if len(interp) < min_pts:
        return None, (None, None), None
    if len(interp) > max_pts:
        interp = interp.sample(max_pts, random_state=42)
    lon_rng = np.arange(interp["longitude"].min(), interp["longitude"].max() + grid_size, grid_size)
    lat_rng = np.arange(interp["latitude"].min(), interp["latitude"].max() + grid_size, grid_size)
    krig_var = None
    krig_best = None
    best_rmse = np.inf
    krig_scores = {}
    for model in model_list:
        try:
            OK = OrdinaryKriging(
                interp["longitude"], interp["latitude"], interp[value_col],
                variogram_model=model, verbose=False, enable_plotting=False
            )
            z, ss = OK.execute("grid", lon_rng, lat_rng)
            rmse = float(np.nanstd(z))
            krig_scores[model] = rmse
            if rmse < best_rmse:
                best_rmse, krig_var, krig_best = rmse, ss, model
        except Exception:
            krig_scores[model] = None
    return krig_var, (lat_rng, lon_rng), krig_best

# 检测空间盲区
def detect_blind_spots(
    agg,
    krig_var,
    lat_rng,
    lon_rng,
    grid_size=DEFAULT_GRID_SIZE,
    blind_dist=None
):
    if blind_dist is None:
        blind_dist = DEFAULT_BLIND_DIST_FACTOR * grid_size
    blind_spots = []
    if krig_var is not None and np.isfinite(krig_var).any():
        v_th = np.nanquantile(krig_var, 0.95)
        for i, lat in enumerate(lat_rng):
            for j, lon in enumerate(lon_rng):
                if krig_var[i, j] > v_th:
                    blind_spots.append({"latitude": lat, "longitude": lon, "method": "kriging_var"})
    coords = agg[["latitude", "longitude"]].dropna().to_numpy()
    if len(coords) >= 2 and lat_rng is not None and lon_rng is not None:
        tree = cKDTree(coords)
        mesh_lat, mesh_lon = np.meshgrid(lat_rng, lon_rng)
        mesh_pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])
        dists, _ = tree.query(mesh_pts, k=1)
        for p, dist in zip(mesh_pts, dists):
            if dist > blind_dist:
                blind_spots.append({"latitude": p[0], "longitude": p[1], "method": "density_dist"})
    try:
        if len(coords) >= 4:
            _ = Voronoi(coords)
    except Exception:
        pass
    blind_df = pd.DataFrame(blind_spots).drop_duplicates(["latitude", "longitude"])
    return blind_df

# 标记盲区flag
def mark_blind_spot_flag(agg, blind_df):
    if not blind_df.empty:
        agg["blind_spot_flag"] = agg.apply(
            lambda r: "blind_zone" if (
                (np.abs(blind_df.latitude - r.latitude) < 1e-6) &
                (np.abs(blind_df.longitude - r.longitude) < 1e-6)
            ).any() else "ok", axis=1
        )
    else:
        agg["blind_spot_flag"] = "ok"
    return agg

# 保存盲区检测结果
def save_blindzone_results(blind_df, agg, out_dir):
    if not blind_df.empty:
        blind_df["suggest"] = "Add Sampling"
        blind_df.to_csv(f"{out_dir}/suggested_blind_spots.csv", index=False)
        stats_df = blind_df["method"].value_counts().rename("count").to_frame()
        stats_df["ratio"] = (stats_df["count"] / len(blind_df)).round(3)
        stats_df.T.to_csv(f"{out_dir}/blind_spot_stats.csv")
    agg.to_csv(f"{out_dir}/agg_with_blind.csv", index=False, encoding="utf-8")

def main(
    agg_path=f"{DEFAULT_OUT_DIR}/agg_analysis.csv",
    value_col="mean_ch4",
    out_dir=DEFAULT_OUT_DIR,
    grid_size=DEFAULT_GRID_SIZE,
    min_pts=DEFAULT_KRIG_MIN_PTS,
    max_pts=DEFAULT_KRIG_MAX_PTS,
    blind_dist_factor=DEFAULT_BLIND_DIST_FACTOR
):
    agg = pd.read_csv(agg_path, low_memory=False)
    krig_var, (lat_rng, lon_rng), krig_best = kriging_interpolation(
        agg, value_col, grid_size, min_pts=min_pts, max_pts=max_pts
    )
    blind_df = detect_blind_spots(
        agg, krig_var, lat_rng, lon_rng,
        grid_size=grid_size, blind_dist=blind_dist_factor*grid_size
    )
    agg = mark_blind_spot_flag(agg, blind_df)
    save_blindzone_results(blind_df, agg, out_dir)

    if agg.empty or agg["latitude"].isnull().all() or agg["longitude"].isnull().all():
        print("Empty data, kriging/blind spot detection skipped.")
        pd.DataFrame().to_csv(f"{out_dir}/agg_with_blind.csv")
        return

if __name__ == "__main__":
    main()
