"""
Step 4–5 : Kriging + Blind-zone Detection
输入  : ch4_sampling_result/agg_analysis.csv
输出  : suggested_blind_spots.csv, blind_spot_stats.csv, agg_with_blind.csv
"""

import os, time, warnings, numpy as np, pandas as pd
from scipy.spatial import cKDTree, Voronoi          # Voronoi 仅作质量检查

# ───── 尝试导入 pykrige ─────
try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None

# ───── 全局参数 ─────
OUT_DIR            = "ch4_sampling_result"
GRID_SIZE          = 1.0
BLIND_DIST         = 3 * GRID_SIZE
KRIG_MIN_PTS       = 50
KRIG_MAX_PTS       = 1000
SAVE_PLOTS         = False          # True 时生成调试 PNG

# ───── 计时工具 ─────
def tic(msg):
    t0 = time.perf_counter(); print(f"\n[{time.strftime('%H:%M:%S')}] {msg} …"); return t0
def toc(t0, msg):
    print(f"    → {msg} finished in {time.perf_counter() - t0:.2f}s")

# ───── 数据读取 ─────
agg = pd.read_csv(f"{OUT_DIR}/agg_analysis.csv", low_memory=False)

# 统一网格经纬度范围（Kriging/盲区共用，避免未定义）  >>> FIX
lat_rng = np.arange(agg["latitude"].min(),  agg["latitude"].max() + GRID_SIZE, GRID_SIZE)
lon_rng = np.arange(agg["longitude"].min(), agg["longitude"].max() + GRID_SIZE, GRID_SIZE)

# ═════════════════════════════════  Step 4  ════════════════════════════════
t4 = tic("Step 4: (optional) Kriging variance")

krig_var = None; krig_best = None; krig_scores = {}

if OrdinaryKriging is None:
    print("[INFO] pykrige not installed ⇒ skip Kriging")
else:
    interp = agg.dropna(subset=["mean_ch4"])
    if len(interp) >= KRIG_MIN_PTS:
        if len(interp) > KRIG_MAX_PTS:
            interp = interp.sample(KRIG_MAX_PTS, random_state=42)
        best_rmse = np.inf
        for model in ["linear", "gaussian", "exponential", "spherical"]:
            try:
                OK = OrdinaryKriging(interp["longitude"], interp["latitude"], interp["mean_ch4"],
                                     variogram_model=model, verbose=False, enable_plotting=False)
                z, ss = OK.execute("grid", lon_rng, lat_rng)
                rmse  = float(np.nanstd(z))
                krig_scores[model] = rmse
                if rmse < best_rmse:
                    best_rmse, krig_var, krig_best = rmse, ss, model
                    if SAVE_PLOTS:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(8,5))
                        plt.pcolormesh(lon_rng, lat_rng, z, shading="auto", cmap="coolwarm")
                        plt.colorbar(label="Kriged CH₄")
                        plt.title(f"Kriged mean ({model})")
                        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/DEBUG_krig_mean.png", dpi=160); plt.close()
            except Exception as e:
                krig_scores[model] = f"FAIL: {e}"
        print(f"[Kriging] best model = {krig_best}, RMSE = {best_rmse:.3g}")
    else:
        print(f"[INFO] Too few points ({len(interp)}) ⇒ skip Kriging")

toc(t4, "Step 4")

# ═════════════════════════════════  Step 5  ════════════════════════════════
t5 = tic("Step 5: Blind-zone detection")

blind_spots = []

# 判据 1：Kriging 方差 95% 分位
if krig_var is not None and np.isfinite(krig_var).any():
    v_th = np.nanquantile(krig_var, 0.95)
    for i, lat in enumerate(lat_rng):
        for j, lon in enumerate(lon_rng):
            if krig_var[i, j] > v_th:
                blind_spots.append({"latitude": lat, "longitude": lon, "method": "kriging_var"})

# 判据 2：最近邻 > BLIND_DIST
coords = agg[["latitude", "longitude"]].dropna().to_numpy()
if len(coords) >= 2:
    tree = cKDTree(coords)
    mesh_lat, mesh_lon = np.meshgrid(lat_rng, lon_rng)
    mesh_pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])
    dists, _ = tree.query(mesh_pts, k=1)
    for p, dist in zip(mesh_pts, dists):
        if dist > BLIND_DIST:
            blind_spots.append({"latitude": p[0], "longitude": p[1], "method": "density_dist"})

# 判据 3：Voronoi 质量检查
try:
    if len(coords) >= 4:
        _ = Voronoi(coords)
except Exception as e:
    warnings.warn(f"[Voronoi skipped] {e}")

# 合并去重 & 统计
blind_df = pd.DataFrame(blind_spots).drop_duplicates(["latitude", "longitude"])
stats_dict = blind_df["method"].value_counts().to_dict()     # 供 Step 6 读取

# -------- 文件输出 --------
if not blind_df.empty:
    blind_df["suggest"] = "Add Sampling"
    blind_df.to_csv(f"{OUT_DIR}/suggested_blind_spots.csv", index=False)

    stats_df = (blind_df["method"].value_counts()
                .rename("count").to_frame())
    stats_df["ratio"] = (stats_df["count"] / len(blind_df)).round(3)
    stats_df.T.to_csv(f"{OUT_DIR}/blind_spot_stats.csv")   # 行 = method

    agg["blind_spot_flag"] = agg.apply(
        lambda r: "blind_zone" if ((np.abs(blind_df.latitude  - r.latitude)  < 1e-6) &
                                   (np.abs(blind_df.longitude - r.longitude) < 1e-6)).any()
        else "ok", axis=1)
else:
    agg["blind_spot_flag"] = "ok"


agg.to_csv(f"{OUT_DIR}/agg_with_blind.csv", index=False, encoding="utf-8")
toc(t5, "Step 5")

print("✓ Blind-zone detection complete — outputs written to folder.")
