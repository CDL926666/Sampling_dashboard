# Y:\Bishe_project\PY3\config.py
"""
Project global configuration for all modules in the sampling pipeline.

- All parameters are declared at the top level.
- Use this file for both hard constraints and commonly tuned parameters.
- For multi-variable/multi-indicator extension, use additional keys or dicts.
"""

import pathlib

# ===== Data Input/Output =====
ROOT_DIR         = pathlib.Path(__file__).resolve().parents[1]
PY3_DIR          = ROOT_DIR / "PY3"
DATA_DIR         = ROOT_DIR / "ch4_sampling_result"
SAMPLING_OUT_DIR = ROOT_DIR / "sampling_engine"

# ===== Spatial Range (Europe; covers UK and mainland) =====
EU_LAT_MIN, EU_LAT_MAX = 35.0, 72.0
EU_LON_MIN, EU_LON_MAX = -25.0, 45.0

# ===== Grid and Outlier Settings =====
GRID_SIZE            = 1.0      # deg; main spatial grid size
MAX_OUTLIER_SIGMA    = 5        # Exclude if value > mean + N * std
DROP_DUPLICATE       = True

# ===== Trend/Change-point Analysis Settings =====
MIN_OBS_TREND        = 5        # Min obs per grid to analyze trend
MIN_OBS_CPT          = 8        # Min obs to allow change point test
TREND_ALPHA          = 0.05     # Significance level for trend test

# ===== Kriging & Blindzone =====
KRIG_MIN_PTS         = 50       # Minimum points for Kriging
KRIG_MAX_PTS         = 1000     # Maximum points to sample for Kriging
BLIND_DIST_FACTOR    = 3.0      # Mesh point > N * grid size to nearest sample = blind

# ===== Priority Score & Binning =====
NORM_COLS = {
    "std_ch4"          : 0.22,
    "trend_slope"      : 0.22,
    "outlier_count"    : 0.14,
    "space_sparse"     : 0.18,
    "mad_stat"         : 0.14,
    "data_quality_flag": 0.10
}
USE_JENKS_BINS       = True     # Try Jenks natural breaks, else fallback to percentiles
PRIORITY_BINS        = 4        # How many classes for priority

# ===== Sampling Simulation Parameters =====
DEFAULT_EPS          = 5.0      # Target 95% CI relative error (%)
DEFAULT_BOOT         = 1000     # Bootstrap iterations
DEFAULT_W            = "0.25,0.25,0.25,0.25" # DJF, MAM, JJA, SON

# ===== Input/Output Filenames =====
DEFAULT_INPUT_FILE   = 'final/output_ch4_flux_qc.csv'
DEFAULT_OUTPUT_DIR   = 'ch4_sampling_result'
SUPPORTED_FORMATS    = ['.csv', '.xlsx']

# ===== Monitoring & Log =====
PROGRESS_JSON        = SAMPLING_OUT_DIR / "progress.json"
LOG_PATH             = SAMPLING_OUT_DIR / "progress.log"
PID_PATH             = SAMPLING_OUT_DIR / "pid.txt"

# ===== Helper: Region filter =====
def in_region(grid_id: str, lat_min=EU_LAT_MIN, lat_max=EU_LAT_MAX, lon_min=EU_LON_MIN, lon_max=EU_LON_MAX) -> bool:
    """Check if grid_id ('lat_lon' string) falls within region."""
    try:
        lat_str, lon_str = grid_id.split("_")
        lat, lon = float(lat_str), float(lon_str)
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    except Exception:
        return False

