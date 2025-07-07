import os
import pandas as pd
import numpy as np
import time
import warnings

# ===== 全局与参数区（支持后续扩展yaml/json外部配置） =====
INPUT_FILE = 'final/output_ch4_flux_qc.csv'
OUTPUT_DIR = 'ch4_sampling_result'
GRID_SIZE = 1.0                  # 空间聚合网格粒度
MIN_OBS_TREND = 5                # 判定趋势分析所需最小观测数
SUPPORTED_FORMATS = ['.csv', '.xlsx']
MAX_OUTLIER_SIGMA = 5            # 极端值过滤标准
COORD_SYSTEM = 'WGS84'           # 坐标系，预留投影接口
TIMEZONE = 'UTC'                 # 保证所有时间统一为UTC
ENABLE_HEX = False               # 预留蜂窝/自适应网格接口
DROP_DUPLICATE = True            # 是否剔除重复点

# >>> MOD: 英国经纬度边界（英国＋近海）
UK_LAT_MIN, UK_LAT_MAX = 49.0, 61.0      # 北纬
UK_LON_MIN, UK_LON_MAX = -11.0, 2.0      # 西经为负

os.makedirs(OUTPUT_DIR, exist_ok=True)

def timer(msg):
    t = time.perf_counter()
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg} ...")
    return t
def elapsed(start, msg):
    print(f"    -> {msg} finished in {time.perf_counter() - start:.2f} s.")

def read_and_standardize_data(file_path):
    """多格式兼容 + 字段/时间/单位标准化 + 基础异常校验"""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # ---- 字段标准化 ----
    rename_dict = {
        'lat':'latitude', 'long':'longitude', 'lon':'longitude',
        'LAT':'latitude', 'LON':'longitude', '经度':'longitude', '纬度':'latitude',
        'ch4':'ch4_emis_total', 'ch4_flux':'ch4_emis_total', '甲烷':'ch4_emis_total'
    }
    df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
    # 必要字段校验
    for col in ['time', 'latitude', 'longitude', 'ch4_emis_total']:
        if col not in df.columns:
            raise ValueError(f"Missing required field: {col}")

    # ---- 时间/坐标标准化 ----
    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['ch4_emis_total'] = pd.to_numeric(df['ch4_emis_total'], errors='coerce')
    return df

# ========== 1. 数据读取与初步科学清洗 ==========
t0 = timer("Step 1: Load & robust clean data")
df = read_and_standardize_data(INPUT_FILE)

# ====== 空值/极值/异常点归档处理 ======
critical_fields = ['time', 'latitude', 'longitude', 'ch4_emis_total']
n_init = len(df)
df = df.dropna(subset=critical_fields)
n_drop_na = n_init - len(df)

# >>> MOD: 仅保留英国范围数据
df = df[
    (df['latitude'].between(UK_LAT_MIN, UK_LAT_MAX)) &
    (df['longitude'].between(UK_LON_MIN, UK_LON_MAX))
]

# 其余校验保持原逻辑
df = df[df['ch4_emis_total'].notnull() & (df['ch4_emis_total'] >= 0) & np.isfinite(df['ch4_emis_total'])]
if DROP_DUPLICATE:
    df = df.drop_duplicates(subset=['time', 'latitude', 'longitude'])

# ---- 极端值剔除 + 类型归档 ----
mean_ch4 = df['ch4_emis_total'].mean()
std_ch4 = df['ch4_emis_total'].std()
upper_outlier = mean_ch4 + MAX_OUTLIER_SIGMA * std_ch4
outlier_mask = (df['ch4_emis_total'] > upper_outlier)
df_outlier = df[outlier_mask].copy()
df = df[~outlier_mask]
n_outlier = len(df_outlier)
if n_outlier > 0:
    df_outlier['outlier_reason'] = f'>{MAX_OUTLIER_SIGMA}σ'
    df_outlier.to_csv(f"{OUTPUT_DIR}/excluded_outlier_samples.csv", index=False)

if len(df) == 0:
    raise ValueError("No valid data after cleaning. Please check input file, format or data quality.")

print(f"  Records loaded: {len(df)} (removed {n_drop_na} null, {n_outlier} extreme outlier)")
print(f"  Time: {df['time'].min().date()} to {df['time'].max().date()}, "
      f"Lat: {df['latitude'].min()}~{df['latitude'].max()}, "
      f"Lon: {df['longitude'].min()}~{df['longitude'].max()}")
elapsed(t0, "Step 1")

# ========== 2. 空间聚合与科学统计 ==========
t1 = timer("Step 2: Spatial aggregation & robust stats")

# ---- 支持后续升级（蜂窝/自适应/掩膜聚合等接口预留） ----
def grid_id(lat, lon, grid_size):
    return f"{np.round(lat/grid_size, 3):.3f}_{np.round(lon/grid_size, 3):.3f}"

df['grid_id'] = df.apply(lambda r: grid_id(r['latitude'], r['longitude'], GRID_SIZE), axis=1)

# ---- 灵活/丰富的分布统计量聚合 ----
agg = df.groupby(['grid_id', 'latitude', 'longitude']).agg(
    mean_ch4=('ch4_emis_total', 'mean'),
    median_ch4=('ch4_emis_total', 'median'),
    std_ch4=('ch4_emis_total', 'std'),
    var_ch4=('ch4_emis_total', 'var'),
    obs_count=('ch4_emis_total', 'count'),
    min_ch4=('ch4_emis_total', 'min'),
    max_ch4=('ch4_emis_total', 'max'),
    q5_ch4=('ch4_emis_total', lambda x: x.quantile(0.05)),
    q10_ch4=('ch4_emis_total', lambda x: x.quantile(0.10)),
    q25_ch4=('ch4_emis_total', lambda x: x.quantile(0.25)),
    q75_ch4=('ch4_emis_total', lambda x: x.quantile(0.75)),
    q90_ch4=('ch4_emis_total', lambda x: x.quantile(0.90)),
    q95_ch4=('ch4_emis_total', lambda x: x.quantile(0.95)),
    skew_ch4=('ch4_emis_total', lambda x: x.skew()),
    kurt_ch4=('ch4_emis_total', lambda x: x.kurt()),
    obs_interval_mean=('time', lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else np.nan),
).reset_index()

# ---- 稀疏度与地理标志 ----
agg['data_quality_flag'] = np.where(agg['obs_count'] < MIN_OBS_TREND, 'sparse', 'ok')

# ========== 3. 科学时序导出 ==========
ts_save_rows = []
for _, row in agg.iterrows():
    sub = df[(df['latitude'] == row['latitude']) & (df['longitude'] == row['longitude'])].sort_values('time')
    y = sub['ch4_emis_total'].values
    x_time = sub['time'].values
    ts_save_rows.append({
        "grid_id": row['grid_id'],
        "latitude": row['latitude'],
        "longitude": row['longitude'],
        "obs_count": len(y),
        "first_time": pd.Timestamp(x_time[0]).strftime('%Y-%m-%d') if len(x_time) else '',
        "last_time":  pd.Timestamp(x_time[-1]).strftime('%Y-%m-%d') if len(x_time) else '',
        "time_series": '|'.join([f"{pd.Timestamp(t).strftime('%Y-%m-%d')},{v:.6e}" for t, v in zip(x_time, y)])
    })

pd.DataFrame(ts_save_rows).to_csv(f"{OUTPUT_DIR}/ts_df.csv", index=False)
agg.to_csv(f'{OUTPUT_DIR}/agg.csv', index=False)
elapsed(t1, "Step 2")

print(f"Data aggregation and series export complete: {len(agg)} grid cells, "
      f"results written to {OUTPUT_DIR}/agg.csv and ts_df.csv\n")
if n_outlier > 0:
    print(f"  {n_outlier} records flagged as outliers and written to {OUTPUT_DIR}/excluded_outlier_samples.csv")
