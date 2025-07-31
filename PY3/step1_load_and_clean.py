# Y:\Bishe_project\PY3\step1_load_and_clean.py

import os
import pandas as pd
import numpy as np

DEFAULT_INPUT_FILE = 'final/output_ch4_flux_qc.csv'
DEFAULT_OUTPUT_DIR = 'ch4_sampling_result'
SUPPORTED_FORMATS = ['.csv', '.xlsx']

# 欧洲范围（含英国和大陆）
EU_LAT_MIN, EU_LAT_MAX = 35.0, 72.0
EU_LON_MIN, EU_LON_MAX = -25.0, 45.0

GRID_SIZE = 2.0
MIN_OBS_TREND = 5
MAX_OUTLIER_SIGMA = 5
DROP_DUPLICATE = True

# 加载数据文件
def load_data(input_path):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path, low_memory=False, encoding='utf-8-sig')
    elif ext == ".xlsx":
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return df

# 数据清洗
def clean_data(
    df,
    drop_duplicate=DROP_DUPLICATE,
    max_outlier_sigma=MAX_OUTLIER_SIGMA,
    lat_min=EU_LAT_MIN,
    lat_max=EU_LAT_MAX,
    lon_min=EU_LON_MIN,
    lon_max=EU_LON_MAX,
):
    # 字段名标准化
    rename_dict = {
        'lat': 'latitude', 'long': 'longitude', 'lon': 'longitude',
        'LAT': 'latitude', 'LON': 'longitude', '经度': 'longitude', '纬度': 'latitude',
        'ch4': 'ch4_emis_total', 'ch4_flux': 'ch4_emis_total', '甲烷': 'ch4_emis_total'
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    required_fields = ['time', 'latitude', 'longitude', 'ch4_emis_total']
    for col in required_fields:
        if col not in df.columns:
            raise ValueError(f"Missing required field: {col}")
    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['ch4_emis_total'] = pd.to_numeric(df['ch4_emis_total'], errors='coerce')
    n_init = len(df)
    df = df.dropna(subset=required_fields)
    n_drop_na = n_init - len(df)
    df = df[
        (df['latitude'].between(lat_min, lat_max)) &
        (df['longitude'].between(lon_min, lon_max))
    ]
    df = df[df['ch4_emis_total'].notnull() & (df['ch4_emis_total'] >= 0) & np.isfinite(df['ch4_emis_total'])]
    if drop_duplicate:
        df = df.drop_duplicates(subset=['time', 'latitude', 'longitude'])
    mean_ch4 = df['ch4_emis_total'].mean()
    std_ch4 = df['ch4_emis_total'].std()
    upper_outlier = mean_ch4 + max_outlier_sigma * std_ch4
    outlier_mask = (df['ch4_emis_total'] > upper_outlier)
    df_outlier = df[outlier_mask].copy()
    df = df[~outlier_mask]
    n_outlier = len(df_outlier)
    return df, df_outlier, n_drop_na, n_outlier

# 空间聚合统计
def spatial_aggregate(df, grid_size=GRID_SIZE, min_obs_trend=MIN_OBS_TREND):
    def grid_id(lat, lon, grid_size):
        return f"{np.round(lat/grid_size, 3):.3f}_{np.round(lon/grid_size, 3):.3f}"
    df['grid_id'] = df.apply(lambda r: grid_id(r['latitude'], r['longitude'], grid_size), axis=1)
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
    agg['data_quality_flag'] = np.where(agg['obs_count'] < min_obs_trend, 'sparse', 'ok')
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
            "last_time": pd.Timestamp(x_time[-1]).strftime('%Y-%m-%d') if len(x_time) else '',
            "time_series": '|'.join([f"{pd.Timestamp(t).strftime('%Y-%m-%d')},{v:.6e}" for t, v in zip(x_time, y)])
        })
    return agg, ts_save_rows

def main(
    input_file=DEFAULT_INPUT_FILE,
    output_dir=DEFAULT_OUTPUT_DIR,
    grid_size=GRID_SIZE,
    min_obs_trend=MIN_OBS_TREND,
    max_outlier_sigma=MAX_OUTLIER_SIGMA,
    lat_min=EU_LAT_MIN,
    lat_max=EU_LAT_MAX,
    lon_min=EU_LON_MIN,
    lon_max=EU_LON_MAX
):
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_data(input_file)
    cleaned_df, outlier_df, n_drop_na, n_outlier = clean_data(
        raw_df,
        drop_duplicate=DROP_DUPLICATE,
        max_outlier_sigma=max_outlier_sigma,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max
    )
    if len(cleaned_df) == 0:
        raise ValueError("No valid data after cleaning. Please check input file, format or data quality.")
    if n_outlier > 0:
        outlier_df['outlier_reason'] = f'>{max_outlier_sigma}σ'
        outlier_df.to_csv(f"{output_dir}/excluded_outlier_samples.csv", index=False)
    agg, ts_save_rows = spatial_aggregate(cleaned_df, grid_size=grid_size, min_obs_trend=min_obs_trend)
    pd.DataFrame(ts_save_rows).to_csv(f"{output_dir}/ts_df.csv", index=False)
    agg.to_csv(f'{output_dir}/agg.csv', index=False)
    print(f"Valid data after cleaning: {len(cleaned_df)}")

if __name__ == "__main__":
    main()
