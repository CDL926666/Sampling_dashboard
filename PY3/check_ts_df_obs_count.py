import pandas as pd

df = pd.read_csv("ch4_sampling_result/ts_df.csv")
print(df['obs_count'].describe())
print((df['obs_count'] >= 12).sum(), "grids have >=12 months")
print((df['obs_count'] < 12).sum(), "grids have <12 months")
