import pandas as pd
import numpy as np

# CSV preprocessing
df = pd.read_csv("/Users/tomdoran/Library/CloudStorage/OneDrive-UniversityofGreenwich/FINAL YEAR PROJECT/SimRacingProject_v1/data/silverstone_20laps_p911gt3_v1.csv",
                 skiprows=14)
df = df.dropna()
df = df.iloc[1:].reset_index(drop=True)
df = df.apply(pd.to_numeric)
df["lap"] = (df["Lap Distance"] < df["Lap Distance"].shift(1)).cumsum()
df = df[df["lap"] != 0]
df["Throttle"] = df["Throttle"] / 100
df["Brake"] = df["Brake"] / 100
df = df[df["lap"] != 21]

lap_counts = df["lap"].value_counts()
lap_groups = df.groupby("lap")

valid_laps = lap_counts[lap_counts > 100].index
df = df[df["lap"].isin(valid_laps)]

# Feature table
features = []
for lap, data in lap_groups:
    lap_time = len(data) / 20 # because of 20hz export

    features.append({
      "lap": lap,
        "lap_time": lap_time,
        "avg_speed": data["Speed"].mean(),
        "max_speed": data["Speed"].max(),
        "avg_throttle": data["Throttle"].mean(),
        "avg_brake": data["Brake"].mean(),
        "steering_variance": data["Steering Wheel Angle"].var()
    })
features_df = pd.DataFrame(features)
print(features_df.head())
print(df.head())
print(df.dtypes)
print(df["lap"].value_counts().sort_index())
