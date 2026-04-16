
#### CODE CHUNKS THAT HAVE BEEN REMOVED
#### FROM MAIN CODE - KEPT IN CASE NEEDED AGAIN

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

block

y_train_s1, y_test_s1 = y_s1.iloc[idx_train], y_s1.iloc[idx_test]
y_train_s2, y_test_s2 = y_s2.iloc[idx_train], y_s2.iloc[idx_test]
y_train_s3, y_test_s3 = y_s3.iloc[idx_train], y_s3.iloc[idx_test]

# Create sector predicting models
model_s1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_s2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_s3 = RandomForestRegressor(n_estimators=100, random_state=42)

#Fit all models using only training data
model_s1.fit(X_train, y_train_s1)
model_s2.fit(X_train, y_train_s2)
model_s3.fit(X_train, y_train_s3)

# Predict all laps using sectors
s1_preds = model_s1.predict(X_test)
s2_preds = model_s2.predict(X_test)
s3_preds = model_s3.predict(X_test)

# Reconstruct lap time using test set values
lap_preds = s1_preds + s2_preds + s3_preds # Array of length 60
lap_actual = y_test_s1 + y_test_s2 + y_test_s3

# Lap time prediction function
# predicted = predict_lap(
    throttle=0.69,
    brake=0.11,
    speed=144.0,
    steering_variance = 0.34
)


X_train, X_test, y_s1_train, y_s1_test = train_test_split(
    X, y_s1, test_size=0.2, random_state=42
)

_, _, y_s2_train, y_s2_test = train_test_split(
    X, y_s2, test_size=0.2, random_state=42
)

_, _, y_s3_train, y_s3_test = train_test_split(
    X, y_s3, test_size=0.2, random_state=42
)

def driver_feedback(lap_index):
    current = X_test.iloc[lap_index]
    diff = current - best_lap

    for i, val in enumerate(diff):
        feature = X.columns[i]

        if abs(val) <= 0.015:
            continue

        sector = feature.split("_")[0]
        metric = "_".join(feature.split("_")[1:])
        feedback = {"s1": [], "s2": [], "s3": []}
        if msg not in feedback[sector]:
            feedback[sector].append(msg)

        # Throttle
        if metric == "Throttle":
            if val < 0:
                feedback[sector].append(f"{track_sections[sector]}): Throttle applied too late, try pushing the throttle earlier coming out of the corner for extra speed.")
            else:
                feedback[sector].append(f"{track_sections[sector]}): Throttle applied too aggresively, try pushing the throttle more gently coming out of the corner for better control.")

        # Speed
        elif metric == "Speed":
            if val < 0:
                feedback[sector].append(f"{track_sections[sector]}): Exit speed from the corner is too low, try carrying more speed through the corner for more speed onto the straight.")
            else:
                feedback[sector].append(f"{track_sections[sector]}): Corner entry speed is too hugh, brake slightly earlier for better control and stability.")

        # Braking
        elif metric == "Brake":
            if val < 0:
                feedback[sector].append(f"{track_sections[sector]}): Braking is too heavy, try being smoother on the brake pedal.")
            else:
                feedback[sector].append(f"{track_sections[sector]}): Braking is too light, try braking earlier and harder for the corner.")

        # Steering
        elif "Steering" in metric:
            if val < 0:
                feedback[sector].append(f"{track_sections[sector]}): Steering input is not consistent, try to turn the wheel smoothly and don't rush turning into corners.")
        if np.all(np.abs(diff.values)<+ 0.02):
            feedback[sector].append(f"\nNo major issues with lap time detected, well done!!!")

    for sector in ["s1", "s2", "s3"]:
        if feedback[sector]:
            print(f"\n{sector.upper()} ({track_sections[sector]}")
            # Limit to 2 improvements
            for msg in feedback[sector][:2]:
                print(f"  . {msg}")
driver_feedback(0)