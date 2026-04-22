import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
plt.ioff()

# Load the data from the csv
df = pd.read_csv("/Users/tomdoran/Library/CloudStorage/OneDrive-UniversityofGreenwich/FINAL YEAR PROJECT/SimRacingProject_v1/data/silverstone_20laps_p911gt3_v1.csv", skiprows=14)
df = df.dropna()
df = df.iloc[1:].reset_index(drop=True)
df = df.apply(pd.to_numeric)
df["lap"] = (df["Lap Distance"] < df["Lap Distance"].shift(1)).cumsum()
df = df[df["lap"] != 0]
df["Throttle"] = df["Throttle"] / 100
df["Brake"] = df["Brake"] / 100
df = df[df["lap"] != 21]

# Lap filtering
lap_counts = df["lap"].value_counts()
valid_laps = lap_counts[lap_counts > 100]. index
df = df[df["lap"].isin(valid_laps)]

# Sector calculation
df["lap_progress"] = df["Lap Distance"] / df.groupby("lap")["Lap Distance"].transform("max")

def assign_sector(p):
    if p < 0.33:
        return "s1"
    elif p < 0.66:
        return "s2"
    else:
        return "s3"

df["sector"] = df["lap_progress"].apply(assign_sector)

# Select features and target
sector_df = df.groupby(["lap", "sector"]).agg({
    "Speed": "mean",
    "Throttle": "mean",
    "Brake": "mean",
    "Steering Wheel Angle": "var"
}).reset_index()

sector_pivot = sector_df.pivot(index="lap", columns="sector")
sector_pivot.columns = [f"{c[1]}_{c[0]}" for c in sector_pivot.columns]
sector_pivot = sector_pivot.reset_index()


sector_time = df.groupby(["lap", "sector"]).size().unstack() / 20
sector_time.columns = ["s1_time", "s2_time", "s3_time"]
sector_time = sector_time.reset_index()

final_df = pd.merge(sector_pivot, sector_time, on="lap")

X = final_df.drop(columns=["lap", "s1_time", "s2_time", "s3_time"])

y_s1 = final_df["s1_time"]
y_s2 = final_df["s2_time"]
y_s3 = final_df["s3_time"]

# Ensures no data leakage and prevents model from cheating
assert 's1_time' not in X.columns
assert 's2_time' not in X.columns
assert 's3_time' not in X.columns

# Split the data into training and test sets for each sector
X_train, X_test, idx_train, idx_test = train_test_split(
    X, range(len(X)), test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

y_train_s1, y_test_s1 = y_s1.iloc[idx_train], y_s1.iloc[idx_test]
y_train_s2, y_test_s2 = y_s2.iloc[idx_train], y_s2.iloc[idx_test]
y_train_s3, y_test_s3 = y_s3.iloc[idx_train], y_s3.iloc[idx_test]

# Create sector predicting models
model_s1 = RandomForestRegressor(
    n_estimators=300, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2)
model_s2 = RandomForestRegressor(
    n_estimators=300, random_state=42,  max_depth=10, min_samples_split=5, min_samples_leaf=2)
model_s3 = RandomForestRegressor(
    n_estimators=300, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2)

# Fit all models using only training data
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

lap_preds_rounded = np.round(lap_preds, 3)
diff = abs(lap_preds[0]- lap_actual.iloc[0])

# Baseline linear regression model

lin_s1 = LinearRegression()
lin_s2 = LinearRegression()
lin_s3 = LinearRegression()

lin_s1.fit(X_train, y_train_s1)
lin_s2.fit(X_train, y_train_s2)
lin_s3.fit(X_train, y_train_s3)

lin_pred_s1 = lin_s1.predict(X_test)
lin_pred_s2 = lin_s2.predict(X_test)
lin_pred_s3 = lin_s3.predict(X_test)

lin_lap_preds = lin_pred_s1 + lin_pred_s2 + lin_pred_s3

print("\n" +"="*50)
print("Linear Regression Baseline")
print("="*50)
mae = mean_absolute_error(lap_actual, lin_lap_preds)
r2 = r2_score(lap_actual, lin_lap_preds)
print(f"Lap MAE = {mae:.3f}")
print(f"Lap R2 = {r2:.3f}")

# Evaluate each sector model
print("\n" +"="*50)
print(" Sector Model Performance")
print("="*50)
for name, y_true, y_pred in zip(
    ['Sector 1', 'Sector 2', 'Sector 3'],
    [y_test_s1, y_test_s2, y_test_s3],
    [s1_preds, s2_preds, s3_preds]
):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE = {mae:.3f}, R2 = {r2:.3f}")

lap_mae = mean_absolute_error(lap_actual, lap_preds)
lap_r2 = r2_score(lap_actual, lap_preds)
print("\n" +"="*50)
print(f"Total Lap Prediction Performance")
print("="*50)
print(f"MAE = {lap_mae:.3f}, R2 = {lap_r2:.3f}")

# Lap predicted time
print("\nLap Comparison (Example lap)")
print("="*50)
print(f'Predicted: {lap_preds[0]:.3f}s')
print(f'Actual lap time: {lap_actual.iloc[0]:.3f}s')
print(f'Delta: {diff:.3f}s')

# print first 4 lap predictions
print("\n" +"="*50)
print("First 4 lap comparison")
print("="*50)
for i in range(min(5, len(lap_preds))):
    print(f"Lap {i+1:<2} Predicted: {lap_preds[i]:.2f}s, Actual: {lap_actual.iloc[i]:.2f}s")

# Lap 1 Sector Predictions
print("\n" +"="*50)
print('Sector Breakdown (Example lap)')
print("="*50)
print('S1 Prediction: ', np.round(s1_preds[0], 2), 'Actual S1 time: ', y_s1.iloc[0])
print('S2 Prediction: ', np.round(s2_preds[0], 2), 'Actual S2 time: ', y_s2.iloc[0])
print('S3 Prediction: ', np.round(s3_preds[0], 2), 'Actual S3 time: ', y_s3.iloc[0])

# Cross validation
print("\n" +"="*50)
print("Cross Validation Results (5-fold)")
print("="*50)

for name, target in zip(
    ['Sector 1', 'Sector 2', 'Sector 3'],
    [y_s1, y_s2, y_s3]
):
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    scores = cross_val_score(
        model,
        X,
        target,
        cv=5,
        scoring='r2'
    )

    print(f"\n{name}")
    print("R2 Scores = ", np.round(scores, 3))
    print('Mean R2 = ', round(np.mean(scores), 3))
    print('Std Dev = ', round(np.std(scores), 3))

def predict_lap(row_index=0):
    sample = X.iloc[[row_index]]

    s1 = model_s1.predict(sample)[0]
    s2 = model_s2.predict(sample)[0]
    s3 = model_s3.predict(sample)[0]

    return round(s1+ s2+ s3, 3)

predicted_lap = predict_lap(0)
print("\n" +"="*50)
print("New lap prediction tool")
print("="*50)
print("Predicted lap time:", predicted_lap, "seconds")


# Feature Importance
print("\n=== Feature Importance by sector ===")

for name, sector_model in zip(
    ["Sector 1", "Sector 2", "Sector 3"],
    [model_s1, model_s2, model_s3]
):
    print(f"\n{name}")

    importances = sector_model.feature_importances_

    # sort telemetry by importance relevant to lap time influence
    sorted_features = sorted(
        zip(X.columns, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in sorted_features:
        print(f"{feature}: {round(importance, 3)}")

# Driver feedback tool!!!

# find best lap
best_idx = np.argmin(lap_actual.values)
best_lap = X_test.mean()

# manually selected from Silverstone International Circuit
track_sections = {
    "s1": "Turns 1-4",
    "s2": "Turns 5-6",
    "s3": "Turns 7-9",
}

print("\n" +"="*50)
print("Driver Feedback Analysis")
print("="*50)

def driver_feedback(lap_index):
    current = X_test.iloc[lap_index]
    diff = current - best_lap
    feedback = {"s1": [], "s2": [], "s3": []}
    for i, val in enumerate(diff):
        feature = X.columns[i]

        if abs(val) <= 0.015:
            continue

        sector = feature.split("_")[0]
        metric = "_".join(feature.split("_")[1:])

        # Speed
        if metric == "Speed":
            if val < 0:
                msg = "Exit speed is lower than optimal, carry more speed through the corner."
            else:
                msg = "Entry speed too high, brake slightly earlier for better control."
        # Throttle
        elif metric == "Throttle":
            if val < 0:
                msg = "Throttle applied too late, get on power earlier."
            else:
                msg = "Throttle too aggressive, apply power more smoothly."
        # Brake
        elif metric == "Brake":
            if val < 0:
                msg = "Braking too heavy, be smoother on the pedal."
            else:
                msg = "Braking too light, brake earlier and harder."
        # Steering
        elif "Steering" in metric:
            msg = "Steering input inconsistent, focus on smoother turn-in."
        else:
            continue

        if msg not in feedback[sector]:
            feedback[sector].append(msg)

    # Print improvements
    printed = False
    for sector in ["s1", "s2", "s3"]:
        if feedback[sector]:
            printed = True
            print(f"\n{sector.upper()} ({track_sections[sector]}):")

            # Limit improvements to 2
            for msg in feedback[sector][:2]:
                print(f"  • {msg}")

    if not printed:
        print("\nLap is very close to optimal — no major issues detected.")

driver_feedback(0)

# Show summary stats
residuals = lap_actual - lap_preds
print("\nResidual mean: ", round(np.mean(residuals),3))
print("Residual standard deviation: ", round(np.std(residuals),3))

# Feature Importance Visualisation
importances = model_s1.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(X.columns)[sorted_idx]
plt.figure(figsize=(8,5))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='skyblue')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel("Feature Importance Score")
plt.title("Feature Importance for Lap Time Prediction")
plt.tight_layout()
plt.show()


# Residual Error analysis
plt.figure(figsize=(8, 5))
plt.scatter(lap_preds, residuals, alpha=0.7, color='orange', edgecolor='k')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel("Predicted lap time")
plt.ylabel("Residual (Actual - predicted)")
plt.title("Residual analysis for Lap time predictions")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Actual vs Predicted plot
plt.figure(figsize=(10,5))
plt.plot(lap_actual.values, label="Actual lap time", marker = 'o')
plt.plot(lap_preds, label="Predicted Lap Time", marker = 'x')
plt.xlabel("Lap number")
plt.ylabel("Lap time (s)")
plt.title("Actual vs Predicted lap times")
plt.legend()
plt.grid(True)
plt.show()





