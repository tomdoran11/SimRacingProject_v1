import pandas as pd
from sklearn.model_selection import train_test_split
from load_data import load_lap_data
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the data from the csv
df = load_lap_data("raw_lap_data.csv")

# Select features and target
features = ['avg_throttle', 'avg_brake', 'avg_speed', 'tire_wear', 'rain_percentage']
target = "lap_time"

X = df[features]
y = df[target]

y_s1 = df['sector_1_time']
y_s2 = df['sector_2_time']
y_s3 = df['sector_3_time']

# Ensures no data leakage and prevents model from cheating
assert 'sector_1_time' not in X.columns
assert 'sector_2_time' not in X.columns
assert 'sector_3_time' not in X.columns

# Split the data into training and test sets for each sector
X_train, X_test, idx_train, idx_test = train_test_split(
    X, range(len(X)), test_size=0.2, random_state=42
)
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

lap_preds_rounded = np.round(lap_preds, 3)
diff = abs(lap_preds[0]- y.iloc[0])

# Evaluate each sector model
print("\n=== Sector Model Performance ===")
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
print(f"\n=== Total Lap Prediction Performance ===")
print(f"MAE = {lap_mae:.3f}, R2 = {lap_r2:.3f}")

# Lap 1 predicted time
print('\nPredicted lap time: ', np.round(lap_preds[0], 3))
print('Actual lap time: ', y.iloc[0])
print('Diff = ', round(diff, 2))

# print first 5 lap predictions
print("\n=== First 5 laps predicted vs actual: ===")
for i in range(5):
    print(f"Lap {i+1} Predicted: {round(lap_preds[i], 2)}, Actual: {round(df['lap_time'].iloc[i], 2)}")

# Lap 1 Sector Predictions
print('\n=== Sector predictions: ===')
print('S1 Prediction: ', np.round(s1_preds[0], 2), 'Actual S1 time: ', y_s1.iloc[0])
print('S2 Prediction: ', np.round(s2_preds[0], 2), 'Actual S2 time: ', y_s2.iloc[0])
print('S3 Prediction: ', np.round(s3_preds[0], 2), 'Actual S3 time: ', y_s3.iloc[0])

# Feature Importance Visualisation
importances = model_s1.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(features)[sorted_idx]
plt.figure(figsize=(8,5))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='skyblue')
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.xlabel("Feature Importance Score")
plt.title("Feature Importance for Sector 1 Lap Time Prediction")
plt.tight_layout()
plt.show()

