import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from load_data import load_lap_data
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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

# Create s1 predicting model
model_s1 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Train model
model_s1.fit(X, y_s1)
X_text = X.iloc[[0]]
s1_pred = model_s1.predict(X_text)[0]

# Create s2 predicting model
model_s2 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Train model
model_s2.fit(X, y_s2)
X_text = X.iloc[[0]]
s2_pred = model_s2.predict(X_text)[0]

# Create s3 predicting model
model_s3 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Train model
model_s3.fit(X, y_s3)
X_text = X.iloc[[0]]
s3_pred = model_s3.predict(X_text)[0]

# Predict a lap
X_test = X.iloc[[0]]

# Predict all laps using sectors
s1_preds = model_s1.predict(X)
s2_preds = model_s2.predict(X)
s3_preds = model_s3.predict(X)

# Reconstruct lap time using sectors

lap_preds = s1_preds + s2_preds + s3_preds # Array of length 60
lap_actual = y_s1.iloc[0] + y_s2.iloc[0] + y_s3.iloc[0]

lap_preds_rounded = np.round(lap_preds, 3)
diff = abs(lap_preds[0]- y.iloc[0])

# Lap 1 predicted time
print('\nPredicted lap time: ', np.round(lap_preds[0], 3))
print('Actual lap time: ', y.iloc[0])
print('Diff = ', round(diff, 2))

# print first 5 lap predictions
print("\nFirst 5 laps predicted vs actual: ")
for i in range(5):
    print(f"Lap {i+1} Predicted: {round(lap_preds[i], 2)}, Actual: {round(df['lap_time'].iloc[i], 2)}")

# Lap 1 Sector Predictions
print('\nSector predictions: ')
print('S1 Prediction: ', np.round(s1_preds[0], 2), 'Actual S1 time: ', y_s1.iloc[0])
print('S2 Prediction: ', np.round(s2_preds[0], 2), 'Actual S2 time: ', y_s2.iloc[0])
print('S3 Prediction: ', np.round(s3_preds[0], 2), 'Actual S3 time: ', y_s3.iloc[0])

