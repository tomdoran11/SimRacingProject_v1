import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from load_data import load_lap_data
from sklearn.ensemble import RandomForestRegressor

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
s1_pred = model_s1.predict(X_text)[0]
s2_pred = model_s2.predict(X_text)[0]
s3_pred = model_s3.predict(X_text)[0]

# Reconstruct lap time using sectors

lap_pred = s1_pred + s2_pred + s3_pred
lap_actual = y_s1.iloc[0] + y_s2.iloc[0] + y_s3.iloc[0]

print('S1 Prediction: ', round(s1_pred, 2), 'Actual S1 time: ', y_s1.iloc[0])
print('S2 Prediction: ', round(s2_pred, 2), 'Actual S2 time: ', y_s2.iloc[0])
print('S3 Prediction: ', round(s3_pred, 2), 'Actual S3 time: ', y_s3.iloc[0])
print('Predicted lap time: ', round(lap_pred, 3))
print('Actual lap time: ', round(lap_actual, 3))
