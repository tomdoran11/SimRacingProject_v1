import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from load_data import load_lap_data

# Load the data from the csv
df = load_lap_data("raw_lap_data.csv")

# Select features and target
features = ['avg_throttle', 'avg_brake', 'avg_speed']
target = "lap_time"

X = df[features]
y = df[target]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and check accuracy
predictions = model.predict(X_test)
print("Predictions: ", predictions[:5])
print("Actual: ", y_test[:5].values)
print("Model score (R^2): ", model.score(X_test, y_test))