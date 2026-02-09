import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from load_data import load_lap_data

# Load the data from the csv
df = load_lap_data("raw_lap_data.csv")

