import os
import pandas as pd

def load_lap_data(filename):
    # BASE_DIR = folder above src
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", filename)
    data = pd.read_csv(csv_path)
    return data

# Call the function correctly with just the CSV file name
df = load_lap_data("raw_lap_data.csv")
print(df.head())
