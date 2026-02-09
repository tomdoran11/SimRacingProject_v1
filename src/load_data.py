import pandas as pd

def load_lap_data(path):
  data = pd.read_csv(path)
  return data

if __name__ == "__main__":
  df = load_lap_data("data/raw_lap_data.csv")
  print(df.head())