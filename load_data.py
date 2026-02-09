import pandas as pd

def load_lap_data(path):
  data = pd.read_csv(path)
  return data

if __name == "__main__":
  df = load_lap_data("data/lap_data.csv")
  print(df.head())
