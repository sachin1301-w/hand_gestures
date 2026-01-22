import pandas as pd
import glob

files = glob.glob("data/*.csv")
df_list = []

for file in files:
    df = pd.read_csv(file, header=None)
    df_list.append(df)

final_df = pd.concat(df_list, ignore_index=True)
final_df.to_csv("all_gestures.csv", index=False)

print("All CSV files merged into all_gestures.csv")
