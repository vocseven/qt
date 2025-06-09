import pandas as pd
import glob
import os

DATA_DIR = 'datasets_2025June6'
output_file = os.path.join(DATA_DIR, 'merged_30min.csv')

files = sorted(glob.glob(os.path.join(DATA_DIR, '*30.csv')))

all_dfs = []
for file in files:
    base = os.path.basename(file)
    name = base.split(',')[0]
    ticker = name.split('_')[-1]
    df = pd.read_csv(file)
    df['ticker'] = ticker
    all_dfs.append(df)

if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(output_file, index=False)
    print(f"Merged {len(files)} files into {output_file}")
else:
    print("No 30-minute files found.")
