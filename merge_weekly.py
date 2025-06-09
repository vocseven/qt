import pandas as pd
import glob
import os

DIR = 'datasets_2025June6'

weekly_files = sorted(glob.glob(os.path.join(DIR, '*1W.csv')))
frames = []
for path in weekly_files:
    ticker = os.path.basename(path).split(',')[0].split('_')[-1]
    df = pd.read_csv(path)
    df['ticker'] = ticker
    frames.append(df)

if frames:
    merged = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(DIR, 'weekly_merged.csv')
    merged.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(merged)} rows")
else:
    print('No weekly files found')
