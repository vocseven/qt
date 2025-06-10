import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load new merged datasets
base_dir = 'datasets_2025June6'
path_daily = f"{base_dir}/merged_daily.csv"
path_weekly = f"{base_dir}/weekly_merged.csv"
path_30 = f"{base_dir}/merged_30min.csv"

print("Loading datasets...")
df_daily = pd.read_csv(path_daily)
df_weekly = pd.read_csv(path_weekly)
df_30 = pd.read_csv(path_30)

# Preprocess daily data
print("Preprocessing daily data...")

def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    return df

df_daily = clean_columns(df_daily)

# Convert epoch to pandas datetime
df_daily['date'] = pd.to_datetime(df_daily['time'], unit='s')
df_daily = df_daily.sort_values(['ticker', 'date'])

# Compute next day return per ticker
df_daily['next_close'] = df_daily.groupby('ticker')['close'].shift(-1)
df_daily['target_ret'] = df_daily['next_close'] / df_daily['close'] - 1
df_daily.dropna(subset=['target_ret'], inplace=True)

# Feature set: choose numeric columns except time and next_close
feature_cols = [c for c in df_daily.columns if c not in ['time', 'date', 'next_close', 'target_ret', 'ticker']]
X = df_daily[feature_cols]
y = df_daily['target_ret']

# Simple train/test split
X_train, X_test, y_train, y_test, tickers_train, tickers_test = train_test_split(
    X, y, df_daily['ticker'], test_size=0.2, random_state=42, shuffle=True
)

model = RandomForestRegressor(n_estimators=50, random_state=42)
print("Training model...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(((y_test - preds) ** 2).mean())
print(f"Test RMSE: {rmse:.6f}")

# Construct portfolio from latest available observations in test set
results = pd.DataFrame({
    'ticker': tickers_test,
    'predicted_return': preds
})

# Select top 5 stocks by predicted return
top = results.sort_values('predicted_return', ascending=False).groupby('ticker').head(1)

# Keep top 5 unique tickers
top_unique = top.drop_duplicates('ticker').nlargest(5, 'predicted_return')

print("Top predicted stocks:")
print(top_unique)

# Equal weight portfolio
if not top_unique.empty:
    equal_weight = 1.0 / len(top_unique)
    portfolio = top_unique.assign(weight=equal_weight)
    expected_portfolio_return = (portfolio['predicted_return'] * portfolio['weight']).sum()
    print("Expected portfolio return (equal weight):", expected_portfolio_return)
else:
    print("No stocks selected.")
