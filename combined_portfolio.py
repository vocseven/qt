import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import cvxpy as cp

base_dir = 'datasets_2025June6'
path_daily = f"{base_dir}/merged_daily.csv"
path_weekly = f"{base_dir}/weekly_merged.csv"
path_30 = f"{base_dir}/merged_30min.csv"

print("Loading datasets...")
df_daily = pd.read_csv(path_daily)
df_weekly = pd.read_csv(path_weekly)
df_30 = pd.read_csv(path_30)

# Clean and add frequency labels
print("Cleaning and combining datasets...")
def clean(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    return df

df_daily = clean(df_daily)
df_weekly = clean(df_weekly)
df_30 = clean(df_30)

df_daily['freq'] = '1d'
df_weekly['freq'] = '1w'
df_30['freq'] = '30m'

# Combine vertically
combined = pd.concat([df_daily, df_weekly, df_30], ignore_index=True)
combined['date'] = pd.to_datetime(combined['time'], unit='s')
combined = combined.sort_values(['ticker', 'date'])

# Target: next period return
combined['next_close'] = combined.groupby('ticker')['close'].shift(-1)
combined['target_ret'] = combined['next_close'] / combined['close'] - 1
combined.dropna(subset=['target_ret'], inplace=True)

# Feature columns
feature_cols = [c for c in combined.columns if c not in ['time','date','next_close','target_ret','ticker','freq']]
X = combined[feature_cols]
y = combined['target_ret']

X_train, X_test, y_train, y_test, tick_train, tick_test = train_test_split(
    X, y, combined['ticker'], test_size=0.2, random_state=42, shuffle=True
)

print("Training model on combined dataset...")
model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(((y_test - preds) ** 2).mean())
print(f"Test RMSE: {rmse:.6f}")

results = pd.DataFrame({'ticker': tick_test, 'pred_return': preds})
# Average prediction per ticker and select top 5
mean_preds = results.groupby('ticker')['pred_return'].mean().reset_index()
top = mean_preds.sort_values('pred_return', ascending=False).head(5)
print("Top predicted tickers:\n", top)

# Portfolio optimization
selected_tickers = top['ticker'].tolist()
exp_returns = top['pred_return'].values

# Use training data returns for covariance
train_df = combined.loc[X_train.index]
ret_mat = train_df[train_df['ticker'].isin(selected_tickers)].pivot_table(
    index='date', columns='ticker', values='target_ret'
).dropna()

# If not enough data, fall back to identity covariance
if ret_mat.shape[0] > 1:
    cov = ret_mat.cov().values
else:
    cov = np.eye(len(selected_tickers)) * 1e-4

weights = cp.Variable(len(selected_tickers))
risk_aversion = 1.0
objective = cp.Maximize(exp_returns @ weights - risk_aversion * cp.quad_form(weights, cov))
constraints = [cp.sum(weights) == 1, weights >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()

allocations = weights.value

opt_port = pd.DataFrame({'ticker': selected_tickers, 'allocation': allocations})
print("\nOptimized portfolio allocations:\n", opt_port)
