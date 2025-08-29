import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'date': [
        '2025-08-29','2025-08-28','2025-08-27','2025-08-26','2025-08-25',
        '2025-08-24','2025-08-23','2025-08-22','2025-08-21','2025-08-20',
        '2025-08-19','2025-08-18','2025-08-17','2025-08-16','2025-08-15',
        '2025-08-14','2025-08-13','2025-08-12','2025-08-11','2025-08-10',
        '2025-08-09','2025-08-08','2025-08-07','2025-08-06','2025-08-05',
        '2025-08-04','2025-08-03','2025-08-02','2025-08-01','2025-07-31'
    ],
    'eth_high': [
        4693.90,4507.56,4500.15,4602.37,4381.63,
        4778.11,4773.88,4829.23,4224.44,4330.49,
        4074.50,4317.28,4487.12,4426.83,4430.53,
        4554.29,4763.65,4606.81,4228.82,4253.59,
        4265.56,4012.98,3911.26,3684.05,3612.44,
        3715.71,3497.57,3397.49,3483.18,3696.66
    ],
    'eth_low': [
        4500.00,4400.00,4400.00,4500.00,4300.00,
        4600.00,4700.00,4700.00,4100.00,4200.00,
        3900.00,4200.00,4400.00,4300.00,4300.00,
        4400.00,4600.00,4500.00,4100.00,4100.00,
        4200.00,3900.00,3800.00,3600.00,3500.00,
        3600.00,3400.00,3300.00,3400.00,3600.00
    ],
    'eth_close': [
        4600.00,4450.00,4450.00,4550.00,4400.00,
        4680.00,4750.00,4780.00,4150.00,4250.00,
        4020.00,4300.00,4450.00,4370.00,4410.00,
        4520.00,4730.00,4580.00,4200.00,4230.00,
        4250.00,3980.00,3880.00,3650.00,3550.00,
        3700.00,3450.00,3350.00,3450.00,3680.00
    ],
    'btc_high': [
        116000.00,112617.30,113464.60,112611.30,112377.00,
        113665.10,115663.30,117024.30,117416.90,114776.30,
        114616.70,116729.20,117552.70,118559.40,117914.90,
        119231.90,124436.80,123651.80,120317.20,122319.50,
        119305.30,117893.00,117625.10,117628.40,115714.00,
        115111.70,115720.90,114758.40,114050.00,116054.60
    ],
    'btc_low': [
        111000.00,111434.20,110887.70,110409.90,108768.30,
        109295.10,110815.90,114576.50,111698.50,112023.00,
        112409.40,112750.90,114653.80,117192.40,117195.60,
        116816.80,117275.10,118959.90,118217.40,118242.50,
        116468.40,116357.70,115917.80,114267.30,113361.20,
        112678.90,114116.10,111992.40,112015.50,112760.60
    ],
    'btc_close': [
        112500.00,112000.00,112800.00,112000.00,111500.00,
        113000.00,115000.00,116500.00,114500.00,113500.00,
        114000.00,116000.00,117000.00,118000.00,117500.00,
        119000.00,124000.00,123000.00,120000.00,122000.00,
        119000.00,117500.00,117000.00,117000.00,115500.00,
        115000.00,115500.00,114500.00,114000.00,116000.00
    ],
    'fear_greed_index': [
        50,48,51,48,47,
        53,60,50,50,44,
        56,60,64,56,60,
        75,73,68,70,69,
        67,74,62,54,60,
        64,53,55,65,72
    ]
}


df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Create lagged features (1, 2, 3 days)
lags = [1, 2, 3]
for lag in lags:
    df[f'btc_high_lag{lag}'] = df['btc_high'].shift(lag)
    df[f'btc_low_lag{lag}'] = df['btc_low'].shift(lag)
    df[f'btc_close_lag{lag}'] = df['btc_close'].shift(lag)
    df[f'eth_high_lag{lag}'] = df['eth_high'].shift(lag)

# Drop rows with NaNs
df = df.dropna()

# Features: lagged BTC + lagged ETH + fear_greed_index
feature_cols = [col for col in df.columns if 'lag' in col] + ['fear_greed_index']
X = df[feature_cols]
y = df['eth_high']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

importances = model.feature_importances_
feature_names = X.columns
sorted_indices = importances.argsort()[::-1]
sorted_features = feature_names[sorted_indices]
print("Metrics in order of importance:")
print(list(sorted_features))


print("Predicted ETH high:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
