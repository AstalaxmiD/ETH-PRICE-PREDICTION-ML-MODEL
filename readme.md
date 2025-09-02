# Ethereum Price Prediction Project

## Project Overview
This project aims to predict the daily high price of Ethereum (ETH) using machine learning techniques. We combine historical cryptocurrency data, market sentiment, and on-chain metrics to improve the accuracy of predictions. 

## Data Used
The model uses a combination of the following data:
- **Ethereum and Bitcoin prices:** high, low, and close prices
- **Fear and Greed Index:** to capture market sentiment


## Methodology
1. Prepare historical data and create lagged features to incorporate the previous day's market activity.
2. Train a Random Forest Regressor to predict the next day's ETH high price.
3. Evaluate the model using metrics like Mean Squared Error (MSE) and R² score.
4. Analyze feature importance to identify which metrics most influence Ethereum’s price movements.

## Objective
The goal is not only to predict ETH highs but also to understand **which factors have the largest impact** on its price. This insight can help traders and analysts make more informed decisions.

## Output:
Metrics in order of importance:
['eth_high_lag1', 'eth_high_lag2', 'eth_high_lag3', 'btc_close_lag3', 'btc_high_lag3', 'btc_low_lag3', 'btc_low_lag1', 'btc_high_lag1', 'btc_low_lag2', 'btc_close_lag1', 'btc_close_lag2', 'fear_greed_index', 'btc_high_lag2']

Predicted ETH high: [4417.4812 4405.4712 4392.505  4591.651  3775.3902 4556.5458]
Mean Squared Error: 32321.99843175987
R2 Score: 0.813199806850879

## Possible biases:
- Feature Bias: Using only a few features (ETH/BTC highs, lows, closes, Fear & Greed index) ignores other important market influences: News events, Macroeconomic data and Exchange-specific activity.

- Stationarity Bias: Cryptocurrency prices are highly non-stationary.Models assuming stable patterns (like linear regression) may fail during trends or crashes.

- Sample Size Bias

## Improvements to minimize biases:
- News Sentiment Score: Use an NLP model to assign a daily sentiment score to crypto news articles. The score can be used by the model to account for major events.

- Add more metrics: eth_inflow, eth_outflow, eth_net_flow, whale_transfer_count, whale_net_flow, active_addresses, new_addresses etc.

- Give the model more data: Could be more than a year of ethereum data.

