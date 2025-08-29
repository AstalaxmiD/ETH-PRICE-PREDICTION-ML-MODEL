Possible biases:
- Feature Bias: Using only a few features (ETH/BTC highs, lows, closes, Fear & Greed index) ignores other important market influences: News events, Macroeconomic data and Exchange-specific activity.

- Stationarity Bias: Cryptocurrency prices are highly non-stationary.Models assuming stable patterns (like linear regression) may fail during trends or crashes.

- Sample Size Bias

Improvements to minimize biases:
- News Sentiment Score: Use an NLP model to assign a daily sentiment score to crypto news articles. The score can be used by the model to account for major events.

- Add more metrics: eth_inflow, eth_outflow, eth_net_flow, whale_transfer_count, whale_net_flow, active_addresses, new_addresses etc.

- Give the model more data: Could be more than a year of ethereum data.

