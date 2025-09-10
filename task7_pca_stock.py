# Task 7 - Part 2: PCA on Stock Market Data
# ------------------------------------------

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Download stock data
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA"]  # 6 big tech companies
data = yf.download(tickers, start="2022-01-01", end="2023-01-01")["Close"]

print("Stock Prices Data:")
print(data.head())

# Step 2: Compute daily returns
returns = data.pct_change().dropna()
print("\nDaily Returns:")
print(returns.head())

# Step 3: Standardize returns
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# Step 4: Apply PCA
pca = PCA()
pca.fit(returns_scaled)

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio by Component:")
print(explained_variance)

# Step 5: Plot Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title("Explained Variance Ratio of Stock Returns (PCA)")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.grid(alpha=0.3)
plt.show()

# Step 6: Cumulative Variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1),
         explained_variance.cumsum(), marker='o', color='red')
plt.title("Cumulative Explained Variance (PCA on Stocks)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Ratio")
plt.grid(alpha=0.3)
plt.show()
