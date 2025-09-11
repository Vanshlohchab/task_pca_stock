# Task 7: PCA + ARIMA Stock Forecasting
# Vansh Lohchab Internship Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ================================
# PART 1: Dimensionality Reduction (PCA on Iris Dataset)
# ================================

print("\n--- Part 1: PCA Visualization (Iris Dataset) ---")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio by PCA:", pca.explained_variance_ratio_)

# Plot PCA
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], alpha=0.7, c=colors[i], label=target_name)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.savefig("pca_iris.png")
plt.show()

# ================================
# PART 2: Stock Price Prediction using ARIMA
# ================================

print("\n--- Part 2: Stock Price Forecasting (ARIMA) ---")

# Load stock data (Apple stock as example)
ticker = "AAPL"
data = yf.download(ticker, start="2022-01-01", end="2023-01-01")

# Keep only Close prices
df = data[["Close"]].dropna()
df.index = pd.to_datetime(df.index)

# Train-test split (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit ARIMA model (p,d,q) = (5,1,0) as baseline
model = ARIMA(train["Close"], order=(5,1,0))
model_fit = model.fit()

# Forecast for test period
forecast = model_fit.forecast(steps=len(test))

# Evaluation metrics
mae = mean_absolute_error(test["Close"], forecast)
rmse = math.sqrt(mean_squared_error(test["Close"], forecast))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot actual vs forecast
plt.figure(figsize=(10,6))
plt.plot(train.index, train["Close"], label="Train")
plt.plot(test.index, test["Close"], label="Test", color="blue")
plt.plot(test.index, forecast, label="Forecast", color="red")
plt.title(f"ARIMA Stock Price Forecast - {ticker}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.savefig("arima_stock_forecast.png")
plt.show()
