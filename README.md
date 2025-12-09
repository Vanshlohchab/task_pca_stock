# Task : Data Analysis with Python – PCA & Stock Price Forecasting  

This project was completed as part of my internship tasks. It demonstrates **Dimensionality Reduction** using PCA and **Stock Price Prediction** using ARIMA.  

---

## 📌 Project Overview  

### 1. Dimensionality Reduction – PCA  
- Dataset: **Iris dataset**  
- Technique: **Principal Component Analysis (PCA)**  
- Goal: Reduce data to 2 dimensions for visualization  
- Output: `pca_iris.png`  

### 2. Stock Price Forecasting – ARIMA  
- Dataset: Apple stock prices (fetched using **Yahoo Finance API**)  
- Technique: **Time Series Forecasting with ARIMA**  
- Evaluation Metrics: **MAE, RMSE**  
- Outputs:  
  - `arima_stock_forecast.png` → Comparison of actual vs forecasted prices  

---

📊 Outputs

- PCA Visualization: pca_iris.png
- Stock Forecasting: arima_stock_forecast.png

📖 Learnings

- Applied PCA for dimensionality reduction and visualization.
- Built an ARIMA model to forecast stock prices using time series data.
- Evaluated forecasting performance with error metrics (MAE, RMSE).

---

## 🚀 How to Run  

### 1. Clone this repository  
```bash
git clone https://github.com/Vanshlohchab/task7_pca_stock.git
cd task7_pca_stock

2. Install dependencies

pip install -r requirements.txt

3. Run the script

python task7_main.py
