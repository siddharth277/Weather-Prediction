# 🌦️ Weather Temperature Forecasting using CNN-LSTM


---

## 🚀 Overview

This project implements a **deep learning-based time series forecasting system** to predict **mean temperature** using historical weather data.

It leverages a **hybrid CNN + LSTM architecture** to capture:
- 📌 Spatial patterns (via CNN)
- 📌 Temporal dependencies (via LSTM)

The system is designed to be **robust, scalable, and deployable**, with an integrated **Flask API for real-time predictions**.

---

## 🧠 Methodology

### 🔹 Phase 1 (Exploration & Feature Engineering)

Work done in `phase1.ipynb`:

- Data cleaning & preprocessing
- Handling missing values
- Feature engineering:
  - Lag features (1, 2, 3, 7 days)
  - Rolling mean & standard deviation (7 & 14 days)
- Seasonality encoding:
  - Sine and Cosine transformation of time
- Feature scaling using StandardScaler
- Visualization of trends and correlations
- Dataset splitting (train/test)

👉 Goal: Transform raw time series into **model-ready structured data**

---

### 🔹 Phase 2 (Model Building)

- CNN layers extract local patterns
- LSTM layers learn sequential dependencies
- Dense layers produce final output

👉 Hybrid approach improves performance over:
- Traditional AR models
- Standalone LSTM networks

---

## 📂 Project Structure

```
├── app.py
├── train_city.py
├── phase1.ipynb
├── best_cnn_lstm.keras
├── x_scaler.pkl
├── y_scaler.pkl
├── feature_cols.pkl
├── train_featured.csv
├── test_featured.csv
├── requirements.txt
└── cnn_lstm_results.png
```

---

## 📊 Features Used

- Lag Features → lag_1, lag_2, lag_3, lag_7  
- Rolling Stats → mean & std (7, 14 days)  
- Seasonal Encoding → sin_day, cos_day  
- Derived Metrics:
  - Temperature-Humidity Index  
  - Wind-Pressure Interaction  
  - Weekly Change  

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python app.py
```

Open: http://127.0.0.1:5000

---

## 📥 Input Format

Provide last 14 days:

```json
[
  {
    "meantemp": 25,
    "humidity": 60,
    "wind_speed": 5,
    "meanpressure": 1010
  }
]
```

---

## 📈 Evaluation Metrics

- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

👉 Used to evaluate prediction accuracy and generalization.

---

## 🌟 Novelty & Contributions

- 🔥 Hybrid CNN-LSTM architecture (captures both spatial + temporal patterns)
- 🧠 Strong feature engineering pipeline
- 🌍 Seasonality encoding using trigonometric functions
- 📊 Stabilized training via scaling & transformations
- ⚡ End-to-end pipeline (training → inference → deployment)
- 🧪 Experiment-driven improvements from Phase 1

---

## 📌 Future Scope

- Transformer-based time series models
- Multi-city weather forecasting
- Real-time API deployment (cloud)
- Interactive dashboard (Streamlit/React)

---

## 👨‍💻 Author

Siddharth Shukla  
AI/ML | Deep Learning | Time Series

---

## 📜 License

Academic / Research Use Only
