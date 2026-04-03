# 🌤️ AI Weather Temperature Predictor

A deep learning web application that predicts the **next day's mean temperature** for any city in the world using a **CNN-LSTM neural network** trained on real historical weather data.

---

## 🧠 How It Works

1. You enter a city name in the web interface.
2. The app fetches the **last 14 days of real weather data** from the [Open-Meteo API](https://open-meteo.com/) (free, no key needed).
3. The data is feature-engineered and passed into a pre-trained **CNN-LSTM model**.
4. The model predicts **tomorrow's mean temperature** in °C.

You can also **retrain the model** for any city using 10 years of historical weather data — right from your terminal.

---

## 📁 Project Structure

```
ai_project/
│
├── app.py                  # Flask web server + prediction API
├── train_city.py           # Retrain the model for any city
│
├── best_cnn_lstm.keras     # Pre-trained CNN-LSTM model weights
├── x_scaler.pkl            # Feature scaler (MinMaxScaler)
├── y_scaler.pkl            # Target scaler (MinMaxScaler)
├── feature_cols.pkl        # Feature column names used during training
│
├── templates/
│   └── index.html          # Frontend UI
│
├── Dataset/
│   ├── Train.csv           # Original training dataset (Delhi weather)
│   └── Test.csv            # Original test dataset
│
├── phase1.ipynb            # Exploratory data analysis & model experiments
└── cnn_lstm_results.png    # Training results visualization
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install manually:

```bash
pip install flask tensorflow scikit-learn numpy pandas requests
```

### 3. Run the Web App

```bash
python app.py
```

Then open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## 🔁 Retrain for Any City

The pre-trained model is optimized for **Delhi, India**. To retrain it for your city:

```bash
python train_city.py
```

You will be prompted to enter a city name (e.g., `London`, `Tokyo`, `Chennai`). The script will:
- Fetch 10 years of historical weather data via Open-Meteo Archive API
- Engineer time-series features
- Train a new CNN-LSTM model
- Save the new model files (`best_cnn_lstm.keras`, scalers, feature list)

After retraining, **restart `app.py`** so it loads the updated model.

---

## 🏗️ Model Architecture

```
Input (14 days × N features)
    ↓
Conv1D (64 filters, kernel=3, ReLU)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.2)
    ↓
LSTM (64 units, return_sequences=True)
    ↓
LSTM (32 units)
    ↓
Dropout (0.2)
    ↓
Dense (32, ReLU)
    ↓
Dense (1) → Predicted Temperature (°C)
```

**Training Details:**
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Early stopping: patience = 6 epochs
- Train/Test split: 90% / 10%
- Input window: 14 days

---

## ⚙️ Features Engineered

| Feature | Description |
|---|---|
| `meantemp`, `humidity`, `wind_speed`, `meanpressure` | Raw daily weather readings |
| `*_lag_1/2/3/7` | Lagged values for each raw feature |
| `temp_roll_mean_7/14` | Rolling mean of temperature |
| `temp_roll_std_7/14` | Rolling std dev (log-scaled) |
| `sin_day`, `cos_day` | Cyclical day-of-year encoding |
| `temp_humidity_index` | Temperature × Humidity / 100 |
| `temp_change_7d` | 7-day temperature delta |
| `wind_pressure_index` | Wind speed × Pressure / 1000 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/fetch-weather?city=<name>` | Fetches last 14 days of real weather for any city |
| `POST` | `/predict` | Accepts 14-day weather JSON, returns temperature prediction |

### Example `/predict` Request

```json
POST /predict
Content-Type: application/json

[
  { "meantemp": 28.5, "humidity": 65.2, "wind_speed": 12.1, "meanpressure": 1005.3 },
  ...  (14 entries total)
]
```

### Example Response

```json
{
  "prediction": 31.47,
  "status": "ok"
}
```

---

## 📊 Dataset

The original dataset is Delhi Climate Data (2013–2017) from Kaggle, containing daily records of:
- Mean Temperature (°C)
- Humidity (%)
- Wind Speed (km/h)
- Mean Pressure (hPa)

Live prediction data is sourced in real-time from the **Open-Meteo API** — no API key required.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ML/DL:** TensorFlow / Keras, Scikit-learn
- **Data:** Pandas, NumPy
- **Weather Data:** [Open-Meteo API](https://open-meteo.com/)
- **Frontend:** HTML, CSS, JavaScript (in `templates/index.html`)

---

## 📌 Notes

- The `.keras` model files and `.pkl` scaler files are large binary files. Consider adding them to `.gitignore` and hosting them separately (e.g., Google Drive, HuggingFace) for production use.
- The app runs in Flask's built-in development server. For production, use **Gunicorn** or **uWSGI** behind **Nginx**.

---

## 📄 License

This project is for educational purposes. Feel free to fork, modify, and build upon it.
