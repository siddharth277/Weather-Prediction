from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ── Load model and scalers once at startup ────────────────────────────
model        = tf.keras.models.load_model("best_cnn_lstm.keras")
x_scaler     = pickle.load(open("x_scaler.pkl", "rb"))
y_scaler     = pickle.load(open("y_scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))

WINDOW = 14  # must match what you trained with

print("Model loaded. Starting server...")


def build_features(entries):
    """
    entries: list of 14 dicts, each with keys:
             meantemp, humidity, wind_speed, meanpressure
    Returns: scaled numpy array of shape (1, 14, n_features)
    """
    df = pd.DataFrame(entries)

    # Apply same transforms as training
    df["wind_speed"] = np.log1p(df["wind_speed"])
    for col in ["meanpressure"]:
        df[col] = df[col].clip(lower=970, upper=1025)

    # Build lag features (same logic as engineer_features)
    for col in ["meantemp", "humidity", "wind_speed", "meanpressure"]:
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for window in [7, 14]:
        rolled = df["meantemp"].shift(1).rolling(window=window, min_periods=1)
        df[f"temp_roll_mean_{window}"] = rolled.mean()
        df[f"temp_roll_std_{window}"]  = np.log1p(rolled.std())

    import math
    df["sin_day"] = df.get("sin_day", np.sin(2 * math.pi * pd.Timestamp.now().timetuple().tm_yday / 365.25))
    df["cos_day"] = df.get("cos_day", np.cos(2 * math.pi * pd.Timestamp.now().timetuple().tm_yday / 365.25))

    df["temp_humidity_index"] = df["meantemp"].shift(1) * df["humidity"] / 100
    df["temp_change_7d"]      = df["meantemp"] - df["meantemp"].shift(7)
    df["wind_pressure_index"] = df["wind_speed"].shift(1) * df["meanpressure"].shift(1) / 1000

    # Backfill missing values so we don't drop rows when sequence length is small
    df = df.bfill().fillna(0)
    df = df.reset_index(drop=True)

    if len(df) < WINDOW:
        raise ValueError(f"Not enough data after feature engineering. Got {len(df)} rows, need {WINDOW}.")

    # Use last WINDOW rows
    X = df[feature_cols].values[-WINDOW:]
    X_scaled = x_scaler.transform(X)
    return X_scaled.reshape(1, WINDOW, len(feature_cols))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/fetch-weather")
def fetch_weather():
    import requests
    
    city = request.args.get('city', 'Delhi')
    
    # 1. Resolve city using Open-Meteo Geocoding API
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    
    try:
        geo_res = requests.get(geo_url, timeout=10)
        geo_res.raise_for_status()
        geo_data = geo_res.json()
        
        if 'results' not in geo_data or len(geo_data['results']) == 0:
            return jsonify({"status": "error", "error": f"City '{city}' not found."})
            
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        resolved_name = geo_data['results'][0]['name']
        if 'country' in geo_data['results'][0]:
            resolved_name += f", {geo_data['results'][0]['country']}"
            
        # 2. Fetch real recent weather data for the resolved coordinates
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&past_days=15&forecast_days=1&timezone=auto"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Convert hourly to daily mean
        df_hourly = pd.DataFrame(data['hourly'])
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])
        daily = df_hourly.groupby(df_hourly['time'].dt.date).mean()
        
        daily = daily.rename(columns={
            "temperature_2m": "meantemp",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "surface_pressure": "meanpressure"
        })
        
        # Exclude today's incomplete data, get the last 14 completed days
        import datetime
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        daily_past = daily[daily.index <= yesterday].tail(14)
        
        entries = []
        for index, row in daily_past.iterrows():
            entries.append({
                "meantemp": round(row["meantemp"], 1),
                "humidity": round(row["humidity"], 1),
                "wind_speed": round(row["wind_speed"], 1),
                "meanpressure": round(row["meanpressure"], 1)
            })
        return jsonify({"status": "ok", "data": entries, "city": resolved_name})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # list of 14 day entries
        X_input = build_features(data)
        y_scaled = model.predict(X_input, verbose=0)
        y_celsius = y_scaler.inverse_transform(y_scaled).flatten()[0]
        return jsonify({"prediction": round(float(y_celsius), 2), "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


if __name__ == "__main__":
    app.run(debug=True)