import os
import requests
import datetime
import numpy as np
import pandas as pd
import pickle
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout

WINDOW = 14

def get_city_coordinates(city_name):
    print(f"[*] Looking up coordinates for '{city_name}'...")
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if 'results' not in data or len(data['results']) == 0:
        raise ValueError(f"City '{city_name}' not found!")
    lat = data['results'][0]['latitude']
    lon = data['results'][0]['longitude']
    name = data['results'][0].get('name', city_name)
    print(f"[+] Found {name} at Lat: {lat}, Lon: {lon}")
    return lat, lon

def fetch_historical_weather(lat, lon, years=10):
    print(f"[*] Downloading {years} years of historical weather data. This may take 5-10 seconds...")
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    # Factor leap years by using 365.25
    start_date = end_date - datetime.timedelta(days=int(365.25 * years))
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&timezone=auto"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    df_hourly = pd.DataFrame(data['hourly'])
    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    daily = df_hourly.groupby(df_hourly['time'].dt.date).mean(numeric_only=True).reset_index()
    daily.rename(columns={'time': 'date'}, inplace=True)
    
    daily = daily.rename(columns={
        "temperature_2m": "meantemp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "surface_pressure": "meanpressure"
    })
    print(f"[+] Downloaded {len(daily)} days of data.")
    return daily

def engineer_features(df):
    print("[*] Engineering features matching the prediction pipeline...")
    df = df.copy()
    
    df["wind_speed"] = np.log1p(df["wind_speed"])
    for col in ["meanpressure"]:
        df[col] = df[col].clip(lower=970, upper=1025)

    for col in ["meantemp", "humidity", "wind_speed", "meanpressure"]:
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for window in [7, 14]:
        rolled = df["meantemp"].shift(1).rolling(window=window)
        df[f"temp_roll_mean_{window}"] = rolled.mean()
        df[f"temp_roll_std_{window}"] = np.log1p(rolled.std())

    # Temporal features (day of year)
    day_of_year = pd.to_datetime(df['date']).dt.dayofyear
    df["sin_day"] = np.sin(2 * math.pi * day_of_year / 365.25)
    df["cos_day"] = np.cos(2 * math.pi * day_of_year / 365.25)

    df["temp_humidity_index"] = df["meantemp"].shift(1) * df["humidity"] / 100
    df["temp_change_7d"] = df["meantemp"] - df["meantemp"].shift(7)
    df["wind_pressure_index"] = df["wind_speed"].shift(1) * df["meanpressure"].shift(1) / 1000

    df = df.dropna().reset_index(drop=True)
    return df

def train_model():
    print("="*60)
    print("   AI Weather Model Retrainer")
    print("="*60)
    city = input("\n>>> Enter city name to train the model for (e.g. Dharwad, London): ").strip()
    if not city:
        print("City name cannot be empty.")
        return

    try:
        lat, lon = get_city_coordinates(city)
        df = fetch_historical_weather(lat, lon, years=10)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    df_engineered = engineer_features(df)
    
    # The sequence window creates an implicit +1 day shift because we grab i+WINDOW
    target_data = df_engineered['meantemp'].values.reshape(-1, 1)
    
    feature_cols = [c for c in df_engineered.columns if c not in ['date', 'time', 'index']]
    feature_data = df_engineered[feature_cols].values
    
    print(f"[*] Scaling data ({len(feature_cols)} features)...")
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_scaled = x_scaler.fit_transform(feature_data)
    y_scaled = y_scaler.fit_transform(target_data)
    
    # Create sequences
    print(f"[*] Creating {WINDOW}-day seqs for time-series learning...")
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - WINDOW):
        X_seq.append(X_scaled[i : i + WINDOW])
        y_seq.append(y_scaled[i + WINDOW])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Split Train/Test (90/10)
    split = int(len(X_seq) * 0.9)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    print(f"[+] Sequence pairs -> Train: {len(X_train)}, Val: {len(X_test)}")
    
    # Define Model
    print("[*] Constructing CNN-LSTM neural architecture...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW, len(feature_cols))),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("[*] Starting training (Model may finish early if optimal)...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        epochs=30, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n[+] Training complete! Saving brain and dependencies...")
    model.save("best_cnn_lstm.keras")
    with open("x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open("y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open("feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    print(f"\n[SUCCESS] Your weather prediction app is now globally optimized for `{city}`!")
    print("Action Required: Restart `app.py` in your terminal so it catches the new AI files.")

if __name__ == "__main__":
    train_model()
