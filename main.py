#python main.py

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)
app.secret_key = 'supersecretkey'

# Load dataset
DATA_PATH = os.path.join(os.getcwd(), r'C:\Users\sriya\Downloads\new_weather\weather\weather_data.csv')
chunk_size = 10000
df_list = [chunk for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size)]
df = pd.concat(df_list, ignore_index=True)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={
    'rain (mm)': 'rain_mm',
    'min temp (°c)': 'min_temp_c',
    'max temp (°c)': 'max_temp_c',
    'min humidity (%)': 'min_humidity_percent',
    'max humidity (%)': 'max_humidity_percent',
    'min wind speed (kmph)': 'min_wind_speed_kmph',
    'max wind speed (kmph)': 'max_wind_speed_kmph'
}, inplace=True)

# Fill missing values
numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Feature Engineering
df['date_num'] = df['date'].map(datetime.toordinal)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Prepare features and target variables
features = ['date_num', 'year', 'month', 'day', 'min_temp_c', 'max_temp_c', 'min_humidity_percent', 'max_humidity_percent', 'min_wind_speed_kmph', 'max_wind_speed_kmph']
target_columns = ['rain_mm', 'min_temp_c', 'max_temp_c', 'min_humidity_percent', 'max_humidity_percent', 'min_wind_speed_kmph', 'max_wind_speed_kmph']

X = df[features]
y = df[target_columns]

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)
joblib.dump(model, 'weather_model_xgb.pkl')

@app.route('/')
def home():
    districts = df['district'].unique()
    return render_template('index.html', districts=districts)

@app.route('/get_districts', methods=['GET'])
def get_districts():
    return jsonify(df['district'].dropna().unique().tolist())

@app.route('/get_mandals', methods=['POST'])
def get_mandals():
    data = request.get_json()
    district = data['district']
    mandals = df[df['district'] == district]['mandal'].dropna().unique().tolist()
    return jsonify(mandals)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    district = data.get('district')
    mandal = data.get('mandal')
    date = data.get('date')

    if not district or not mandal or not date:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        target_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    existing_data = df.loc[
        (df['district'] == district) &
        (df['mandal'] == mandal) &
        (df['date'].dt.strftime('%Y-%m-%d') == target_date)
    ]

    if not existing_data.empty:
        row = existing_data.iloc[0]
        result = {
            'rainfall': float(round(row['rain_mm'], 2)),
            'temp_min': float(round(row['min_temp_c'], 2)),
            'temp_max': float(round(row['max_temp_c'], 2)),
            'humidity_min': float(round(row['min_humidity_percent'], 2)),
            'humidity_max': float(round(row['max_humidity_percent'], 2)),
            'wind_speed_min': float(round(row['min_wind_speed_kmph'], 2)),
            'wind_speed_max': float(round(row['max_wind_speed_kmph'], 2))
        }
        return jsonify(result)

    filtered_data = df[(df['district'] == district) & (df['mandal'] == mandal)].copy()
    if filtered_data.empty or filtered_data.shape[0] < 2:
        return jsonify({"error": "Not enough data to train model"})

    filtered_data['date_num'] = filtered_data['date'].map(datetime.toordinal)
    target_ordinal = datetime.strptime(date, '%Y-%m-%d').toordinal()
    target_df = pd.DataFrame([[target_ordinal, target_ordinal, target_ordinal, target_ordinal, target_ordinal, target_ordinal, target_ordinal, target_ordinal, target_ordinal, target_ordinal]],
                             columns=features)

    def train_and_predict(column):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(filtered_data[features], filtered_data[column])
        return float(round(model.predict(target_df)[0], 2))

    result = {
        'rainfall': train_and_predict('rain_mm'),
        'temp_min': train_and_predict('min_temp_c'),
        'temp_max': train_and_predict('max_temp_c'),
        'humidity_min': train_and_predict('min_humidity_percent'),
        'humidity_max': train_and_predict('max_humidity_percent'),
        'wind_speed_min': train_and_predict('min_wind_speed_kmph'),
        'wind_speed_max': train_and_predict('max_wind_speed_kmph')
    }

    return jsonify(result)

@app.route('/visualization')
def visualization_page():
    return render_template('visualization.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.get_json()
    
    # Convert single-value fields to lists if needed
    visualization_data = {
        "dates": ["Day 1", "Day 2", "Day 3"],
        "rainfall": [data.get("rainfall")] * 3,
        "temp_min": [data.get("temp_min")] * 3,
        "temp_max": [data.get("temp_max")] * 3,
        "humidity_min": [data.get("humidity_min")] * 3,
        "humidity_max": [data.get("humidity_max")] * 3,
        "wind_speed_min": [data.get("wind_speed_min")] * 3,
        "wind_speed_max": [data.get("wind_speed_max")] * 3
    }

    return jsonify(visualization_data)


if __name__ == '__main__':
    app.run(debug=True)
