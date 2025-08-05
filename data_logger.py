import csv
from data_acquisition import RealOBDSensorData
import numpy as np
import joblib
from tensorflow import keras
from collections import deque

# --- Autoencoder integration ---
MODEL_FILE = 'autoencoder_model.h5'
SCALER_FILE = 'autoencoder_scaler.save'
WINDOW_SIZE = 20

# Load model and scaler
try:
    autoencoder = keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("[INFO] Autoencoder and scaler loaded.")
    # Load training loss distribution to set threshold (optional, fallback to fixed value)
    try:
        train_losses = np.load('autoencoder_train_losses.npy')
        ANOMALY_THRESHOLD = float(np.percentile(train_losses, 99.5))
        print(f"[INFO] Anomaly threshold set to {ANOMALY_THRESHOLD:.6f} (99.5th percentile of training loss)")
    except Exception:
        ANOMALY_THRESHOLD = 0.01  # fallback, should adjust after first training
        print(f"[WARN] Using fallback anomaly threshold: {ANOMALY_THRESHOLD}")
except Exception as e:
    autoencoder = None
    scaler = None
    ANOMALY_THRESHOLD = None
    print(f"[WARN] Autoencoder/scaler not loaded: {e}")

# --- End Autoencoder integration ---

def check_thresholds(reading):
    alerts = []
    # RPM
    if reading['rpm'] is not None and (reading['rpm'] > 5500 or reading['rpm'] < 600):
        alerts.append(f"RPM out of range: {reading['rpm']}")
    # Coolant Temp
    if reading['coolant_temp'] is not None and (reading['coolant_temp'] > 105 or reading['coolant_temp'] < 70):
        alerts.append(f"Coolant Temp out of range: {reading['coolant_temp']}°C")
    # Intake Pressure
    if reading['intake_pressure'] is not None and reading['intake_pressure'] > 120:
        alerts.append(f"Intake Pressure high: {reading['intake_pressure']} kPa")
    # MAF
    if reading['maf'] is not None and reading['maf'] > 120:
        alerts.append(f"MAF unusually high: {reading['maf']} g/s")
    # Throttle Pos
    if reading['throttle_pos'] is not None and (reading['throttle_pos'] > 95 or reading['throttle_pos'] < 2):
        alerts.append(f"Throttle Position out of range: {reading['throttle_pos']}%")
    # Engine Load
    if reading['engine_load'] is not None and (reading['engine_load'] > 90 or reading['engine_load'] < 10):
        alerts.append(f"Engine Load out of range: {reading['engine_load']}%")
    # Vehicle Speed
    if reading['vehicle_speed'] is not None and reading['vehicle_speed'] > 180:
        alerts.append(f"Vehicle Speed unusually high: {reading['vehicle_speed']} km/h")
    # Intake Air Temp
    if reading['intake_air_temp'] is not None and reading['intake_air_temp'] > 60:
        alerts.append(f"Intake Air Temp high: {reading['intake_air_temp']}°C")
    # Voltage
    if reading['voltage'] is not None and (reading['voltage'] < 11.5 or reading['voltage'] > 15):
        alerts.append(f"Voltage out of range: {reading['voltage']}V")
    return alerts

def log_data(port_str, interval=1.0, csv_file='vehicle_data_log.csv', max_rows=1000):
    sensor = RealOBDSensorData(port_str=port_str, interval=interval)
    fieldnames = [
        'timestamp', 'rpm', 'coolant_temp', 'intake_pressure', 'maf',
        'throttle_pos', 'engine_load', 'vehicle_speed', 'intake_air_temp', 'voltage'
    ]
    window = deque(maxlen=WINDOW_SIZE)
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, reading in enumerate(sensor.generate()):
            writer.writerow(reading)
            alerts = check_thresholds(reading)
            print(f"Logged row {i+1}: {reading}")
            if alerts:
                print(f"[ALERT] {', '.join(alerts)}")
            # --- Autoencoder anomaly detection ---
            if autoencoder is not None and scaler is not None:
                # Prepare input for autoencoder
                features = [reading['rpm'], reading['coolant_temp'], reading['intake_pressure'], reading['maf'],
                            reading['throttle_pos'], reading['engine_load'], reading['vehicle_speed'],
                            reading['intake_air_temp'], reading['voltage']]
                window.append(features)
                if len(window) == WINDOW_SIZE:
                    window_np = np.array(window).reshape(1, WINDOW_SIZE, 9)
                    window_scaled = scaler.transform(window_np[0])
                    window_scaled = window_scaled.reshape(1, WINDOW_SIZE, 9)
                    recon = autoencoder.predict(window_scaled, verbose=0)
                    mse = np.mean(np.square(window_scaled - recon))
                    if mse > ANOMALY_THRESHOLD:
                        print(f"[ANOMALY ALERT] Autoencoder anomaly score: {mse:.6f} (threshold: {ANOMALY_THRESHOLD:.6f})")
            # --- End Autoencoder anomaly detection ---
            if i+1 >= max_rows:
                break

if __name__ == '__main__':
    # Replace 'COM10' with your actual port if needed
    log_data(port_str='COM10', interval=1.0, csv_file='vehicle_data_log.csv', max_rows=1000)
