import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Parameters
CSV_FILE = 'vehicle_data_log.csv'
MODEL_FILE = 'autoencoder_model.h5'
SCALER_FILE = 'autoencoder_scaler.save'
WINDOW_SIZE = 20  # number of timesteps

# 1. Load and clean data
df = pd.read_csv(CSV_FILE)
# Drop rows with any missing values (or use .fillna if you prefer)
df = df.dropna()

# 2. Select features for training (use actual CSV header names from data_logger.py)
data_cols = [
    'rpm',
    'coolant_temp',
    'intake_pressure',
    'maf',
    'throttle_pos',
    'engine_load',
    'vehicle_speed',
    'intake_air_temp',
    'voltage'
]

# Clean and convert values to float
def clean_value(val):
    if isinstance(val, str):
        val = val.replace('%','').replace('"','').replace(',','.')
    try:
        return float(val)
    except:
        return np.nan

for col in data_cols:
    df[col] = df[col].apply(clean_value)

data = df[data_cols].values

# 3. Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
joblib.dump(scaler, SCALER_FILE)

# 4. Create sliding windows for time-series Autoencoder
X = []
for i in range(len(data_scaled) - WINDOW_SIZE + 1):
    X.append(data_scaled[i:i+WINDOW_SIZE])
X = np.array(X)
print(f"Windowed shape for training: {X.shape}")

# 5. Build Autoencoder model
def build_autoencoder(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(np.prod(input_shape), activation='linear'),
        layers.Reshape(input_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

autoencoder = build_autoencoder(X.shape[1:])
autoencoder.summary()

# 6. Train Autoencoder
history = autoencoder.fit(X, X, epochs=30, batch_size=32, validation_split=0.1)
autoencoder.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
print(f"Scaler saved to {SCALER_FILE}")
