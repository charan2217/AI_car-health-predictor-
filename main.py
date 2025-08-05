from data_acquisition import RealOBDSensorData
from preprocessing import DataPreprocessor
# from models.lstm_model import LSTMModel
# from models.autoencoder import AutoencoderModel
# from models.xgboost_classifier import XGBoostClassifier
# from models.ensemble import EnsembleModel
import obd
import pandas as pd

if __name__ == '__main__':
    # --- Scan for supported OBD-II PIDs ---
    port = 'COM10'  # Replace with your actual COM port
    print(f"Scanning {port} for supported OBD-II PIDs...")
    connection = obd.OBD(port)
    supported = connection.supported_commands
    print("Supported OBD-II commands:")
    for cmd in supported:
        print(cmd)
    print("--- End of supported PID list ---\n")

    # --- Stream live OBD-II data ---
    sensor = RealOBDSensorData(port_str=port, interval=1.0)
    preprocessor = DataPreprocessor(window_size=20)
    data_buffer = []
    for i, reading in enumerate(sensor.generate()):
        data_buffer.append(reading)
        if len(data_buffer) >= 30:
            df = pd.DataFrame(data_buffer)
            scaled = preprocessor.fit_transform(df)
            windows = preprocessor.sliding_window(scaled)
            print(f"Windowed shape: {windows.shape}")
            break
    print("Data pipeline test complete.")
