# Vehicle Health Monitoring & Predictive Maintenance using AI

A real-time intelligent system to predict vehicle component failures (engine, brakes, battery, etc.) before they occur, using onboard sensor data and advanced AI models. Ideal for portfolio demonstration and technical showcasing.

## Features
- Real-time sensor data aggregation (OBD-II, ECU, IoT, mock simulation)
- LSTM neural network for time-series forecasting
- Autoencoder for anomaly detection
- XGBoost for failure classification
- Ensemble model for robust prediction
- Streamlit dashboard for live monitoring and alerts
- Modular, production-ready Python codebase

## Project Structure
```
vehicle_health_ai/
├── data_acquisition.py
├── preprocessing.py
├── models/
│   ├── lstm_model.py
│   ├── autoencoder.py
│   ├── xgboost_classifier.py
│   └── ensemble.py
├── dashboard/
│   └── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Test the data pipeline:
   ```bash
   python main.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## Customization
- Integrate real sensor data by replacing `MockSensorData` in `data_acquisition.py`.
- Add/modify models in the `models/` directory.

## Requirements
See `requirements.txt` for all dependencies.

---
Developed for AI, ML, and automotive portfolio projects.
