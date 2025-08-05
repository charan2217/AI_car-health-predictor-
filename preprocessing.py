import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """Clean, normalize, and window vehicle sensor data."""
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.fitted = False

    def clean(self, df):
        """Handle missing values and outliers."""
        df = df.ffill().bfill()
        # Optionally, clip or remove outliers here
        return df

    def fit_transform(self, df):
        df = self.clean(df)
        features = df.drop(['timestamp'], axis=1)
        scaled = self.scaler.fit_transform(features)
        self.fitted = True
        return pd.DataFrame(scaled, columns=features.columns)

    def transform(self, df):
        df = self.clean(df)
        features = df.drop(['timestamp'], axis=1)
        if not self.fitted:
            raise RuntimeError('Call fit_transform first!')
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, columns=features.columns)

    def sliding_window(self, df):
        """Generate sliding window sequences for time-series models."""
        X = []
        for i in range(len(df) - self.window_size + 1):
            X.append(df.iloc[i:i+self.window_size].values)
        return np.array(X)

if __name__ == '__main__':
    # Example usage
    data = pd.DataFrame([
        {'timestamp': pd.Timestamp.now(), 'rpm': 1000, 'coolant_temp': 90, 'voltage': 13, 'fuel_trim': 0, 'brake_pad_wear': 10}
        for _ in range(30)
    ])
    prep = DataPreprocessor(window_size=5)
    scaled = prep.fit_transform(data)
    windows = prep.sliding_window(scaled)
    print(windows.shape)
