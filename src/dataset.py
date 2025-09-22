import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class StockPreprocessor:
    def __init__(self, lookback=10):
        self.lookback = lookback
    
    def add_features(self, df: pd.DataFrame):
        # Only compute features if not already in df
        required_cols = ['return','ma_5','ma_20','vol_10']
        if not all(col in df.columns for col in required_cols):
            # Make sure Close is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close']).reset_index(drop=True)
            
            df['return'] = df['Close'].pct_change()
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['vol_10'] = df['return'].rolling(10).std()
            df = df.dropna().reset_index(drop=True)
        return df
    
    def make_supervised(self, df: pd.DataFrame):
        features = df[['return','ma_5','ma_20','vol_10']].values

        X, y = [], []
        for i in range(self.lookback, len(features)):
            X.append(features[i-self.lookback:i])
            y.append(df['return'].iloc[i])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        return X, y

class StockDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        if self.transform:
            x = self.transform(x)
        return x, y

def load_datasets(csv_path, lookback=10, splits=(0.7,0.15,0.15), preprocessed=False):
    """
    Load datasets for training, validation, testing.
    
    Args:
        csv_path: path to CSV (raw or preprocessed)
        lookback: sliding window size
        splits: train/val/test ratio
        preprocessed: True if CSV already has 'return','ma_5','ma_20','vol_10'
        
    Returns:
        train_ds, val_ds, test_ds (StockDataset)
    """

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    pre = StockPreprocessor(lookback)
    if not preprocessed:
        df = pre.add_features(df)
    X, y = pre.make_supervised(df)

    n = len(X)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return (
        StockDataset(X_train, y_train),
        StockDataset(X_val, y_val),
        StockDataset(X_test, y_test)
    )

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "AAPL_raw.csv")
    train_ds, val_ds, test_ds = load_datasets(data_dir, lookback=10, preprocessed=False)
    print(train_ds)