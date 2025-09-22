import torch
from model import LSTMModel
from dataset import StockPreprocessor
import pandas as pd

def load_model(checkpoint_path, input_size, hidden_size=64, num_layers=1):
    model = LSTMModel(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

def predict(model, data: pd.DataFrame, lookback=10):
    pre = StockPreprocessor(lookback)
    df = pre.add_features(data)

    X, _ = pre.make_supervised(df)

    if len(X) == 0:
        return None 
    
    # Take only the last sequence for next-day prediction
    last_seq = X[-1].unsqueeze(0)  # shape: (1, seq_len, input_size)

    with torch.no_grad():
        pred_return = model(last_seq).numpy().flatten()[0]
    
    # Compute predicted price
    last_close = df['Close'].iloc[-1]
    predicted_price = last_close * (1 + pred_return)

    # print(f"Last close price: {last_close}")
    # print(f"Predicted next-day return: {pred_return:.4f}")
    # print(f"Predicted next-day price: {predicted_price:.2f}")

    return predicted_price