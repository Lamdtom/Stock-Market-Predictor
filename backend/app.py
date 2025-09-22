from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from typing import Optional
from predict import load_model, predict
from ingest import fetch

app = FastAPI(title="Stock Predictor API")

# Model config
CHECKPOINT_PATH = "model/model.pt"
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LOOKBACK = 10
model = load_model(CHECKPOINT_PATH, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

@app.post("/predict")
async def predict_stock(
    ticker: Optional[str] = Form(None),
    period: str = Form("5y"),
    interval: str = Form("1d"),
    file: Optional[UploadFile] = File(None)
):
    """
    Predict next return from either:
    - Uploaded CSV file (multipart)
    - Stock ticker (multipart/form-data)
    """
    # Determine data source
    if file is not None:
        if not file.filename.endswith(".csv"):
            return {"error": "Only CSV files are supported"}
        df = pd.read_csv(file.file, parse_dates=['Date'])
    elif ticker:
        df = fetch(ticker, period=period, interval=interval)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        return {"error": "You must provide either a ticker or a CSV file"}

    pred = predict(model, df, lookback=LOOKBACK)
    
    if pred is None:
        return {"error": "Not enough data to make prediction"}
    
    if hasattr(pred, "item"):
        pred = pred.item()
    
    response = {"prediction": pred}
    if ticker:
        response["ticker"] = ticker
    if file:
        response["file_name"] = file.filename
    
    return response
