import yfinance as yf
import pandas as pd
import datetime as datetime
import os

def fetch(symbol='AAPL', period='5y', interval='1d'):
    """
    Fetch historical stock price data from Yahoo Finance.

    Parameters
    ----------
    symbol : str, default 'AAPL'
        The stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft).
    period : str, default '5y'
        The time range of historical data (e.g., '1y', '5y', 'max').
    interval : str, default '1d'
        The data frequency (e.g., '1d' = daily, '1wk' = weekly, '1mo' = monthly).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the stock's historical data with columns like
        Date, Open, High, Low, Close, Adj Close, and Volume.
    """

    # Ensure data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Download data
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
    # Rename columns
    df = df.rename(columns={'Close_AAPL': 'Close', 
                        'Open_AAPL': 'Open',
                        'High_AAPL': 'High',
                        'Low_AAPL': 'Low',
                        'Volume_AAPL': 'Volume'})
    df.reset_index(inplace=True)
    
    # Save to CSV
    filepath = os.path.join(data_dir, f"{symbol}_raw.csv")
    df.to_csv(filepath, index=False)

    return df

if __name__ == '__main__':
    df = fetch('AAPL')
    print(df)