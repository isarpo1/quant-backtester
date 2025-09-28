import yfinance as yf
import pandas as pd

def load_data(ticker="USO", start="2020-01-01", end="2024-12-31"):
    data = yf.download(ticker, start=start, end=end, interval="1d")
    
    # 🔹 Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    return data

def add_indicators(data):
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["Return"] = data["Close"].pct_change()
    return data
