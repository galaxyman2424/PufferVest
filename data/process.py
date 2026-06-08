import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("/home/connor/quant-research/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"
]

def process_ticker(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)

    df["daily_return"]    = df["close"].pct_change()
    df["log_return"]      = np.log(df["close"] / df["close"].shift(1))
    df["vol_20"]          = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["vol_50"]          = df["log_return"].rolling(50).std() * np.sqrt(252)
    df["vol_100"]         = df["log_return"].rolling(100).std() * np.sqrt(252)

    df.dropna(subset=["daily_return"], inplace=True)
    return df

def main():
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df = process_ticker(ticker)
        out = PROCESSED_DIR / f"{ticker}.csv"
        df.to_csv(out)
        print(f"  {len(df)} rows → {out}")
        print(f"  mean daily return : {df['daily_return'].mean():.5f}")
        print(f"  std daily return  : {df['daily_return'].std():.5f}")
        print(f"  skewness          : {df['daily_return'].skew():.4f}")
        print(f"  kurtosis          : {df['daily_return'].kurt():.4f}")

if __name__ == "__main__":
    main()