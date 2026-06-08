import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "^IXIC", "^TNX"
]

START = "2000-01-01"
END   = "2025-01-01"

def download_ticker(ticker: str) -> pd.DataFrame:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df

def main():
    for ticker in TICKERS:
        out = RAW_DIR / f"{ticker.replace('^', '')}.csv"
        if out.exists():
            print(f"  Skipping {ticker}, already exists")
            continue
        df = download_ticker(ticker)
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            continue
        out = RAW_DIR / f"{ticker}.csv"
        df.to_csv(out)
        print(f"  Saved {len(df)} rows → {out}")

if __name__ == "__main__":
    main()