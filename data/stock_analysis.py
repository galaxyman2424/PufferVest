from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2025-01-01"

def download_ticker(ticker):
    print(f"Downloading {ticker}")

    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    return df

def process_ticker(df):
    df = df.copy()

    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["vol_20"] = (
        df["log_return"].rolling(20).std() * np.sqrt(252)
    )

    df["vol_50"] = (
        df["log_return"].rolling(50).std() * np.sqrt(252)
    )

    df["vol_100"] = (
        df["log_return"].rolling(100).std() * np.sqrt(252)
    )

    return df.dropna()

def summarize_ticker(ticker, df):
    latest = df.iloc[-1]

    return {
        "Ticker": ticker,
        "Date": df.index[-1].strftime("%Y-%m-%d"),
        "Price": round(latest["close"], 2),
        "Volume": int(latest["volume"]),
        "Daily_Return": round(
            latest["daily_return"] * 100,
            2
        ),
        "Vol_20": round(latest["vol_20"], 4),
        "Vol_50": round(latest["vol_50"], 4),
        "Vol_100": round(latest["vol_100"], 4),
    }

def load_tickers_from_file(file_path: str) -> list[str]:
    """
    Reads symbols from a specified text file.
    Expects one ticker per line.
    Returns a clean list of tickers, or an empty list if the file is not found.
    """
    print(f"🔍 Attempting to load tickers from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace (like newlines), and filter out any empty lines
            tickers = [line.strip().upper() for line in f if line.strip()]

        if tickers:
            print(f"✅ Successfully loaded {len(tickers)} tickers.")
        else:
            print("🟡 Warning: The file was read, but no tickers were found. Please check the file content.")

        return tickers

    except FileNotFoundError:
        print("🔥🔥 ERROR: Ticker file not found.")
        print(f"         Please ensure a file named '{file_path}' exists in the same directory.")
        print("         The script cannot run until this file is correctly configured.")
        return []
    except Exception as e:
        print(f"🔥🔥 AN UNEXPECTED ERROR OCCURRED while reading the file: {e}")
        return []


def main():

    tickers = load_tickers_from_file(
        "/home/connor/quant-research/data/stock_tickers.txt"
    )

    portfolio_rows = []

    for ticker in tickers:

        raw_file = RAW_DIR / f"{ticker}.csv"
        processed_file = PROCESSED_DIR / f"{ticker}.csv"

        if raw_file.exists():
            print(f"Using existing {ticker}")
            raw_df = pd.read_csv(
                raw_file,
                index_col=0,
                parse_dates=True
            )
        else:
            raw_df = download_ticker(ticker)

            if raw_df.empty:
                print(f"Skipping {ticker}")
                continue

            raw_df.to_csv(raw_file)

        processed_df = process_ticker(raw_df)

        processed_df.to_csv(processed_file)

        portfolio_rows.append(
            summarize_ticker(
                ticker,
                processed_df
            )
        )

    pd.DataFrame(portfolio_rows).to_csv(
        "data/portfolio_summary.csv",
        index=False
    )

    print(
        f"Saved summary for {len(portfolio_rows)} tickers"
    )

if __name__ == "__main__":
    main()