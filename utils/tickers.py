from config import TICKER_FILE

def load_tickers():
    with open(TICKER_FILE) as f:
        return [
            line.strip().upper()
            for line in f
            if line.strip()
        ]