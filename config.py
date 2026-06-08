from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
VISUALIZATION_DIR = ROOT_DIR / "visualizations" / "output"

# Files
TICKER_FILE = DATA_DIR / "stock_tickers.txt"

# Download settings
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"

# Research parameters
LOOKBACKS = [1, 5, 20, 60]
LOOKAHEADS = [1, 5, 20, 60]

