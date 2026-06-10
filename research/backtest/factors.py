"""
factors.py — Unified factor library

Computes five factors per ticker and writes a combined feature CSV to FEATURE_DIR.

Factors
-------
  momentum      — price momentum over 20d and 60d windows
  mean_reversion— z-score of price vs rolling mean (20d, 60d) + RSI(14)
  low_vol       — realized vol rank, vol-of-vol, vol ratio (vol_20 / vol_100)
  value         — P/E and P/B from yfinance snapshot (static, no history)
  quality       — ROE and debt-to-equity from yfinance snapshot (static, no history)

NOTE: Value and Quality use yf.Ticker().info which is a point-in-time snapshot of
current fundamentals. There is NO historical time series. These values are broadcast
across all dates — this introduces look-ahead bias and should only be used for
exploratory factor research, not production backtesting.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

from config import PROCESSED_DIR, FEATURE_DIR
from utils.tickers import load_tickers


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_processed(ticker: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / sig.replace(0, np.nan)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Factor functions ──────────────────────────────────────────────────────────

def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["mom_20"]  = df["close"].pct_change(20)
    out["mom_60"]  = df["close"].pct_change(60)
    out["mom_120"] = df["close"].pct_change(120)
    # Skip the most recent month to avoid short-term reversal contaminating momentum
    out["mom_60_skip1m"] = df["close"].shift(20).pct_change(60)
    return out


def compute_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rev_zscore_20"]  = rolling_zscore(df["close"], 20)
    out["rev_zscore_60"]  = rolling_zscore(df["close"], 60)
    out["rsi_14"]         = rsi(df["close"], 14)
    # Invert so that oversold (low RSI / negative z) → positive mean-reversion score
    out["rev_zscore_20"]  = -out["rev_zscore_20"]
    out["rev_zscore_60"]  = -out["rev_zscore_60"]
    out["rsi_mr"]         = 50 - out["rsi_14"]   # positive when oversold
    return out


def compute_low_vol(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Vol ratio: low values mean the short-term vol is compressed → low-vol signal
    out["vol_ratio"]   = df["vol_20"] / df["vol_100"].replace(0, np.nan)
    # Vol of vol: rolling std of vol_20
    out["vol_of_vol"]  = df["vol_20"].rolling(20).std()
    # Realized vol rank over trailing 252 days (percentile, lower = calmer)
    out["vol_rank_252"] = (
        df["vol_20"]
        .rolling(252)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    # Invert so that low vol → high score (easier to interpret as "factor strength")
    out["vol_ratio"]    = -out["vol_ratio"]
    out["vol_of_vol"]   = -out["vol_of_vol"]
    out["vol_rank_252"] = 1 - out["vol_rank_252"]
    return out


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetches snapshot fundamentals from yfinance.
    Returns NaN for any field that is unavailable.
    LIMITATION: point-in-time snapshot only — no historical series.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":       info.get("trailingPE",          np.nan),
            "pb_ratio":       info.get("priceToBook",         np.nan),
            "roe":            info.get("returnOnEquity",       np.nan),
            "debt_to_equity": info.get("debtToEquity",        np.nan),
        }
    except Exception:
        return {k: np.nan for k in ["pe_ratio", "pb_ratio", "roe", "debt_to_equity"]}


def compute_value(df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """
    Value factor: lower P/E and P/B → higher value score.
    Broadcast static snapshot across all dates.
    """
    out = pd.DataFrame(index=df.index)
    pe  = fundamentals["pe_ratio"]
    pb  = fundamentals["pb_ratio"]
    # Invert so cheap (low multiple) → positive score; cap at ±10 to avoid outlier blow-up
    out["value_pe"] = np.clip(-pe, -10, 10) if not np.isnan(pe) else np.nan
    out["value_pb"] = np.clip(-pb, -10, 10) if not np.isnan(pb) else np.nan
    return out


def compute_quality(df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """
    Quality factor: higher ROE and lower D/E → higher quality score.
    Broadcast static snapshot across all dates.
    """
    out = pd.DataFrame(index=df.index)
    roe = fundamentals["roe"]
    de  = fundamentals["debt_to_equity"]
    out["quality_roe"] = roe if not np.isnan(roe) else np.nan
    out["quality_de"]  = np.clip(-de / 100, -10, 10) if not np.isnan(de) else np.nan
    return out


# ── Composite factor score ────────────────────────────────────────────────────

FACTOR_WEIGHTS = {
    "momentum":       0.25,
    "mean_reversion": 0.20,
    "low_vol":        0.20,
    "value":          0.175,
    "quality":        0.175,
}

def composite_score(df: pd.DataFrame) -> pd.Series:
    groups = {
        "momentum":       ["mom_20", "mom_60", "mom_120", "mom_60_skip1m"],
        "mean_reversion": ["rev_zscore_20", "rev_zscore_60", "rsi_mr"],
        "low_vol":        ["vol_ratio", "vol_of_vol", "vol_rank_252"],
        "value":          ["value_pe", "value_pb"],
        "quality":        ["quality_roe", "quality_de"],
    }

    group_scores = {}
    for name, cols in groups.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        z = df[available].apply(lambda s: (s - s.mean()) / s.std())
        group_score = z.mean(axis=1, skipna=True)
        if group_score.isna().all():
            continue
        group_scores[name] = group_score

    if not group_scores:
        return pd.Series(np.nan, index=df.index)

    total_weight = sum(FACTOR_WEIGHTS[g] for g in group_scores)
    composite = sum(
        group_scores[g] * (FACTOR_WEIGHTS[g] / total_weight)
        for g in group_scores
    )
    return composite


# ── Main ──────────────────────────────────────────────────────────────────────

def build_factors(ticker: str) -> pd.DataFrame:
    df = load_processed(ticker)

    fundamentals = fetch_fundamentals(ticker)

    mom   = compute_momentum(df)
    rev   = compute_mean_reversion(df)
    vol   = compute_low_vol(df)
    val   = compute_value(df, fundamentals)
    qual  = compute_quality(df, fundamentals)

    features = pd.concat([mom, rev, vol, val, qual], axis=1)

    # Keep base columns for context
    features.insert(0, "close",        df["close"])
    features.insert(1, "daily_return", df["daily_return"])
    features.insert(2, "log_return",   df["log_return"])

    features["composite_score"] = composite_score(features)

    features.attrs["ticker"]       = ticker
    features.attrs["fundamentals"] = fundamentals

    return features


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers()

    for ticker in tickers:
        print(f"Building factors — {ticker}")

        try:
            features = build_factors(ticker)
        except FileNotFoundError:
            print(f"  Skipping {ticker}: no processed data found")
            continue

        out = FEATURE_DIR / f"{ticker}_factors.csv"
        features.to_csv(out)

        print(f"  Rows: {len(features)}  |  Cols: {len(features.columns)}")
        print(f"  Fundamentals: {features.attrs['fundamentals']}")
        print(f"  Saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()