"""
signals.py — Factor-to-signal conversion

Loads factor CSVs produced by factors.py, cross-sectionally normalizes them,
and outputs a signal DataFrame consumed by both backtest engines.

Signal logic
------------
  For each date, z-score the composite_score across all tickers.
  Apply thresholds to assign: long / short / neutral

  Default thresholds (configurable):
    z >= LONG_THRESHOLD  → long  ( 1)
    z <= SHORT_THRESHOLD → short (-1)
    otherwise            → neutral (0)

Output columns
--------------
  date, ticker, signal, composite_score, composite_zscore,
  mom_score, rev_score, vol_score, val_score, qual_score
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from config import FEATURE_DIR
from utils.tickers import load_tickers


# ── Config ────────────────────────────────────────────────────────────────────

LONG_THRESHOLD  =  0.5   # cross-sectional z-score cutoff for long
SHORT_THRESHOLD = -0.5   # cross-sectional z-score cutoff for short

# Which factor columns belong to each group (must match factors.py output)
FACTOR_GROUPS = {
    "mom_score":  ["mom_20", "mom_60", "mom_120", "mom_60_skip1m"],
    "rev_score":  ["rev_zscore_20", "rev_zscore_60", "rsi_mr"],
    "vol_score":  ["vol_ratio", "vol_of_vol", "vol_rank_252"],
    "val_score":  ["value_pe", "value_pb"],
    "qual_score": ["quality_roe", "quality_de"],
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_factor_matrix(tickers: list[str]) -> pd.DataFrame:
    """
    Loads all ticker factor CSVs and stacks them into a long-format DataFrame:
      index = (date, ticker)
    """
    frames = []
    for ticker in tickers:
        path = FEATURE_DIR / f"{ticker}_factors.csv"
        if not path.exists():
            print(f"  Warning: no factor file for {ticker}, skipping")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No factor files found in {FEATURE_DIR}. Run factors.py first."
        )

    combined = pd.concat(frames)
    combined.index.name = "date"
    return combined


# ── Factor group scores ───────────────────────────────────────────────────────

def group_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Average of available columns, each standardized over their own history."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    z = df[available].apply(lambda s: (s - s.mean()) / s.std())
    return z.mean(axis=1)


def add_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for score_col, cols in FACTOR_GROUPS.items():
        df[score_col] = group_score(df, cols)
    return df


# ── Cross-sectional normalization ─────────────────────────────────────────────

def cross_sectional_zscore(long_df: pd.DataFrame, col: str) -> pd.Series:
    """
    For each date, z-score `col` across all tickers.
    long_df must have a MultiIndex or a 'date' index with a 'ticker' column.
    """
    return (
        long_df
        .groupby("date")[col]
        .transform(lambda x: (x - x.mean()) / x.std())
    )


# ── Signal assignment ─────────────────────────────────────────────────────────

def assign_signals(
    composite_z: pd.Series,
    long_threshold:  float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> pd.Series:
    signals = pd.Series("neutral", index=composite_z.index)
    signals[composite_z >= long_threshold]  = "long"
    signals[composite_z <= short_threshold] = "short"
    return signals


# ── Main signal builder ───────────────────────────────────────────────────────

def build_signals(
    tickers: list[str] | None = None,
    long_threshold:  float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> pd.DataFrame:
    """
    Full pipeline: load factors → group scores → cross-sectional z-score → signals.

    Returns a long-format DataFrame with columns:
      date, ticker, signal, composite_score, composite_zscore,
      mom_score, rev_score, vol_score, val_score, qual_score
    """
    if tickers is None:
        tickers = load_tickers()

    print(f"Loading factor matrices for {len(tickers)} tickers...")
    df = load_factor_matrix(tickers)

    print("Computing group scores...")
    df = add_group_scores(df)

    print("Cross-sectional z-scoring composite score...")
    df["composite_zscore"] = cross_sectional_zscore(df.reset_index(), "composite_score").values

    print("Assigning signals...")
    df["signal"] = assign_signals(df["composite_zscore"], long_threshold, short_threshold)

    # Keep only the columns the backtester needs
    keep = [
        "ticker", "close", "daily_return", "log_return",
        "composite_score", "composite_zscore", "signal",
        "mom_score", "rev_score", "vol_score", "val_score", "qual_score",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out.index.name = "date"

    return out


# ── Summary / diagnostics ─────────────────────────────────────────────────────

def print_signal_summary(signals: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  Signal Summary")
    print(f"{'='*55}")

    counts = signals.groupby("ticker")["signal"].value_counts().unstack(fill_value=0)
    for col in ["long", "neutral", "short"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["long", "neutral", "short"]]
    counts["total"] = counts.sum(axis=1)
    counts["long%"]  = (counts["long"]  / counts["total"] * 100).round(1)
    counts["short%"] = (counts["short"] / counts["total"] * 100).round(1)
    print(counts.to_string())

    print(f"\n  Latest signals ({signals.index.max().date()}):")
    latest = signals[signals.index == signals.index.max()][["ticker", "signal", "composite_zscore"]]
    latest = latest.sort_values("composite_zscore", ascending=False)
    for _, row in latest.iterrows():
        arrow = "▲" if row.signal == "long" else ("▼" if row.signal == "short" else "—")
        print(f"    {row.ticker:6}  {arrow}  {row.composite_zscore:+.3f}")


def save_signals(signals: pd.DataFrame):
    out = FEATURE_DIR / "signals.csv"
    signals.to_csv(out)
    print(f"\n  Saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signals = build_signals()
    print_signal_summary(signals)
    save_signals(signals)