"""
backtest_vectorized.py — Fast vectorized backtester

Consumes the signals DataFrame from signals.py and simulates a long/short
equity strategy using pure pandas/numpy. No row-by-row loops — everything
is matrix operations, making it fast enough for parameter sweeps.

Strategy
--------
  Each day, go long tickers with signal == "long", short tickers with
  signal == "short". Equal-weight within each side. Neutral tickers get
  zero allocation.

  Position sizing is controlled by:
    LONG_WEIGHT   — total gross allocation to longs  (e.g. 1.0 = fully invested)
    SHORT_WEIGHT  — total gross allocation to shorts (e.g. 0.5 = half short)

  Optional: scale position size by regime_size from detector.py if available.

Performance metrics
-------------------
  Sharpe ratio, annualized return, max drawdown, win rate,
  calmar ratio, long/short P&L split, per-factor contribution.

Usage
-----
  python backtest_vectorized.py                  # default params
  python backtest_vectorized.py --sweep          # parameter sweep mode
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product as iterproduct

from config import FEATURE_DIR, PROCESSED_DIR
from utils.tickers import load_tickers


# ── Config ────────────────────────────────────────────────────────────────────

LONG_WEIGHT   = 1.0    # total weight allocated to long side
SHORT_WEIGHT  = 0.5    # total weight allocated to short side
COST_BPS      = 10     # one-way transaction cost in basis points
REBAL_FREQ    = 1      # rebalance every N days (1 = daily)
ANNUAL_FACTOR = 252


# ── Data loading ──────────────────────────────────────────────────────────────

def load_signals() -> pd.DataFrame:
    path = FEATURE_DIR / "signals.csv"
    if not path.exists():
        raise FileNotFoundError(f"signals.csv not found in {FEATURE_DIR}. Run signals.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def pivot_column(signals: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot long-format signals to wide format: rows=date, cols=ticker."""
    return signals.reset_index().pivot(index="date", columns="ticker", values=col)


# ── Position construction ─────────────────────────────────────────────────────

def build_positions(
    signals: pd.DataFrame,
    long_weight:  float = LONG_WEIGHT,
    short_weight: float = SHORT_WEIGHT,
    rebal_freq:   int   = REBAL_FREQ,
) -> pd.DataFrame:
    """
    Returns a wide DataFrame of position weights: rows=date, cols=ticker.
    Weights sum to long_weight on the long side and -short_weight on the short side.
    """
    signal_wide = pivot_column(signals, "signal")

    long_mask  = (signal_wide == "long").astype(float)
    short_mask = (signal_wide == "short").astype(float)

    # Equal weight within each side, normalized to target weight
    n_long  = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    long_weights  = long_mask.div(n_long,  axis=0) * long_weight
    short_weights = short_mask.div(n_short, axis=0) * short_weight * -1

    positions = long_weights.fillna(0) + short_weights.fillna(0)

    # Only rebalance every N days — hold positions in between
    if rebal_freq > 1:
        rebal_dates = positions.index[::rebal_freq]
        positions = positions.reindex(signal_wide.index, method="ffill")
        mask = pd.Series(False, index=positions.index)
        mask[rebal_dates] = True
        positions = positions.where(mask, positions.shift(1))
        positions = positions.fillna(0)

    return positions


# ── Return computation ────────────────────────────────────────────────────────

def compute_portfolio_returns(
    positions: pd.DataFrame,
    signals:   pd.DataFrame,
    cost_bps:  float = COST_BPS,
) -> pd.Series:
    """
    Computes daily portfolio returns net of transaction costs.
    Cost is applied on the absolute change in position weights each day.
    """
    returns_wide = pivot_column(signals, "daily_return")

    # Align
    positions, returns_wide = positions.align(returns_wide, join="inner")

    # Gross daily P&L
    gross = (positions.shift(1) * returns_wide).sum(axis=1)

    # Transaction cost: cost_bps * sum of absolute weight changes
    turnover = positions.diff().abs().sum(axis=1)
    cost     = turnover * (cost_bps / 10_000)

    net = gross - cost
    net.name = "portfolio_return"
    return net.dropna()


# ── Performance metrics ───────────────────────────────────────────────────────

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    curve = equity_curve(returns)
    peak  = curve.cummax()
    dd    = (curve - peak) / peak
    return dd.min()


def sharpe(returns: pd.Series, annual_factor: int = ANNUAL_FACTOR) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annual_factor)


def calmar(returns: pd.Series, annual_factor: int = ANNUAL_FACTOR) -> float:
    ann_ret = returns.mean() * annual_factor
    mdd     = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else np.nan


def summarize(returns: pd.Series, label: str = "Strategy") -> dict:
    ann_ret = returns.mean() * ANNUAL_FACTOR
    ann_vol = returns.std()  * np.sqrt(ANNUAL_FACTOR)
    return {
        "label":       label,
        "ann_return":  round(ann_ret,          4),
        "ann_vol":     round(ann_vol,          4),
        "sharpe":      round(sharpe(returns),  4),
        "max_dd":      round(max_drawdown(returns), 4),
        "calmar":      round(calmar(returns),  4),
        "win_rate":    round((returns > 0).mean(), 4),
        "n_days":      len(returns),
    }


# ── Side split ────────────────────────────────────────────────────────────────

def side_returns(
    positions: pd.DataFrame,
    returns_wide: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Split P&L into long-only and short-only contributions."""
    long_pos  = positions.clip(lower=0)
    short_pos = positions.clip(upper=0)

    long_ret  = (long_pos.shift(1)  * returns_wide).sum(axis=1).dropna()
    short_ret = (short_pos.shift(1) * returns_wide).sum(axis=1).dropna()

    return long_ret, short_ret


# ── Factor contribution ───────────────────────────────────────────────────────

def factor_contribution(
    signals:   pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each factor group score, compute the correlation between
    that score and subsequent 1-day return to gauge factor contribution.
    """
    score_cols = [c for c in signals.columns if c.endswith("_score")]
    returns_wide = pivot_column(signals, "daily_return")

    rows = []
    for col in score_cols:
        score_wide = pivot_column(signals, col)
        score_wide, ret = score_wide.align(returns_wide, join="inner")

        # Flatten and correlate
        flat_score = score_wide.shift(1).stack()
        flat_ret   = ret.stack()
        flat_score, flat_ret = flat_score.align(flat_ret, join="inner")
        corr = flat_score.corr(flat_ret)

        rows.append({"factor": col.replace("_score", ""), "ic": round(corr, 4)})

    return pd.DataFrame(rows).sort_values("ic", ascending=False)


# ── Parameter sweep ───────────────────────────────────────────────────────────

def parameter_sweep(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Grid search over long/short thresholds and rebalance frequency.
    Returns a DataFrame of results sorted by Sharpe.
    """
    from research.backtest.signals import assign_signals, cross_sectional_zscore

    long_thresholds  = [0.25, 0.5, 0.75, 1.0]
    short_thresholds = [-0.25, -0.5, -0.75, -1.0]
    rebal_freqs      = [1, 5, 20]
    cost_bps_list    = [0, 5, 10, 20]

    rows = []
    total = len(long_thresholds) * len(short_thresholds) * len(rebal_freqs) * len(cost_bps_list)
    i = 0

    for lt, st, rf, cb in iterproduct(long_thresholds, short_thresholds, rebal_freqs, cost_bps_list):
        i += 1
        if i % 20 == 0:
            print(f"  Sweep progress: {i}/{total}")

        sig = signals.copy()
        sig["signal"] = assign_signals(sig["composite_zscore"], lt, st)

        pos  = build_positions(sig, rebal_freq=rf)
        rets = compute_portfolio_returns(pos, sig, cost_bps=cb)

        if len(rets) < 100:
            continue

        m = summarize(rets)
        m.update({
            "long_threshold":  lt,
            "short_threshold": st,
            "rebal_freq":      rf,
            "cost_bps":        cb,
        })
        rows.append(m)

    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ── Printing ──────────────────────────────────────────────────────────────────

def print_summary(metrics: dict):
    print(f"\n{'='*55}")
    print(f"  {metrics['label']}")
    print(f"{'='*55}")
    print(f"  Annualized Return : {metrics['ann_return']:>8.2%}")
    print(f"  Annualized Vol    : {metrics['ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio      : {metrics['sharpe']:>8.3f}")
    print(f"  Max Drawdown      : {metrics['max_dd']:>8.2%}")
    print(f"  Calmar Ratio      : {metrics['calmar']:>8.3f}")
    print(f"  Win Rate          : {metrics['win_rate']:>8.2%}")
    print(f"  Trading Days      : {metrics['n_days']:>8,}")


def print_factor_ic(ic_df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  Factor Information Coefficients (IC)")
    print(f"{'='*55}")
    for _, row in ic_df.iterrows():
        if pd.isna(row.ic):
            print(f"  {row.factor:>15}  N/A")
            continue
        bar  = "█" * int(abs(row.ic) * 50)
        sign = "+" if row.ic >= 0 else "-"
        print(f"  {row.factor:>15}  {sign}{abs(row.ic):.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(sweep: bool = False):
    print("Loading signals...")
    signals = load_signals()

    print(f"  {signals['ticker'].nunique()} tickers  |  "
          f"{signals.index.nunique()} dates  |  "
          f"{len(signals):,} rows")

    print("\nBuilding positions...")
    positions    = build_positions(signals)


    returns      = compute_portfolio_returns(positions, signals)
    returns_wide = pivot_column(signals, "daily_return")

    # Overall metrics
    metrics = summarize(returns, label="Factor Strategy (L/S)")
    print_summary(metrics)

    # Side split
    long_ret, short_ret = side_returns(positions, returns_wide.reindex(returns.index))
    print_summary(summarize(long_ret,  label="Long Side"))
    print_summary(summarize(short_ret, label="Short Side"))

    # Factor ICs
    ic_df = factor_contribution(signals, positions)
    print_factor_ic(ic_df)

    # Equity curve to CSV
    curve = equity_curve(returns).rename("equity")
    out   = FEATURE_DIR / "equity_curve_vectorized.csv"
    curve.to_csv(out)
    print(f"\n  Equity curve saved → {out}")

    # Parameter sweep
    if sweep:
        print("\nRunning parameter sweep...")
        sweep_results = parameter_sweep(signals)
        sweep_out = FEATURE_DIR / "sweep_results.csv"
        sweep_results.to_csv(sweep_out, index=False)
        print(f"  Top 10 parameter sets by Sharpe:")
        print(sweep_results.head(10).to_string(index=False))
        print(f"\n  Full sweep saved → {sweep_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()
    run(sweep=args.sweep)