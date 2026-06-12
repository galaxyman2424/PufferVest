"""
strategy_runner.py — Phase 11: Strategy Library & Comparison

Runs six strategies through a unified backtesting engine and produces
a side-by-side performance comparison against SPY and QQQ benchmarks.

Strategies
----------
  1. Buy & Hold SPY          — benchmark
  2. Buy & Hold QQQ          — benchmark
  3. Momentum (12-1)         — classic academic momentum, monthly rebalance
  4. Mean Reversion          — RSI + z-score oversold/overbought, weekly rebalance
  5. Factor L/S              — composite factor strategy (existing signals.py output)
  6. Regime-Aware Factor     — Factor L/S scaled by Bayesian opportunity/risk scores

Timeframes
----------
  Each strategy is run at rebalance frequencies: 1d, 21d (monthly), 63d (quarterly).
  This shows how each strategy's edge changes with holding period.

Outputs
-------
  data/features/strategy_comparison.csv
  visualizations/output/strategy_curves.png

Usage
-----
  python -m research.strategies.strategy_runner
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path

from config import FEATURE_DIR, PROCESSED_DIR, VISUALIZATION_DIR
from utils.tickers import load_tickers

# Re-use metrics from existing backtester — no duplication
from research.backtesting.backtest_vectorized import (
    equity_curve,
    max_drawdown,
    sharpe,
    calmar,
    summarize,
    pivot_column,
    compute_portfolio_returns,
)

# ── Config ────────────────────────────────────────────────────────────────────

COST_BPS      = 10
ANNUAL_FACTOR = 252

REBAL_FREQS = {
    "daily":     1,
    "monthly":  21,
    "quarterly": 63,
}

# Timeframes to run for signal-based strategies
# Buy & hold benchmarks only need one run (rebal_freq irrelevant)
STRATEGY_TIMEFRAMES = {
    "momentum":       ["daily", "monthly", "quarterly"],
    "mean_reversion": ["daily", "monthly", "quarterly"],
    "factor_ls":      ["daily", "monthly", "quarterly"],
    "regime_aware":   ["daily", "monthly", "quarterly"],
}

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_signals() -> pd.DataFrame:
    path = FEATURE_DIR / "signals.csv"
    if not path.exists():
        raise FileNotFoundError(f"signals.csv not found. Run signals.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_bayesian() -> pd.DataFrame:
    path = FEATURE_DIR / "bayesian_combined.csv"
    if not path.exists():
        print("  Warning: bayesian_combined.csv not found — regime-aware strategy disabled.")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_price_series(ticker: str) -> pd.Series:
    path = PROCESSED_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["daily_return"].rename(ticker)


def load_all_prices(tickers: list[str]) -> pd.DataFrame:
    frames = {t: load_price_series(t) for t in tickers}
    return pd.DataFrame(frames).dropna(how="all")


# ── Strategy 1 & 2: Buy and Hold ─────────────────────────────────────────────

def strategy_buy_and_hold(returns: pd.DataFrame, ticker: str) -> pd.Series:
    """100% long a single ticker, no rebalancing, no costs."""
    if ticker not in returns.columns:
        raise ValueError(f"{ticker} not in returns DataFrame")
    r = returns[ticker].dropna()
    r.name = "portfolio_return"
    return r


# ── Strategy 3: Momentum (12-1) ───────────────────────────────────────────────

def strategy_momentum(
    returns: pd.DataFrame,
    rebal_freq: int = 21,
    n_long: int = 3,
    cost_bps: float = COST_BPS,
) -> pd.Series:
    """
    Classic 12-1 momentum: rank tickers by 12-month return skipping most
    recent month (to avoid short-term reversal). Go long top N tickers.
    Equal weight. Rebalance every rebal_freq days.
    Long-only — matches how momentum is typically implemented in practice.
    """
    # Reconstruct price index from returns
    prices = (1 + returns.fillna(0)).cumprod()

    # 12-month return skipping 1 month: (price[t-21] / price[t-252]) - 1
    mom_signal = prices.shift(21) / prices.shift(252) - 1

    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    rebal_dates = returns.index[::rebal_freq]

    for date in rebal_dates:
        if date not in mom_signal.index:
            continue
        scores = mom_signal.loc[date].dropna()
        if len(scores) < 2:
            continue
        top = scores.nlargest(min(n_long, len(scores))).index
        weight = 1.0 / len(top)
        positions.loc[date, :] = 0.0
        positions.loc[date, top] = weight

    # Hold between rebalance dates
    positions = positions.replace(0.0, np.nan)
    positions = positions.ffill().fillna(0.0)
    # Zero out before first rebalance signal
    first_signal = positions[positions.sum(axis=1) > 0].index
    if len(first_signal) > 0:
        positions.loc[:first_signal[0]] = 0.0

    gross   = (positions.shift(1) * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    cost    = turnover * (cost_bps / 10_000)
    net     = (gross - cost).dropna()
    net.name = "portfolio_return"
    return net


# ── Strategy 4: Mean Reversion ────────────────────────────────────────────────

def strategy_mean_reversion(
    returns:    pd.DataFrame,
    signals_df: pd.DataFrame,
    rebal_freq: int   = 5,
    rsi_low:    float = 35.0,
    rsi_high:   float = 65.0,
    z_low:      float = -1.5,
    z_high:     float =  1.5,
    cost_bps:   float = COST_BPS,
) -> pd.Series:
    """
    Buy tickers that are oversold (RSI < rsi_low AND rev_zscore < z_low).
    Sell (go short) tickers that are overbought (RSI > rsi_high AND rev_zscore > z_high).
    Equal weight within each side. Rebalance every rebal_freq days.

    Pulls RSI and rev_zscore from the factor files since they're already computed.
    """
    tickers = returns.columns.tolist()

    # Load factor files for rsi and z-score
    rsi_wide = {}
    rev_wide = {}
    for ticker in tickers:
        fpath = FEATURE_DIR / f"{ticker}_factors.csv"
        if not fpath.exists():
            continue
        fdf = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if "rsi_14" in fdf.columns:
            rsi_wide[ticker] = fdf["rsi_14"]
        # rev_zscore_20 is inverted in factors.py — reload raw
        if "rev_zscore_20" in fdf.columns:
            rev_wide[ticker] = -fdf["rev_zscore_20"]  # un-invert

    if not rsi_wide:
        print("  Warning: no RSI data found for mean reversion strategy")
        return pd.Series(dtype=float, name="portfolio_return")

    rsi_df = pd.DataFrame(rsi_wide).reindex(returns.index)
    rev_df = pd.DataFrame(rev_wide).reindex(returns.index)

    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    rebal_dates = returns.index[::rebal_freq]

    for date in rebal_dates:
        if date not in rsi_df.index:
            continue
        rsi_row = rsi_df.loc[date]
        rev_row = rev_df.loc[date]

        oversold   = (rsi_row < rsi_low)  & (rev_row < z_low)
        overbought = (rsi_row > rsi_high) & (rev_row > z_high)

        long_tickers  = oversold[oversold].index.tolist()
        short_tickers = overbought[overbought].index.tolist()

        positions.loc[date, :] = 0.0
        if long_tickers:
            positions.loc[date, long_tickers] = 1.0 / len(long_tickers)
        if short_tickers:
            positions.loc[date, short_tickers] = -0.5 / len(short_tickers)

    positions = positions.replace(0.0, np.nan)
    positions = positions.ffill().fillna(0.0)

    gross    = (positions.shift(1) * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    cost     = turnover * (cost_bps / 10_000)
    net      = (gross - cost).dropna()
    net.name = "portfolio_return"
    return net


# ── Strategy 5: Factor L/S ────────────────────────────────────────────────────

def strategy_factor_ls(
    signals:    pd.DataFrame,
    rebal_freq: int   = 1,
    long_weight: float = 1.0,
    short_weight: float = 0.5,
    cost_bps:   float = COST_BPS,
) -> pd.Series:
    """
    Composite factor long/short strategy — reuses existing signals.csv output.
    Mirrors the logic in backtest_vectorized.py exactly.
    """
    signal_wide = pivot_column(signals, "signal")
    returns_wide = pivot_column(signals, "daily_return")

    long_mask  = (signal_wide == "long").astype(float)
    short_mask = (signal_wide == "short").astype(float)

    n_long  = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    long_w  = long_mask.div(n_long,  axis=0) * long_weight
    short_w = short_mask.div(n_short, axis=0) * short_weight * -1
    positions = long_w.fillna(0) + short_w.fillna(0)

    if rebal_freq > 1:
        rebal_dates = positions.index[::rebal_freq]
        held = positions.copy()
        for i in range(1, len(positions)):
            date = positions.index[i]
            if date not in rebal_dates:
                held.iloc[i] = held.iloc[i - 1]
        positions = held

    positions, returns_wide = positions.align(returns_wide, join="inner")
    gross    = (positions.shift(1) * returns_wide).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    cost     = turnover * (cost_bps / 10_000)
    net      = (gross - cost).dropna()
    net.name = "portfolio_return"
    return net


# ── Strategy 6: Regime-Aware Factor ──────────────────────────────────────────

def strategy_regime_aware(
    signals:    pd.DataFrame,
    bayes_df:   pd.DataFrame,
    rebal_freq: int   = 1,
    cost_bps:   float = COST_BPS,
) -> pd.Series:
    """
    Factor L/S with position sizing scaled by Bayesian posteriors.

    Scaling rules:
      gross_scale = opportunity_score * (1 - risk_score)
      This produces a scalar in [0, 1] that:
        - approaches 1 when opportunity is high and risk is low  (bull, low_vol)
        - approaches 0 when risk is high and opportunity is low  (crash, bear)
      Minimum gross scale is 0.1 so we never go completely flat.

    The scale is applied symmetrically to both long and short sides.
    """
    if bayes_df.empty:
        print("  Regime-aware strategy: no Bayesian data, falling back to Factor L/S")
        return strategy_factor_ls(signals, rebal_freq=rebal_freq, cost_bps=cost_bps)

    signal_wide  = pivot_column(signals, "signal")
    returns_wide = pivot_column(signals, "daily_return")

    long_mask  = (signal_wide == "long").astype(float)
    short_mask = (signal_wide == "short").astype(float)

    n_long  = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    long_w  = long_mask.div(n_long,  axis=0).fillna(0)
    short_w = (short_mask.div(n_short, axis=0) * -1).fillna(0)
    base_positions = long_w + short_w

    # Build SPY-level gross scale from Bayesian posteriors
    # Use SPY as market proxy; fall back to cross-ticker mean if SPY absent
    if "SPY" in bayes_df["ticker"].values if "ticker" in bayes_df.columns else False:
        spy_bayes = bayes_df[bayes_df["ticker"] == "SPY"][
            ["opportunity_score", "risk_score"]
        ]
    else:
        # Average across all tickers per date
        spy_bayes = (
            bayes_df.groupby(bayes_df.index)[["opportunity_score", "risk_score"]]
            .mean()
        )

    spy_bayes = spy_bayes.reindex(signal_wide.index).ffill().fillna(
        {"opportunity_score": 0.5, "risk_score": 0.3}
    )

    gross_scale = (
        spy_bayes["opportunity_score"] * (1 - spy_bayes["risk_score"])
    ).clip(lower=0.1, upper=1.0)

    # Apply scale to positions
    positions = base_positions.mul(gross_scale, axis=0)

    if rebal_freq > 1:
        rebal_dates = positions.index[::rebal_freq]
        held = positions.copy()
        for i in range(1, len(positions)):
            date = positions.index[i]
            if date not in rebal_dates:
                held.iloc[i] = held.iloc[i - 1]
        positions = held

    positions, returns_wide = positions.align(returns_wide, join="inner")
    gross    = (positions.shift(1) * returns_wide).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    cost     = turnover * (cost_bps / 10_000)
    net      = (gross - cost).dropna()
    net.name = "portfolio_return"
    return net


# ── Run all strategies ────────────────────────────────────────────────────────

def run_all(
    signals:  pd.DataFrame,
    returns:  pd.DataFrame,
    bayes_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Returns a dict of label → daily returns Series for every
    strategy × timeframe combination.
    """
    results = {}

    # ── Benchmarks ─────────────────────────────────────────────────────────
    for ticker in ["SPY", "QQQ"]:
        if ticker in returns.columns:
            r = strategy_buy_and_hold(returns, ticker)
            results[f"Buy&Hold {ticker}"] = r
            print(f"  Buy&Hold {ticker}: {len(r)} days")

    # ── Momentum ───────────────────────────────────────────────────────────
    for tf_name, rf in REBAL_FREQS.items():
        if tf_name not in STRATEGY_TIMEFRAMES["momentum"]:
            continue
        label = f"Momentum ({tf_name})"
        r = strategy_momentum(returns, rebal_freq=rf)
        if len(r) > 100:
            results[label] = r
            print(f"  {label}: {len(r)} days")

    # ── Mean Reversion ─────────────────────────────────────────────────────
    for tf_name, rf in REBAL_FREQS.items():
        if tf_name not in STRATEGY_TIMEFRAMES["mean_reversion"]:
            continue
        label = f"MeanRev ({tf_name})"
        r = strategy_mean_reversion(returns, signals, rebal_freq=rf)
        if len(r) > 100:
            results[label] = r
            print(f"  {label}: {len(r)} days")

    # ── Factor L/S ─────────────────────────────────────────────────────────
    for tf_name, rf in REBAL_FREQS.items():
        if tf_name not in STRATEGY_TIMEFRAMES["factor_ls"]:
            continue
        label = f"Factor L/S ({tf_name})"
        r = strategy_factor_ls(signals, rebal_freq=rf)
        if len(r) > 100:
            results[label] = r
            print(f"  {label}: {len(r)} days")

    # ── Regime-Aware Factor ────────────────────────────────────────────────
    if not bayes_df.empty:
        for tf_name, rf in REBAL_FREQS.items():
            if tf_name not in STRATEGY_TIMEFRAMES["regime_aware"]:
                continue
            label = f"Regime-Aware ({tf_name})"
            r = strategy_regime_aware(signals, bayes_df, rebal_freq=rf)
            if len(r) > 100:
                results[label] = r
                print(f"  {label}: {len(r)} days")

    return results


# ── Metrics table ─────────────────────────────────────────────────────────────

def build_metrics_table(results: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, r in results.items():
        if len(r) < 50:
            continue
        m = summarize(r, label=label)

        # Additional metrics
        ann_ret = r.mean() * ANNUAL_FACTOR
        ann_vol = r.std() * np.sqrt(ANNUAL_FACTOR)

        # Sortino: downside deviation only
        downside = r[r < 0]
        down_vol = downside.std() * np.sqrt(ANNUAL_FACTOR) if len(downside) > 0 else np.nan
        sortino  = ann_ret / down_vol if down_vol and down_vol > 0 else np.nan

        # Best / worst year
        annual = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        best_yr  = annual.max()
        worst_yr = annual.min()

        m["sortino"]   = round(sortino,   3) if not np.isnan(sortino) else np.nan
        m["best_year"] = round(best_yr,   4)
        m["worst_year"] = round(worst_yr, 4)
        rows.append(m)

    df = pd.DataFrame(rows).set_index("label")
    col_order = [
        "ann_return", "ann_vol", "sharpe", "sortino",
        "max_dd", "calmar", "win_rate",
        "best_year", "worst_year", "n_days",
    ]
    return df[[c for c in col_order if c in df.columns]]


# ── Visualization ─────────────────────────────────────────────────────────────

# Color palette — benchmarks in gray, strategies in distinct colors
STRATEGY_COLORS = {
    "Buy&Hold SPY":       "#7f8c8d",
    "Buy&Hold QQQ":       "#95a5a6",
    "Momentum (daily)":   "#3498db",
    "Momentum (monthly)": "#2980b9",
    "Momentum (quarterly)":"#1a5276",
    "MeanRev (daily)":    "#e67e22",
    "MeanRev (monthly)":  "#d35400",
    "MeanRev (quarterly)":"#784212",
    "Factor L/S (daily)": "#2ecc71",
    "Factor L/S (monthly)":"#27ae60",
    "Factor L/S (quarterly)":"#1a6b3c",
    "Regime-Aware (daily)":"#9b59b6",
    "Regime-Aware (monthly)":"#8e44ad",
    "Regime-Aware (quarterly)":"#6c3483",
}

def _color(label: str) -> str:
    return STRATEGY_COLORS.get(label, "#2c3e50")


def plot_results(
    results: dict[str, pd.Series],
    metrics: pd.DataFrame,
):
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Phase 11 — Strategy Comparison", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── Panel 1: Equity curves (all strategies) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for label, r in results.items():
        curve = equity_curve(r)
        lw    = 2.5 if "Buy&Hold" in label else 1.2
        alpha = 0.9 if "Buy&Hold" in label else 0.75
        ax1.plot(curve.index, curve, label=label, color=_color(label), lw=lw, alpha=alpha)
    ax1.set_title("Equity Curves — All Strategies (log scale)", fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.legend(fontsize=6, ncol=3, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Drawdown comparison ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for label, r in results.items():
        curve = equity_curve(r)
        peak  = curve.cummax()
        dd    = (curve - peak) / peak
        lw    = 2.0 if "Buy&Hold" in label else 0.9
        ax2.plot(dd.index, dd * 100, label=label, color=_color(label), lw=lw, alpha=0.8)
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Annual returns heatmap ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    annual_data = {}
    for label, r in results.items():
        annual = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        annual_data[label] = annual

    annual_df = pd.DataFrame(annual_data)
    annual_df.index = annual_df.index.year

    im = ax3.imshow(
        annual_df.T.values * 100,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-40,
        vmax=40,
    )
    ax3.set_xticks(range(len(annual_df.index)))
    ax3.set_xticklabels(annual_df.index, rotation=45, fontsize=7)
    ax3.set_yticks(range(len(annual_df.columns)))
    ax3.set_yticklabels(annual_df.columns, fontsize=6)
    ax3.set_title("Annual Returns Heatmap (%)", fontweight="bold")
    plt.colorbar(im, ax=ax3, fraction=0.03)

    # ── Panel 4: Sharpe ratio bar chart ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    sharpes = metrics["sharpe"].sort_values(ascending=True)
    colors  = [_color(l) for l in sharpes.index]
    ax4.barh(sharpes.index, sharpes.values, color=colors, alpha=0.85)
    ax4.axvline(0, color="black", lw=0.8)
    ax4.set_title("Sharpe Ratio by Strategy", fontweight="bold")
    ax4.set_xlabel("Sharpe Ratio")
    ax4.tick_params(axis="y", labelsize=7)

    # ── Panel 5: Risk/Return scatter ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    for label in metrics.index:
        x   = metrics.loc[label, "ann_vol"]  * 100
        y   = metrics.loc[label, "ann_return"] * 100
        ax5.scatter(x, y, color=_color(label), s=60, zorder=3)
        ax5.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(5, 3),
            fontsize=6, color=_color(label),
        )
    ax5.axhline(0, color="black", lw=0.5, linestyle="--")
    ax5.set_title("Risk / Return Scatter", fontweight="bold")
    ax5.set_xlabel("Annualized Volatility (%)")
    ax5.set_ylabel("Annualized Return (%)")
    ax5.grid(alpha=0.3)

    out = VISUALIZATION_DIR / "strategy_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved → {out}")


# ── Console print ─────────────────────────────────────────────────────────────

def print_metrics(metrics: pd.DataFrame):
    print(f"\n{'='*100}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*100}")
    print(f"  {'Strategy':<30} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>7} "
          f"{'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'WinRate':>8} "
          f"{'BestYr':>8} {'WorstYr':>9}")
    print(f"  {'-'*98}")

    for label, row in metrics.iterrows():
        def _f(col, fmt):
            v = row.get(col, np.nan)
            return fmt.format(v) if not pd.isna(v) else "   N/A"

        print(
            f"  {label:<30} "
            f"{_f('ann_return', '{:>7.2%}')}  "
            f"{_f('ann_vol',    '{:>7.2%}')}  "
            f"{_f('sharpe',     '{:>6.3f}')}  "
            f"{_f('sortino',    '{:>7.3f}')}  "
            f"{_f('max_dd',     '{:>7.2%}')}  "
            f"{_f('calmar',     '{:>6.3f}')}  "
            f"{_f('win_rate',   '{:>7.2%}')}  "
            f"{_f('best_year',  '{:>7.2%}')}  "
            f"{_f('worst_year', '{:>8.2%}')}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tickers = load_tickers()

    print("="*60)
    print("  Phase 11 — Strategy Library & Comparison")
    print("="*60)

    print("\nLoading data...")
    signals  = load_signals()
    returns  = load_all_prices(tickers)
    bayes_df = load_bayesian()

    print(f"  Signals : {len(signals):,} rows  |  "
          f"{signals.index.min().date()} → {signals.index.max().date()}")
    print(f"  Returns : {returns.shape[0]} days × {returns.shape[1]} tickers")
    if not bayes_df.empty:
        print(f"  Bayesian: {len(bayes_df):,} rows loaded")

    print("\nRunning strategies...")
    results = run_all(signals, returns, bayes_df)

    print(f"\n  {len(results)} strategies completed")

    print("\nBuilding metrics table...")
    metrics = build_metrics_table(results)

    # Save
    out = FEATURE_DIR / "strategy_comparison.csv"
    metrics.to_csv(out)
    print(f"  Saved → {out}")

    # Print
    print_metrics(metrics)

    # Plot
    print("\nGenerating visualization...")
    plot_results(results, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()