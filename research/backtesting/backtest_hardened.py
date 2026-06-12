"""
backtest_hardened.py — Hardened vectorized backtester

Extends backtest_vectorized.py with:
  1. Hard train/test split — parameter tuning is confined to train period only.
     Test period is evaluated exactly once and never used for optimization.
  2. Market impact slippage model — cost scales with trade size, not flat bps.
  3. Per-ticker position size cap — prevents concentration in a single name.
  4. Benchmark-relative metrics — alpha, beta, information ratio, tracking error vs SPY.

Usage
-----
  python backtest_hardened.py                        # factor strategy, train+test
  python backtest_hardened.py --train-only           # factor strategy, training period only
  python backtest_hardened.py --sweep                # parameter sweep on TRAIN period only
  python backtest_hardened.py --test-only            # final evaluation on TEST period only
                                                     # (use sparingly — this is your held-out set)
  python backtest_hardened.py --ml rf                # use ML Random Forest signals
  python backtest_hardened.py --ml gbm               # use ML Gradient Boosting signals
  python backtest_hardened.py --ml both              # run factor, RF, and GBM side-by-side
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

LONG_WEIGHT      = 1.0     # total gross weight on long side
SHORT_WEIGHT     = 0.5     # total gross weight on short side
BASE_COST_BPS    = 5       # base one-way transaction cost (bps)
IMPACT_BPS       = 10      # market impact coefficient (bps per unit of weight change)
                            # effective cost = BASE_COST_BPS + IMPACT_BPS * |Δweight|
REBAL_FREQ       = 1       # rebalance every N days
MAX_SINGLE_WEIGHT = 0.30   # maximum absolute weight for any single ticker
ANNUAL_FACTOR    = 252

# Train/test split
TRAIN_END  = "2018-12-31"  # last day of training period (inclusive)
TEST_START = "2019-01-01"  # first day of test period


# ── Data loading ──────────────────────────────────────────────────────────────

def load_signals() -> pd.DataFrame:
    path = FEATURE_DIR / "signals.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"signals.csv not found in {FEATURE_DIR}. Run signals.py first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def split_signals(
    signals: pd.DataFrame,
    train_end:  str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hard partition signals into train and test sets.
    No data leaks between them — test period is never touched during tuning.
    """
    train = signals[signals.index <= train_end].copy()
    test  = signals[signals.index >= test_start].copy()
    return train, test


def load_spy_returns() -> pd.Series:
    """Load SPY daily returns for benchmark comparison."""
    path = PROCESSED_DIR / "SPY.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="SPY")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["daily_return"].rename("SPY")


def load_ml_signals() -> pd.DataFrame | None:
    """
    Load ml_signals.csv produced by ml_signals.py.
    Returns None if the file doesn't exist (falls back to factor signals).
    """
    path = FEATURE_DIR / "ml_signals.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def inject_ml_signals(
    signals:    pd.DataFrame,
    ml_signals: pd.DataFrame,
    model:      str,              # "rf" or "gbm"
) -> pd.DataFrame:
    """
    Replace the 'signal' column in signals with the ML model's signal.

    Merges on (date, ticker). Rows where the ML signal is missing (NaN)
    fall back to "neutral" rather than the original factor signal, so the
    comparison is clean — you're testing the ML signal on its own, not
    patching gaps with the factor signal.

    Args:
        signals:    factor signals DataFrame (long format, date index)
        ml_signals: output of ml_signals.py (date index, ticker column)
        model:      "rf" or "gbm"
    """
    col = f"ml_{model}_signal"
    if col not in ml_signals.columns:
        raise ValueError(
            f"Column '{col}' not found in ml_signals.csv. "
            f"Available: {list(ml_signals.columns)}"
        )

    out = signals.copy()

    # Merge ML signal column on (date, ticker)
    ml_col = ml_signals.reset_index()[["date", "ticker", col]].copy()
    out    = out.reset_index().merge(ml_col, on=["date", "ticker"], how="left")
    out    = out.set_index("date")
    out.index.name = "date"

    # Replace signal column; missing ML predictions → neutral
    out["signal"] = out[col].fillna("neutral")
    out = out.drop(columns=[col])

    return out


def pivot_column(signals: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot long-format signals to wide format: rows=date, cols=ticker."""
    return signals.reset_index().pivot(index="date", columns="ticker", values=col)


# ── Position construction ─────────────────────────────────────────────────────

def build_positions(
    signals:          pd.DataFrame,
    long_weight:      float = LONG_WEIGHT,
    short_weight:     float = SHORT_WEIGHT,
    rebal_freq:       int   = REBAL_FREQ,
    max_single_weight: float = MAX_SINGLE_WEIGHT,
) -> pd.DataFrame:
    """
    Returns a wide DataFrame of position weights: rows=date, cols=ticker.

    Steps:
      1. Equal-weight within long and short sides, normalized to target gross weight.
      2. Cap any single ticker at +/- max_single_weight.
      3. Re-normalize after capping so gross weights still sum to targets.
      4. Apply rebalance frequency (hold positions between rebal dates).
    """
    signal_wide = pivot_column(signals, "signal")

    long_mask  = (signal_wide == "long").astype(float)
    short_mask = (signal_wide == "short").astype(float)

    n_long  = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    long_weights  = long_mask.div(n_long,  axis=0) * long_weight
    short_weights = short_mask.div(n_short, axis=0) * short_weight * -1

    positions = long_weights.fillna(0) + short_weights.fillna(0)

    # ── Position size cap ─────────────────────────────────────────────────────
    # Clip at +/- max_single_weight, then re-scale each side so gross exposure
    # is preserved (prevents unintended deleveraging when many names are capped).
    positions = _apply_position_cap(positions, max_single_weight, long_weight, short_weight)

    # ── Rebalance frequency ───────────────────────────────────────────────────
    if rebal_freq > 1:
        rebal_dates = positions.index[::rebal_freq]
        rebal_mask  = pd.Series(False, index=positions.index)
        rebal_mask[rebal_dates] = True
        # Forward-fill positions on non-rebal days
        positions = positions.where(rebal_mask, other=np.nan).ffill().fillna(0)

    return positions


def _apply_position_cap(
    positions:         pd.DataFrame,
    max_weight:        float,
    target_long_gross: float,
    target_short_gross: float,
) -> pd.DataFrame:
    """
    Cap each individual weight at +/- max_weight, then re-normalize each side
    so total gross exposure is preserved.
    """
    pos = positions.copy()

    # Separate sides
    long_pos  = pos.clip(lower=0)
    short_pos = pos.clip(upper=0)

    # Cap magnitudes
    long_pos  = long_pos.clip(upper=max_weight)
    short_pos = short_pos.clip(lower=-max_weight)

    # Re-normalize long side
    long_gross = long_pos.sum(axis=1).replace(0, np.nan)
    long_pos   = long_pos.div(long_gross, axis=0) * target_long_gross

    # Re-normalize short side
    short_gross = short_pos.abs().sum(axis=1).replace(0, np.nan)
    short_pos   = short_pos.div(short_gross, axis=0) * target_short_gross

    return long_pos.fillna(0) + short_pos.fillna(0)


# ── Return computation ────────────────────────────────────────────────────────

def compute_portfolio_returns(
    positions:    pd.DataFrame,
    signals:      pd.DataFrame,
    base_cost_bps: float = BASE_COST_BPS,
    impact_bps:   float  = IMPACT_BPS,
) -> pd.Series:
    """
    Computes daily portfolio returns net of transaction costs.

    Slippage model:
      For each trade (weight change), cost = base_cost_bps + impact_bps * |Δw|
      This means a 10% position change costs more than a 1% tweak,
      which is more realistic than flat bps across all trade sizes.

      effective_cost_per_unit = (base_cost_bps + impact_bps * |Δw|) / 10_000
      total_cost_day = sum over tickers of effective_cost_per_unit * |Δw|
                     = base_cost_bps/10_000 * turnover
                       + impact_bps/10_000 * sum(Δw²)
    """
    returns_wide = pivot_column(signals, "daily_return")
    positions, returns_wide = positions.align(returns_wide, join="inner")

    # Gross P&L
    gross = (positions.shift(1) * returns_wide).sum(axis=1)

    # Market impact slippage
    weight_changes = positions.diff().fillna(0)
    turnover       = weight_changes.abs().sum(axis=1)               # linear term
    impact_term    = (weight_changes ** 2).sum(axis=1)              # quadratic term

    cost = (base_cost_bps / 10_000) * turnover + (impact_bps / 10_000) * impact_term

    net = gross - cost
    net.name = "portfolio_return"
    return net.dropna()


# ── Performance metrics ───────────────────────────────────────────────────────

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    curve = equity_curve(returns)
    peak  = curve.cummax()
    return ((curve - peak) / peak).min()


def sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(ANNUAL_FACTOR)


def calmar(returns: pd.Series) -> float:
    ann_ret = returns.mean() * ANNUAL_FACTOR
    mdd     = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else np.nan


def sortino(returns: pd.Series) -> float:
    """Sortino ratio — penalizes only downside volatility."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return (returns.mean() / downside.std()) * np.sqrt(ANNUAL_FACTOR)


def benchmark_metrics(
    returns:    pd.Series,
    spy_returns: pd.Series,
) -> dict:
    """
    Compute alpha, beta, information ratio, and tracking error vs SPY.

    Beta  = cov(strategy, SPY) / var(SPY)
    Alpha = annualized(strategy mean - beta * SPY mean)
    TE    = annualized std of (strategy - SPY)
    IR    = annualized mean(strategy - SPY) / TE
    """
    aligned = pd.concat([returns, spy_returns], axis=1, join="inner")
    aligned.columns = ["strategy", "spy"]
    aligned = aligned.dropna()

    if len(aligned) < 30:
        return {"alpha": np.nan, "beta": np.nan, "info_ratio": np.nan, "tracking_error": np.nan}

    cov_matrix  = aligned.cov()
    beta        = cov_matrix.loc["strategy", "spy"] / cov_matrix.loc["spy", "spy"]
    alpha_daily = aligned["strategy"].mean() - beta * aligned["spy"].mean()
    alpha_ann   = alpha_daily * ANNUAL_FACTOR

    active_returns  = aligned["strategy"] - aligned["spy"]
    tracking_error  = active_returns.std() * np.sqrt(ANNUAL_FACTOR)
    info_ratio      = (active_returns.mean() * ANNUAL_FACTOR) / tracking_error if tracking_error > 0 else np.nan

    return {
        "alpha":          round(alpha_ann,     4),
        "beta":           round(beta,          4),
        "info_ratio":     round(info_ratio,    4),
        "tracking_error": round(tracking_error, 4),
    }


def summarize(
    returns:     pd.Series,
    spy_returns: pd.Series | None = None,
    label:       str = "Strategy",
) -> dict:
    ann_ret = returns.mean() * ANNUAL_FACTOR
    ann_vol = returns.std()  * np.sqrt(ANNUAL_FACTOR)

    metrics = {
        "label":      label,
        "ann_return": round(ann_ret,             4),
        "ann_vol":    round(ann_vol,             4),
        "sharpe":     round(sharpe(returns),     4),
        "sortino":    round(sortino(returns),    4) if not np.isnan(sortino(returns)) else np.nan,
        "max_dd":     round(max_drawdown(returns), 4),
        "calmar":     round(calmar(returns),     4),
        "win_rate":   round((returns > 0).mean(), 4),
        "n_days":     len(returns),
    }

    if spy_returns is not None:
        metrics.update(benchmark_metrics(returns, spy_returns))

    return metrics


# ── Side split ────────────────────────────────────────────────────────────────

def side_returns(
    positions:    pd.DataFrame,
    returns_wide: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    long_pos  = positions.clip(lower=0)
    short_pos = positions.clip(upper=0)
    long_ret  = (long_pos.shift(1)  * returns_wide).sum(axis=1).dropna()
    short_ret = (short_pos.shift(1) * returns_wide).sum(axis=1).dropna()
    return long_ret, short_ret


# ── Factor IC ─────────────────────────────────────────────────────────────────

def factor_contribution(
    signals:   pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    score_cols   = [c for c in signals.columns if c.endswith("_score")]
    returns_wide = pivot_column(signals, "daily_return")

    rows = []
    for col in score_cols:
        score_wide = pivot_column(signals, col)
        score_wide, ret = score_wide.align(returns_wide, join="inner")
        flat_score = score_wide.shift(1).stack()
        flat_ret   = ret.stack()
        flat_score, flat_ret = flat_score.align(flat_ret, join="inner")
        corr = flat_score.corr(flat_ret)
        rows.append({"factor": col.replace("_score", ""), "ic": round(corr, 4)})

    return pd.DataFrame(rows).sort_values("ic", ascending=False)


# ── Parameter sweep (train period only) ──────────────────────────────────────

def parameter_sweep(train_signals: pd.DataFrame, spy_returns: pd.Series) -> pd.DataFrame:
    """
    Grid search over thresholds, rebal frequency, and cost assumptions.
    MUST only be called on training data — never pass test signals here.
    """
    from signals import assign_signals  # adjust import path to your project layout

    long_thresholds  = [0.25, 0.5, 0.75, 1.0]
    short_thresholds = [-0.25, -0.5, -0.75, -1.0]
    rebal_freqs      = [1, 5, 20]
    cost_bps_list    = [5, 10, 20]

    rows  = []
    total = len(long_thresholds) * len(short_thresholds) * len(rebal_freqs) * len(cost_bps_list)

    for i, (lt, st, rf, cb) in enumerate(
        iterproduct(long_thresholds, short_thresholds, rebal_freqs, cost_bps_list), 1
    ):
        if i % 20 == 0:
            print(f"  Sweep {i}/{total}")

        sig = train_signals.copy()
        sig["signal"] = assign_signals(sig["composite_zscore"], lt, st)

        pos  = build_positions(sig, rebal_freq=rf)
        rets = compute_portfolio_returns(pos, sig, base_cost_bps=cb)

        if len(rets) < 100:
            continue

        m = summarize(rets, spy_returns, label=f"lt={lt} st={st} rf={rf} cb={cb}")
        m.update({"long_threshold": lt, "short_threshold": st, "rebal_freq": rf, "cost_bps": cb})
        rows.append(m)

    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ── Printing ──────────────────────────────────────────────────────────────────

def print_summary(metrics: dict):
    print(f"\n{'='*60}")
    print(f"  {metrics['label']}")
    print(f"{'='*60}")
    print(f"  Annualized Return  : {metrics['ann_return']:>8.2%}")
    print(f"  Annualized Vol     : {metrics['ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio       : {metrics['sharpe']:>8.3f}")
    print(f"  Sortino Ratio      : {metrics.get('sortino', float('nan')):>8.3f}")
    print(f"  Max Drawdown       : {metrics['max_dd']:>8.2%}")
    print(f"  Calmar Ratio       : {metrics['calmar']:>8.3f}")
    print(f"  Win Rate           : {metrics['win_rate']:>8.2%}")
    print(f"  Trading Days       : {metrics['n_days']:>8,}")
    if "alpha" in metrics:
        print(f"  ── vs SPY ──────────────────────────")
        print(f"  Alpha (ann.)       : {metrics['alpha']:>8.2%}")
        print(f"  Beta               : {metrics['beta']:>8.3f}")
        print(f"  Information Ratio  : {metrics['info_ratio']:>8.3f}")
        print(f"  Tracking Error     : {metrics['tracking_error']:>8.2%}")


def print_factor_ic(ic_df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  Factor Information Coefficients (IC)")
    print(f"{'='*60}")
    for _, row in ic_df.iterrows():
        if pd.isna(row.ic):
            print(f"  {row.factor:>15}  N/A")
            continue
        bar  = "█" * int(abs(row.ic) * 50)
        sign = "+" if row.ic >= 0 else "-"
        print(f"  {row.factor:>15}  {sign}{abs(row.ic):.4f}  {bar}")


def print_split_banner(label: str, start: str, end: str):
    print(f"\n{'#'*60}")
    print(f"  {label}  |  {start} → {end}")
    print(f"{'#'*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _evaluate_signal_set(
    label:      str,
    signals:    pd.DataFrame,
    spy_rets:   pd.Series,
    train_only: bool,
    test_only:  bool,
) -> dict:
    """
    Run train/test evaluation for a single signal set (factor, RF, or GBM).
    Returns a dict keyed by period label containing summarize() output.
    """
    train, test = split_signals(signals)

    periods = []
    if not test_only:
        periods.append(("TRAIN", train))
    if not train_only:
        periods.append(("TEST  (held-out)", test))

    results = {}

    for period_label, period_signals in periods:
        start = period_signals.index.min().date()
        end   = period_signals.index.max().date()
        print_split_banner(f"{label} — {period_label}", str(start), str(end))

        positions    = build_positions(period_signals)
        returns      = compute_portfolio_returns(positions, period_signals)
        returns_wide = pivot_column(period_signals, "daily_return")
        spy_aligned  = spy_rets.reindex(returns.index)

        metrics = summarize(returns, spy_aligned, label=f"{label} — {period_label}")
        print_summary(metrics)
        results[period_label] = metrics

        long_ret, short_ret = side_returns(positions, returns_wide.reindex(returns.index))
        print_summary(summarize(long_ret,  spy_aligned, label=f"Long Side"))
        print_summary(summarize(short_ret, spy_aligned, label=f"Short Side"))

        ic_df = factor_contribution(period_signals, positions)
        print_factor_ic(ic_df)

        curve = equity_curve(returns).rename("equity")
        tag   = label.lower().replace(" ", "_")
        period_tag = "train" if "TRAIN" in period_label else "test"
        out   = FEATURE_DIR / f"equity_curve_{tag}_{period_tag}.csv"
        curve.to_csv(out)
        print(f"\n  Equity curve saved → {out}")

    return results


def _print_comparison_table(all_results: dict[str, dict]):
    """
    Print a side-by-side comparison across multiple signal sets.
    all_results: { signal_label: { period_label: metrics_dict } }
    """
    keys = ["ann_return", "ann_vol", "sharpe", "max_dd", "calmar",
            "win_rate", "alpha", "beta", "info_ratio"]

    for period in ["TRAIN", "TEST  (held-out)"]:
        period_results = {
            label: res[period]
            for label, res in all_results.items()
            if period in res
        }
        if not period_results:
            continue

        print(f"\n{'='*75}")
        print(f"  COMPARISON — {period}")
        print(f"{'='*75}")

        labels = list(period_results.keys())
        header = f"  {'Metric':<20}" + "".join(f"{l:>16}" for l in labels)
        print(header)
        print(f"  {'-'*70}")

        for k in keys:
            fmt  = ".2%" if k in ("ann_return","ann_vol","max_dd","alpha","tracking_error") else ".3f"
            row  = f"  {k:<20}"
            for l in labels:
                v = period_results[l].get(k, np.nan)
                row += f"{v:>16{fmt}}" if isinstance(v, float) and not np.isnan(v) else f"{'N/A':>16}"
            print(row)


def run(
    train_only: bool = False,
    test_only:  bool = False,
    sweep:      bool = False,
    ml:         str  = "none",    # "none" | "rf" | "gbm" | "both"
):
    print("Loading signals...")
    base_signals = load_signals()
    spy_rets     = load_spy_returns()
    train, _     = split_signals(base_signals)

    print(f"  Full dataset : {base_signals.index.min().date()} → {base_signals.index.max().date()}  "
          f"({base_signals.index.nunique()} dates, {base_signals['ticker'].nunique()} tickers)")

    # ── Parameter sweep (factor signals, train only) ───────────────────────────
    if sweep:
        print("\nRunning parameter sweep on TRAIN period only...")
        sweep_df = parameter_sweep(train, spy_rets)
        out = FEATURE_DIR / "sweep_results_hardened.csv"
        sweep_df.to_csv(out, index=False)
        print(f"\n  Top 5 by Sharpe (train period):")
        print(sweep_df.head(5)[["long_threshold","short_threshold","rebal_freq","cost_bps",
                                 "sharpe","ann_return","max_dd","alpha","beta"]].to_string(index=False))
        print(f"\n  Saved → {out}")
        return

    # ── Build signal sets to evaluate ─────────────────────────────────────────
    signal_sets: dict[str, pd.DataFrame] = {"Factor": base_signals}

    if ml in ("rf", "gbm", "both"):
        ml_data = load_ml_signals()
        if ml_data is None:
            print("\n  Warning: ml_signals.csv not found. Run ml_signals.py first.")
            print("  Falling back to factor signals only.\n")
        else:
            models = ["rf", "gbm"] if ml == "both" else [ml]
            for model in models:
                label = f"ML-{model.upper()}"
                try:
                    signal_sets[label] = inject_ml_signals(base_signals, ml_data, model)
                    print(f"  Loaded {label} signals.")
                except ValueError as e:
                    print(f"  Warning: could not load {label} signals — {e}")

    # ── Evaluate each signal set ───────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for label, signals in signal_sets.items():
        all_results[label] = _evaluate_signal_set(
            label, signals, spy_rets, train_only, test_only
        )

    # ── Cross-signal comparison table ──────────────────────────────────────────
    if len(all_results) > 1 or (len(all_results) == 1 and not train_only and not test_only):
        _print_comparison_table(all_results)

    # ── In-sample vs out-of-sample for each signal set ─────────────────────────
    for label, results in all_results.items():
        if "TRAIN" in results and "TEST  (held-out)" in results:
            print(f"\n{'='*60}")
            print(f"  IN-SAMPLE vs OUT-OF-SAMPLE — {label}")
            print(f"{'='*60}")
            keys = ["ann_return", "ann_vol", "sharpe", "max_dd", "calmar",
                    "win_rate", "alpha", "beta", "info_ratio"]
            train_m = results["TRAIN"]
            test_m  = results["TEST  (held-out)"]
            print(f"  {'Metric':<20} {'Train':>10} {'Test':>10}  {'Δ':>10}")
            print(f"  {'-'*52}")
            for k in keys:
                tv = train_m.get(k, np.nan)
                xv = test_m.get(k,  np.nan)
                if isinstance(tv, float) and isinstance(xv, float) and not (np.isnan(tv) or np.isnan(xv)):
                    delta = xv - tv
                    fmt   = ".2%" if k in ("ann_return","ann_vol","max_dd","alpha","tracking_error") else ".3f"
                    print(f"  {k:<20} {tv:>10{fmt}} {xv:>10{fmt}}  {delta:>+10{fmt}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true", help="Evaluate on training period only")
    parser.add_argument("--test-only",  action="store_true", help="Evaluate on test period only (use sparingly)")
    parser.add_argument("--sweep",      action="store_true", help="Parameter sweep on train period only")
    parser.add_argument("--ml",         default="none",      help="ML signal mode: none | rf | gbm | both")
    args = parser.parse_args()

    run(
        train_only=args.train_only,
        test_only=args.test_only,
        sweep=args.sweep,
        ml=args.ml,
    )