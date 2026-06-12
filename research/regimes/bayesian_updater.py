"""
bayesian_updater.py — Phase 10: Bayesian Belief Updating

Replaces the hand-coded step-function likelihoods in bayesian.py with
empirically learned ones derived from the actual return history of each ticker.

Two hypothesis tracks
---------------------
  Track A — Market Regime  (4 states, mutually exclusive)
    p_bull, p_bear, p_high_vol, p_low_vol

  Track B — Forward Outcome  (3 independent hypotheses)
    p_crash       — P(5d return < CRASH_THRESH)
    p_rally       — P(5d return > RALLY_THRESH)
    p_vol_expand  — P(vol_20 rises > VOL_EXPAND_THRESH in 5d)

Composite scores (output columns, range 0–1)
--------------------------------------------
  risk_score        = 0.6 * p_crash + 0.4 * p_high_vol
  opportunity_score = 0.6 * p_rally + 0.4 * p_bull

These two scalars are the primary inputs to position sizing in strategies.

Likelihood learning
-------------------
  For each feature × hypothesis pair, we bin the feature into discrete
  states using rolling quantiles (computed on a trailing window to avoid
  look-ahead bias) and measure:

    P(feature_state | hypothesis_true)

  using only history up to each date. On each new day we update the
  likelihood table with one new observation before using it — strict
  walk-forward, no future leakage.

Features used
-------------
  momentum_20, momentum_60, vol_ratio, rsi_14, rev_zscore_20

Outputs
-------
  data/features/{ticker}_bayes.csv   — per-ticker posteriors time series
  data/features/bayesian_combined.csv — all tickers stacked
  visualizations/output/bayesian_updater.png — 4-panel summary chart
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from config import PROCESSED_DIR, FEATURE_DIR, VISUALIZATION_DIR
from utils.tickers import load_tickers

# ── Output dirs ───────────────────────────────────────────────────────────────

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# ── Hypothesis thresholds ─────────────────────────────────────────────────────

CRASH_THRESH      = -0.04   # 5d return < -4%  → crash event
RALLY_THRESH      =  0.04   # 5d return > +4%  → rally event
VOL_EXPAND_THRESH =  0.15   # vol_20 increases by > 15% in 5d → vol expansion

# ── Regime priors (Track A) ───────────────────────────────────────────────────

REGIME_PRIORS = {
    "bull":     0.40,
    "bear":     0.20,
    "high_vol": 0.20,
    "low_vol":  0.20,
}

# ── Outcome priors (Track B) — base rates from long-run history ───────────────

OUTCOME_PRIORS = {
    "crash":      0.08,   # ~8% of 5d windows are -4% or worse
    "rally":      0.10,   # ~10% of 5d windows are +4% or better
    "vol_expand": 0.20,   # ~20% of 5d windows see vol expand >15%
}

# ── Feature definitions ───────────────────────────────────────────────────────

N_BINS    = 5        # discretize each feature into 5 equal-frequency bins
MIN_OBS   = 252      # minimum history before we start learning likelihoods
SMOOTHING = 0.5      # Laplace smoothing count per bin (avoids zero likelihoods)
PRIOR_BLEND = 0.15   # how much to blend back toward prior each day (prevents lock-in)


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["momentum_20"]   = df["close"].pct_change(20)
    df["momentum_60"]   = df["close"].pct_change(60)
    df["vol_ratio"]     = df["vol_20"] / df["vol_100"].replace(0, np.nan)

    # RSI(14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Mean reversion z-score (not inverted — raw direction)
    mu20  = df["close"].rolling(20).mean()
    sd20  = df["close"].rolling(20).std()
    df["rev_zscore_20"] = (df["close"] - mu20) / sd20.replace(0, np.nan)

    return df


def compute_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward 5d outcome labels. These are FUTURE values — only used
    during likelihood learning (on past data), never leaked into posteriors.
    """
    df = df.copy()
    fwd5 = df["daily_return"].rolling(5).sum().shift(-5)

    df["outcome_crash"]      = (fwd5 < CRASH_THRESH).astype(float)
    df["outcome_rally"]      = (fwd5 > RALLY_THRESH).astype(float)

    vol_fwd = df["vol_20"].shift(-5)
    df["outcome_vol_expand"] = ((vol_fwd - df["vol_20"]) / df["vol_20"].replace(0, np.nan)
                                > VOL_EXPAND_THRESH).astype(float)

    # Regime labels — defined by realized returns and vol over trailing window
    df["regime_bull"]     = ((df["momentum_20"] > 0.02) & (df["vol_ratio"] < 1.1)).astype(float)
    df["regime_bear"]     = ((df["momentum_20"] < -0.02) | (df["momentum_60"] < -0.05)).astype(float)
    df["regime_high_vol"] = (df["vol_ratio"] > 1.3).astype(float)
    df["regime_low_vol"]  = (df["vol_ratio"] < 0.85).astype(float)

    return df


# ── Likelihood table ──────────────────────────────────────────────────────────

FEATURE_COLS = ["momentum_20", "momentum_60", "vol_ratio", "rsi_14", "rev_zscore_20"]

HYPOTHESES_A = list(REGIME_PRIORS.keys())   # regime track
HYPOTHESES_B = list(OUTCOME_PRIORS.keys())  # outcome track

ALL_HYPOTHESES = HYPOTHESES_A + HYPOTHESES_B

HYPOTHESIS_OUTCOME_MAP = {
    "bull":       "regime_bull",
    "bear":       "regime_bear",
    "high_vol":   "regime_high_vol",
    "low_vol":    "regime_low_vol",
    "crash":      "outcome_crash",
    "rally":      "outcome_rally",
    "vol_expand": "outcome_vol_expand",
}


class LikelihoodTable:
    """
    Maintains a running count table:
      counts[hypothesis][feature][bin] = (n_times_feature_was_in_bin_when_hypothesis_was_true,
                                          n_times_feature_was_in_bin_total)

    On each new observation we update counts, then compute:
      P(feature_bin | hypothesis) = (count_true + smoothing) / (count_total + smoothing * N_BINS)
    """

    def __init__(self):
        # counts[hyp][feat] = np.array of shape (N_BINS, 2)  col0=true, col1=total
        self.counts = {
            h: {f: np.full((N_BINS, 2), SMOOTHING) for f in FEATURE_COLS}
            for h in ALL_HYPOTHESES
        }
        # Running quantile boundaries per feature (updated as we see more data)
        self.boundaries = {f: None for f in FEATURE_COLS}
        self.history    = {f: [] for f in FEATURE_COLS}

    def _bin(self, feature: str, value: float) -> int | None:
        """Discretize a feature value into 0..N_BINS-1 using current boundaries."""
        if np.isnan(value) or self.boundaries[feature] is None:
            return None
        b = self.boundaries[feature]
        idx = np.searchsorted(b, value, side="right") - 1
        return int(np.clip(idx, 0, N_BINS - 1))

    def update_boundaries(self, feature: str, value: float):
        if not np.isnan(value):
            self.history[feature].append(value)
        if len(self.history[feature]) >= MIN_OBS:
            quantiles = np.linspace(0, 100, N_BINS + 1)[1:-1]
            self.boundaries[feature] = np.percentile(self.history[feature], quantiles)

    def observe(self, row: pd.Series, outcomes: dict[str, float]):
        """
        Record one new observation.
        outcomes: dict mapping hypothesis name → 1.0 (true) or 0.0 (false)
        """
        for feat in FEATURE_COLS:
            val = row.get(feat, np.nan)
            self.update_boundaries(feat, val)
            b = self._bin(feat, val)
            if b is None:
                continue
            for hyp in ALL_HYPOTHESES:
                self.counts[hyp][feat][b, 1] += 1   # total
                if outcomes.get(hyp, 0.0) == 1.0:
                    self.counts[hyp][feat][b, 0] += 1  # true

    def likelihood(self, hyp: str, row: pd.Series) -> float:
        """
        P(evidence | hypothesis) — product of per-feature likelihoods.
        Features with no bin assignment are skipped (treated as uninformative).
        """
        p = 1.0
        n_used = 0
        for feat in FEATURE_COLS:
            val = row.get(feat, np.nan)
            b   = self._bin(feat, val)
            if b is None:
                continue
            c_true  = self.counts[hyp][feat][b, 0]
            c_total = self.counts[hyp][feat][b, 1]
            p *= c_true / c_total
            n_used += 1
        # Geometric mean to prevent underflow with many features
        return p ** (1.0 / n_used) if n_used > 0 else 1.0


# ── Sequential Bayesian updater ───────────────────────────────────────────────

def run_bayesian_update(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk forward through the DataFrame one row at a time.

    For each date t:
      1. Update likelihood table with observation at t-1 (uses outcome at t+4,
         so we only call observe() once that outcome is realized — enforced by
         a 5-day lag buffer).
      2. Compute posteriors using current likelihoods and priors.
      3. Store posteriors.
      4. Blend priors toward posterior for next step.
    """
    df = df.copy()
    table = LikelihoodTable()

    # Separate priors for each track
    priors_a = np.array([REGIME_PRIORS[h]  for h in HYPOTHESES_A])
    priors_b = np.array([OUTCOME_PRIORS[h] for h in HYPOTHESES_B])

    base_a = priors_a.copy()
    base_b = priors_b.copy()

    # Output storage
    records = {h: [] for h in ALL_HYPOTHESES}
    indices = []

    # We need outcomes which require 5 future days — use a lag buffer
    # so we only learn from outcomes that have fully materialized
    obs_buffer = []   # (row, outcomes_dict) pairs waiting 5 days to be learned

    rows = list(df.iterrows())

    for i, (date, row) in enumerate(rows):
        # ── Flush any observations whose 5d outcome is now realized ──────────
        if len(obs_buffer) >= 5:
            old_row, old_outcomes = obs_buffer.pop(0)
            table.observe(old_row, old_outcomes)

        # ── Compute posteriors with current table ─────────────────────────────
        # Track A: regime (mutually exclusive → normalize)
        likes_a = np.array([table.likelihood(h, row) for h in HYPOTHESES_A])
        post_a  = priors_a * likes_a
        total_a = post_a.sum()
        post_a  = post_a / total_a if total_a > 0 else priors_a

        # Track B: outcomes (independent → each normalized separately)
        post_b = []
        for j, h in enumerate(HYPOTHESES_B):
            lk  = table.likelihood(h, row)
            p   = priors_b[j] * lk
            # Two-hypothesis normalization: P(h|e) = P(e|h)*P(h) / [P(e|h)*P(h) + P(e|¬h)*P(¬h)]
            # Approximate P(e|¬h) = 1 - lk (uninformative complement)
            complement = (1 - priors_b[j]) * max(1 - lk, 1e-6)
            post_b.append(p / (p + complement) if (p + complement) > 0 else priors_b[j])
        post_b = np.array(post_b)

        # Store
        for k, h in enumerate(HYPOTHESES_A):
            records[h].append(float(post_a[k]))
        for k, h in enumerate(HYPOTHESES_B):
            records[h].append(float(post_b[k]))
        indices.append(date)

        # ── Buffer this observation with its outcome labels ───────────────────
        outcome_cols = {
            h: float(row.get(HYPOTHESIS_OUTCOME_MAP[h], np.nan))
            for h in ALL_HYPOTHESES
        }
        obs_buffer.append((row, outcome_cols))

        # ── Update priors for next step ───────────────────────────────────────
        priors_a = (1 - PRIOR_BLEND) * post_a + PRIOR_BLEND * base_a
        priors_b = (1 - PRIOR_BLEND) * post_b + PRIOR_BLEND * base_b

    # ── Assemble result DataFrame ─────────────────────────────────────────────
    result = pd.DataFrame(records, index=indices)
    result.index.name = "date"

    # Rename for clarity
    result = result.rename(columns={
        "bull":       "p_bull",
        "bear":       "p_bear",
        "high_vol":   "p_high_vol",
        "low_vol":    "p_low_vol",
        "crash":      "p_crash",
        "rally":      "p_rally",
        "vol_expand": "p_vol_expand",
    })

    # Composite scores
    result["risk_score"]        = 0.6 * result["p_crash"]  + 0.4 * result["p_high_vol"]
    result["opportunity_score"] = 0.6 * result["p_rally"]  + 0.4 * result["p_bull"]

    # Dominant regime label
    regime_cols = ["p_bull", "p_bear", "p_high_vol", "p_low_vol"]
    result["dominant_regime"] = result[regime_cols].idxmax(axis=1).str.replace("p_", "")

    # Merge back base price columns for context
    base_cols = ["close", "daily_return", "log_return", "vol_20"]
    available = [c for c in base_cols if c in df.columns]
    result = result.join(df[available], how="left")

    return result


# ── Per-ticker pipeline ───────────────────────────────────────────────────────

def build_posteriors(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    df = compute_features(df)
    df = compute_outcomes(df)
    df = df.dropna(subset=["momentum_20", "vol_ratio"])

    result = run_bayesian_update(df)
    return result


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_ticker(ticker: str, df: pd.DataFrame, ax_risk, ax_opp, ax_regime, ax_returns):
    """Fill four axes for one ticker."""
    color_map = {
        "bull":     "#2ecc71",
        "bear":     "#e74c3c",
        "high_vol": "#e67e22",
        "low_vol":  "#3498db",
    }

    # Risk & opportunity scores
    ax_risk.plot(df.index, df["risk_score"],        color="#e74c3c", lw=1, label="risk")
    ax_risk.plot(df.index, df["opportunity_score"], color="#2ecc71", lw=1, label="opportunity")
    ax_risk.axhline(0.5, lw=0.5, linestyle="--", color="gray")
    ax_risk.set_title(f"{ticker} — Risk / Opportunity Score", fontsize=9, fontweight="bold")
    ax_risk.set_ylabel("Posterior probability")
    ax_risk.legend(fontsize=7)
    ax_risk.set_ylim(0, 1)

    # Regime posteriors stacked
    regime_cols = ["p_bull", "p_bear", "p_high_vol", "p_low_vol"]
    labels      = ["bull", "bear", "high_vol", "low_vol"]
    bottoms     = np.zeros(len(df))
    for col, lbl in zip(regime_cols, labels):
        vals = df[col].fillna(0).values
        ax_opp.bar(df.index, vals, bottom=bottoms, label=lbl,
                   color=color_map[lbl], alpha=0.7, width=1)
        bottoms += vals
    ax_opp.set_title(f"{ticker} — Regime Posteriors", fontsize=9, fontweight="bold")
    ax_opp.set_ylabel("Probability")
    ax_opp.legend(fontsize=6, loc="upper right")
    ax_opp.set_ylim(0, 1)

    # Dominant regime timeline
    regime_num = df["dominant_regime"].map(
        {"bull": 3, "low_vol": 2, "high_vol": 1, "bear": 0}
    ).fillna(2)
    scatter_colors = [color_map.get(r, "gray") for r in df["dominant_regime"].fillna("low_vol")]
    ax_regime.scatter(df.index, regime_num, c=scatter_colors, s=1, alpha=0.6)
    ax_regime.set_yticks([0, 1, 2, 3])
    ax_regime.set_yticklabels(["bear", "high_vol", "low_vol", "bull"], fontsize=7)
    ax_regime.set_title(f"{ticker} — Dominant Regime", fontsize=9, fontweight="bold")

    # Price + risk overlay
    if "close" in df.columns:
        ax2 = ax_returns.twinx()
        ax_returns.plot(df.index, df["close"], color="black", lw=0.8, alpha=0.6, label="price")
        ax2.fill_between(df.index, df["risk_score"], alpha=0.15, color="#e74c3c", label="risk")
        ax2.fill_between(df.index, df["opportunity_score"], alpha=0.15, color="#2ecc71", label="opp")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Score", fontsize=7)
        ax_returns.set_title(f"{ticker} — Price + Risk Overlay", fontsize=9, fontweight="bold")
        ax_returns.set_ylabel("Price")


def plot_all(results: dict[str, pd.DataFrame]):
    tickers = list(results.keys())
    n = len(tickers)

    fig = plt.figure(figsize=(20, 5 * n))
    fig.suptitle("Phase 10 — Bayesian Belief Updating", fontsize=14, fontweight="bold", y=1.001)

    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.6)

    for i, ticker in enumerate(tickers):
        df  = results[ticker]
        inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[i], wspace=0.35)
        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1])
        ax3 = fig.add_subplot(inner[2])
        ax4 = fig.add_subplot(inner[3])
        plot_ticker(ticker, df, ax1, ax2, ax3, ax4)

    out = VISUALIZATION_DIR / "bayesian_updater.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved → {out}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(ticker: str, df: pd.DataFrame):
    latest = df.iloc[-1]
    print(f"\n  {ticker}")
    print(f"    Dominant regime  : {latest['dominant_regime']}")
    print(f"    p_bull           : {latest['p_bull']:.3f}")
    print(f"    p_bear           : {latest['p_bear']:.3f}")
    print(f"    p_high_vol       : {latest['p_high_vol']:.3f}")
    print(f"    p_low_vol        : {latest['p_low_vol']:.3f}")
    print(f"    p_crash          : {latest['p_crash']:.3f}")
    print(f"    p_rally          : {latest['p_rally']:.3f}")
    print(f"    p_vol_expand     : {latest['p_vol_expand']:.3f}")
    print(f"    risk_score       : {latest['risk_score']:.3f}")
    print(f"    opportunity_score: {latest['opportunity_score']:.3f}")

    regime_counts = df["dominant_regime"].value_counts(normalize=True)
    print(f"    Regime breakdown :")
    for regime, pct in regime_counts.items():
        print(f"      {regime:<10} {pct:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tickers = load_tickers()
    results = {}

    print(f"{'='*60}")
    print("  Phase 10 — Bayesian Belief Updating")
    print(f"{'='*60}")

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            df = build_posteriors(ticker)
            results[ticker] = df

            # Save per-ticker CSV
            out = FEATURE_DIR / f"{ticker}_bayes.csv"
            df.to_csv(out)
            print(f"  Rows: {len(df)}  |  Saved → {out}")

        except FileNotFoundError:
            print(f"  Skipping {ticker}: no processed data found")
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            raise

    if not results:
        print("No results — check that processed data exists.")
        return

    # Save combined CSV
    frames = []
    for ticker, df in results.items():
        df_copy = df.copy()
        df_copy["ticker"] = ticker
        frames.append(df_copy)
    combined = pd.concat(frames)
    combined.index.name = "date"
    combined_out = FEATURE_DIR / "bayesian_combined.csv"
    combined.to_csv(combined_out)
    print(f"\n  Combined CSV saved → {combined_out}")

    # Print latest posteriors
    print(f"\n{'='*60}")
    print("  LATEST POSTERIORS")
    print(f"{'='*60}")
    for ticker, df in results.items():
        print_summary(ticker, df)

    # Visualize
    print(f"\nGenerating visualization...")
    plot_all(results)

    print("\nDone.")


if __name__ == "__main__":
    main()