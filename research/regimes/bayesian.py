# Replace the top of bayesian.py
from pathlib import Path
import pandas as pd
import numpy as np
from config import PROCESSED_DIR

# Prior probabilities for each regime
PRIORS = {
    "bull":         0.40,
    "bear":         0.20,
    "high_vol":     0.20,
    "low_vol":      0.20,
}

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["momentum_20"]  = df["close"].pct_change(20)
    df["momentum_60"]  = df["close"].pct_change(60)
    df["vol_ratio"]    = df["vol_20"] / df["vol_100"]
    return df

def likelihood(row: pd.Series, regime: str) -> float:
    m20  = row["momentum_20"]
    m60  = row["momentum_60"]
    vr   = row["vol_ratio"]

    if regime == "bull":
        p_m20 = 1.0 if m20 > 0.02  else (0.5 if m20 > 0    else 0.1)
        p_m60 = 1.0 if m60 > 0.05  else (0.5 if m60 > 0    else 0.1)
        p_vr  = 1.0 if vr  < 0.9   else (0.5 if vr  < 1.1  else 0.2)
    elif regime == "bear":
        p_m20 = 1.0 if m20 < -0.02 else (0.5 if m20 < 0    else 0.1)
        p_m60 = 1.0 if m60 < -0.05 else (0.5 if m60 < 0    else 0.1)
        p_vr  = 1.0 if vr  > 1.2   else (0.5 if vr  > 1.0  else 0.2)
    elif regime == "high_vol":
        p_m20 = 0.5
        p_m60 = 0.5
        p_vr  = 1.0 if vr  > 1.3   else (0.5 if vr  > 1.1  else 0.1)
    else:  # low_vol
        p_m20 = 0.5
        p_m60 = 0.5
        p_vr  = 1.0 if vr  < 0.8   else (0.5 if vr  < 1.0  else 0.1)

    return p_m20 * p_m60 * p_vr

def bayesian_regimes(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    df = compute_features(df).dropna()

    regimes = list(PRIORS.keys())
    prob_cols = {r: [] for r in regimes}

    priors = np.array([PRIORS[r] for r in regimes])

    for _, row in df.iterrows():
        likelihoods = np.array([likelihood(row, r) for r in regimes])
        posterior   = priors * likelihoods
        total       = posterior.sum()
        posterior   = posterior / total if total > 0 else priors

        for i, r in enumerate(regimes):
            prob_cols[r].append(posterior[i])

        # Update priors with smoothing to avoid getting stuck
        priors = 0.85 * posterior + 0.15 * np.array([PRIORS[r] for r in regimes])

    for r in regimes:
        df[f"bayes_prob_{r}"] = prob_cols[r]

    df["bayes_regime"] = df[[f"bayes_prob_{r}" for r in regimes]].idxmax(axis=1)
    df["bayes_regime"] = df["bayes_regime"].str.replace("bayes_prob_", "")

    return df

def summarize_bayesian(df: pd.DataFrame, ticker: str):
    print(f"\n{'='*55}")
    print(f"  Bayesian Regime Summary — {ticker}")
    print(f"{'='*55}")
    summary = df.groupby("bayes_regime").agg(
        days     = ("log_return", "count"),
        mean_ret = ("log_return", "mean"),
        mean_vol = ("vol_20",     "mean"),
    )
    summary["freq"] = summary["days"] / summary["days"].sum()
    print(summary.to_string())

if __name__ == "__main__":
    for ticker in ["SPY", "QQQ", "AAPL", "NVDA"]:
        df = bayesian_regimes(ticker)
        summarize_bayesian(df, ticker)