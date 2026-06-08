import pandas as pd
import numpy as np
from hmm import fit_hmm
from bayesian import bayesian_regimes

POSITION_MULTIPLIERS = {
    # HMM labels
    "bull":         1.0,
    "neutral_bull": 0.75,
    "neutral_bear": 0.25,
    "bear":         0.0,
    # Bayesian labels
    "low_vol":      1.0,
    "high_vol":     0.5,
}

def detect(ticker: str, n_hmm_regimes: int = 4) -> pd.DataFrame:
    hmm_df   = fit_hmm(ticker, n_regimes=n_hmm_regimes)
    bayes_df = bayesian_regimes(ticker)

    cols_hmm   = ["hmm_regime", "hmm_label"] + [c for c in hmm_df.columns   if c.startswith("hmm_prob")]
    cols_bayes = ["bayes_regime"]             + [c for c in bayes_df.columns if c.startswith("bayes_prob")]

    df = hmm_df.join(bayes_df[cols_bayes], how="inner")

    df["hmm_size"]   = df["hmm_label"].map(POSITION_MULTIPLIERS).fillna(0.5)
    df["bayes_size"] = df["bayes_regime"].map(POSITION_MULTIPLIERS).fillna(0.5)
    df["regime_size"] = (df["hmm_size"] + df["bayes_size"]) / 2

    return df

if __name__ == "__main__":
    for ticker in ["SPY", "QQQ", "AAPL", "NVDA"]:
        print(f"\nRunning detector for {ticker}...")
        df = detect(ticker)
        print(df[["log_return", "hmm_label", "bayes_regime", "regime_size"]].tail(10).to_string())