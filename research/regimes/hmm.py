import pandas as pd
import numpy as np
from hmmlearn import hmm
from config import PROCESSED_DIR

def fit_hmm(ticker: str, n_regimes: int = 4) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    df = df.dropna(subset=["log_return", "vol_20"])

    features = np.column_stack([
        df["log_return"].values,
        df["vol_20"].values
    ])

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(features)

    hidden_states = model.predict(features)
    state_probs   = model.predict_proba(features)

    result = df.copy()
    result["hmm_regime"] = hidden_states

    for i in range(n_regimes):
        result[f"hmm_prob_{i}"] = state_probs[:, i]

    # Label regimes by their mean return so they're consistent across tickers
    regime_means = result.groupby("hmm_regime")["log_return"].mean().sort_values()
    label_map = {
        regime_means.index[0]: "bear",
        regime_means.index[1]: "neutral_bear",
        regime_means.index[2]: "neutral_bull",
        regime_means.index[3]: "bull",
    }
    result["hmm_label"] = result["hmm_regime"].map(label_map)

    return result

def summarize_hmm(df: pd.DataFrame, ticker: str):
    print(f"\n{'='*55}")
    print(f"  HMM Regime Summary — {ticker}")
    print(f"{'='*55}")
    summary = df.groupby("hmm_label").agg(
        days       = ("log_return", "count"),
        mean_ret   = ("log_return", "mean"),
        volatility = ("vol_20",     "mean"),
    )
    summary["freq"] = summary["days"] / summary["days"].sum()
    print(summary.to_string())

if __name__ == "__main__":
    for ticker in ["SPY", "QQQ", "AAPL", "NVDA"]:
        df = fit_hmm(ticker)
        summarize_hmm(df, ticker)