import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]

def load_returns(ticker: str) -> pd.Series:
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    return df["daily_return"].dropna()

def analyze_tails(ticker: str, returns: pd.Series):
    mu, sigma = returns.mean(), returns.std()
    n = len(returns)
    z_scores = (returns - mu) / sigma

    print(f"\n{'='*80}")
    print(f"  {ticker}  |  n={n}  |  μ={mu:.5f}  |  σ={sigma:.5f}")
    print(f"{'='*80}")
    print(f"  {'k':>4}  {'Obs Both':>9}  {'Obs Down':>9}  {'Obs Up':>9}  {'Normal':>9}  {'Chebyshev':>10}  {'Cantelli':>9}  {'Ratio':>7}")
    print(f"  {'-'*75}")

    for k in [1, 2, 3, 4]:
        obs_both  = (np.abs(z_scores) >= k).mean()
        obs_down  = (z_scores <= -k).mean()
        obs_up    = (z_scores >= k).mean()
        normal    = 2 * (1 - stats.norm.cdf(k))
        chebyshev = 1 / k**2
        cantelli  = 1 / (1 + k**2)
        ratio     = obs_both / normal if normal > 0 else float("inf")

        print(f"  {k:>4}σ  {obs_both:>9.4%}  {obs_down:>9.4%}  {obs_up:>9.4%}  {normal:>9.4%}  {chebyshev:>10.4%}  {cantelli:>9.4%}  {ratio:>6.2f}x")

def main():
    for ticker in TICKERS:
        returns = load_returns(ticker)
        analyze_tails(ticker, returns)

    print(f"\n{'='*80}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*80}")
    print("  Obs Down / Obs Up  — asymmetry between downside and upside tails")
    print("  Chebyshev          — upper bound for any distribution (both tails)")
    print("  Cantelli           — upper bound for one tail of any distribution")
    print("  Ratio              — Observed / Normal, how much fatter than Gaussian")

if __name__ == "__main__":
    main()