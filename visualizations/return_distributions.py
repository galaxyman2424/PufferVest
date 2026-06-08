import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("visualizations/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]

def load_returns(ticker: str) -> pd.Series:
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    return df["daily_return"].dropna()

def plot_distribution(ticker: str, returns: pd.Series):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{ticker} — Return Distribution Analysis", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    mu, sigma = returns.mean(), returns.std()
    nu, loc, scale = stats.t.fit(returns)
    x = np.linspace(returns.min(), returns.max(), 500)

    # Histogram with overlays
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(returns, bins=150, density=True, alpha=0.5, color="steelblue", label="Actual")
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal")
    ax1.plot(x, stats.t.pdf(x, nu, loc, scale), "g-", lw=2, label=f"Student-t (ν={nu:.1f})")
    ax1.set_xlim(-0.15, 0.15)
    ax1.set_title("Return Distribution vs Normal vs Student-t")
    ax1.set_xlabel("Daily Return")
    ax1.set_ylabel("Density")
    ax1.legend()

    # QQ plot
    ax2 = fig.add_subplot(gs[1, 0])
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("QQ Plot vs Normal")

    # Rolling volatility
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df.index, df["vol_20"],  lw=1,   label="20d")
    ax3.plot(df.index, df["vol_50"],  lw=1.5, label="50d")
    ax3.plot(df.index, df["vol_100"], lw=2,   label="100d")
    ax3.set_title("Rolling Annualized Volatility")
    ax3.set_ylabel("Volatility")
    ax3.legend()

    out = OUT_DIR / f"{ticker}_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

def main():
    for ticker in TICKERS:
        print(f"Plotting {ticker}...")
        returns = load_returns(ticker)
        plot_distribution(ticker, returns)

if __name__ == "__main__":
    main()