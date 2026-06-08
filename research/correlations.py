import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("visualizations/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]

def load_returns(tickers: list) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        df = pd.read_csv(PROCESSED_DIR / f"{t}.csv", index_col=0, parse_dates=True)
        frames[t] = df["daily_return"]
    return pd.DataFrame(frames).dropna()

def plot_correlation_matrix(returns: pd.DataFrame):
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — Daily Returns (2000-Present)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → correlation_matrix.png")

def plot_covariance_matrix(returns: pd.DataFrame):
    cov = returns.cov() * 252
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cov, annot=True, fmt=".4f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Annualized Covariance Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "covariance_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → covariance_matrix.png")

def plot_rolling_correlations(returns: pd.DataFrame, window: int = 60):
    spy = returns["SPY"]
    tickers = [t for t in TICKERS if t != "SPY"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"Rolling {window}-Day Correlation vs SPY", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        rolling_corr = returns[ticker].rolling(window).corr(spy)
        axes[i].plot(rolling_corr.index, rolling_corr, lw=1, color="steelblue")
        axes[i].axhline(returns[ticker].corr(spy), color="red", lw=1, linestyle="--", label="Full period avg")
        axes[i].axhline(0, color="black", lw=0.5, linestyle=":")
        axes[i].set_title(ticker)
        axes[i].set_ylim(-1, 1)
        axes[i].set_ylabel("Correlation")
        axes[i].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → rolling_correlations.png")

def print_summary(returns: pd.DataFrame):
    corr = returns.corr()
    print("\nTop correlations with SPY:")
    spy_corr = corr["SPY"].drop("SPY").sort_values(ascending=False)
    for ticker, val in spy_corr.items():
        print(f"  {ticker:6} {val:.4f}")

def main():
    print("Loading returns...")
    returns = load_returns(TICKERS)
    print(f"  {len(returns)} overlapping trading days across all tickers")

    plot_correlation_matrix(returns)
    plot_covariance_matrix(returns)
    plot_rolling_correlations(returns)
    print_summary(returns)

if __name__ == "__main__":
    main()