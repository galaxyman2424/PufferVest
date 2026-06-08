import pandas as pd
from itertools import product

from config import (
    PROCESSED_DIR,
    FEATURE_DIR,
    LOOKBACKS,
    LOOKAHEADS,
)

from utils.tickers import load_tickers

def load(ticker: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)

def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lb in LOOKBACKS:
        df[f"mom_{lb}"] = df["close"].pct_change(lb)
    for la in LOOKAHEADS:
        df[f"fwd_{la}"] = df["close"].pct_change(la).shift(-la)
    return df.dropna()

def analyze_momentum(ticker: str) -> pd.DataFrame:
    df = load(ticker)
    df = compute_momentum(df)

    rows = []
    for lb, la in product(LOOKBACKS, LOOKAHEADS):
        mom_col = f"mom_{lb}"
        fwd_col = f"fwd_{la}"

        long_mask  = df[mom_col] > 0
        short_mask = df[mom_col] < 0

        long_ret  = df.loc[long_mask,  fwd_col].mean()
        short_ret = df.loc[short_mask, fwd_col].mean()
        long_wr   = (df.loc[long_mask,  fwd_col] > 0).mean()
        short_wr  = (df.loc[short_mask, fwd_col] > 0).mean()
        spread    = long_ret - short_ret

        rows.append({
            "ticker":    ticker,
            "lookback":  lb,
            "lookahead": la,
            "long_ret":  round(long_ret,  5),
            "short_ret": round(short_ret, 5),
            "spread":    round(spread,    5),
            "long_wr":   round(long_wr,   4),
            "short_wr":  round(short_wr,  4),
        })

    return pd.DataFrame(rows)

def print_results(results: pd.DataFrame, ticker: str):
    df = results[results["ticker"] == ticker]
    print(f"\n{'='*75}")
    print(f"  Momentum Analysis — {ticker}")
    print(f"{'='*75}")
    print(f"  {'LB':>4}  {'LA':>4}  {'Long Ret':>10}  {'Short Ret':>10}  {'Spread':>10}  {'Long WR':>8}  {'Short WR':>9}")
    print(f"  {'-'*70}")
    for _, row in df.iterrows():
        print(f"  {int(row.lookback):>4}  {int(row.lookahead):>4}  {row.long_ret:>10.5f}  {row.short_ret:>10.5f}  {row.spread:>10.5f}  {row.long_wr:>8.2%}  {row.short_wr:>9.2%}")

def main():
    tickers = load_tickers()

    all_results = []

    for ticker in tickers:
        results = analyze_momentum(ticker)
        all_results.append(results)
        print_results(results, ticker)

    combined = pd.concat(all_results)

    out = FEATURE_DIR / "momentum.csv"

    out.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(out, index=False)

if __name__ == "__main__":
    main()