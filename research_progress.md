# Quantitative Market Research — Progress Log

## Environment

- **OS**: Linux Mint
- **IDE**: VSCode
- **Python**: 3.12
- **Venv**: `venv/` in project root
- **Activated via**: `source venv/bin/activate`

### Dependencies Installed

```
pandas, numpy, scipy, matplotlib, seaborn, statsmodels, scikit-learn, yfinance, hmmlearn
```

---

## Project Structure

```
quant-research/
├── data/
│   ├── raw/              # Downloaded CSVs from yfinance
│   ├── processed/        # Returns, log returns, rolling vol
│   └── features/         # Momentum signals, regime labels
├── research/
│   ├── __init__.py
│   ├── tail_analysis.py
│   ├── correlations.py
│   ├── momentum.py
│   └── regimes/
│       ├── __init__.py
│       ├── hmm.py
│       ├── bayesian.py
│       └── detector.py
└── visualizations/
    ├── return_distributions.py
    └── output/           # PNG outputs
```

---

## Phase 1 — Data Collection

**Script**: `data/collect.py`

**Tickers downloaded**:

- ETFs: SPY, QQQ, DIA, IWM
- Stocks: AAPL, MSFT, NVDA, AMZN, META, GOOGL
- Benchmarks: IXIC (Nasdaq Composite), TNX (10-Year Treasury Yield)

**Date range**: 2000-01-01 to 2025-01-01

**Notes**:
- Used `yfinance` with `auto_adjust=True`
- Recent yfinance versions return a MultiIndex column structure — fixed by calling `df.columns.get_level_values(0)` before lowercasing
- `^IXIC` and `^TNX` saved with `^` stripped from filename to avoid Linux path issues
- Row counts vary by ticker due to IPO dates (META: 2012, GOOGL: 2004)
- TNX downloaded for use as risk-free rate in Sharpe ratio calculations

---

## Phase 2 — Exploratory Data Analysis

**Script**: `data/process.py`

**Computed per ticker**:

- `daily_return` = `pct_change(close)`
- `log_return` = `ln(close_t / close_t-1)`
- `vol_20`, `vol_50`, `vol_100` = rolling annualized volatility (std * sqrt(252))

**Key observations from summary statistics**:

| Ticker | Mean Daily Return | Std Dev | Skewness | Kurtosis |
|--------|------------------|---------|----------|----------|
| SPY    | 0.00037          | 0.01221 | -0.0167  | 11.60    |
| QQQ    | 0.00044          | 0.01701 | 0.2100   | 7.31     |
| DIA    | 0.00037          | 0.01171 | 0.0967   | 15.38    |
| IWM    | 0.00042          | 0.01512 | -0.2790  | 5.37     |
| AAPL   | 0.00122          | 0.02440 | -1.4693  | 36.66    |
| MSFT   | 0.00057          | 0.01903 | 0.1624   | 9.57     |
| NVDA   | 0.00186          | 0.03741 | 0.5849   | 12.08    |
| AMZN   | 0.00109          | 0.03109 | 1.1283   | 15.87    |
| META   | 0.00118          | 0.02510 | 0.4021   | 21.12    |
| GOOGL  | 0.00103          | 0.01927 | 0.6172   | 9.15     |

**Visualizations**: `visualizations/return_distributions.py`

Produces one PNG per ticker containing:
- Histogram overlaid with Normal and Student-t fit
- QQ plot vs Normal distribution
- Rolling annualized volatility (20/50/100 day)

**Key finding**: QQ plots show strong tail deviation — both ends curl sharply away from the normal line, visually confirming excess kurtosis across all tickers.

---

## Phase 3 — Distribution Research

**Conclusion**: Markets are not Gaussian. Every ticker shows significant excess kurtosis (all well above 0). AAPL's kurtosis of 36.7 is extreme. Skewness is mixed — indices cluster near zero while individual stocks show more asymmetry (AAPL strongly negative, AMZN strongly positive).

---

## Phase 4 — Chebyshev & Tail Analysis

**Script**: `research/tail_analysis.py`

**Methodology**: For each ticker, compute z-scores and measure observed frequency of moves at 1σ, 2σ, 3σ, 4σ. Compare against:

- Normal distribution prediction
- Chebyshev bound (both tails): `1/k²`
- Cantelli bound (one tail): `1/(1+k²)`

Split observed frequency into upside and downside tails separately to expose asymmetry.

**Key findings**:

- At **1σ**: All tickers observe *fewer* moves than normal predicts (~20% vs 31.7%). Returns cluster near the mean more tightly than Gaussian.
- At **2σ**: Near parity with normal (~1.0–1.2x ratio).
- At **3σ**: 4.7x–6.9x more frequent than normal predicts.
- At **4σ**: 80x–140x more frequent than normal predicts.

**Notable extremes**:
- AMZN 4σ ratio: **140.6x**
- QQQ 4σ ratio: **108x**
- AAPL 4σ ratio: **80x**

**Implication**: Any risk model assuming normality (e.g. standard VaR) will catastrophically underestimate tail risk. This is the empirical foundation for fat-tail finance.

**Chebyshev result**: The inequality holds as a valid upper bound for all observations but is far too loose to be practically useful in financial markets. Cantelli's one-sided bound is tighter and more appropriate when asymmetric tail risk matters.

---

## Phase 5 — Correlation & Covariance

**Script**: `research/correlations.py`

**Outputs**:
- Static correlation matrix heatmap
- Annualized covariance matrix heatmap
- Rolling 60-day correlation vs SPY for each ticker

**Note**: Overlapping data starts at META's IPO (2012), giving 3174 trading days for the full cross-ticker analysis. The 2008 crash is excluded from full-matrix analysis as a result.

**SPY correlations (full period)**:

| Ticker | Correlation |
|--------|-------------|
| DIA    | 0.9542      |
| QQQ    | 0.9274      |
| IWM    | 0.8676      |
| MSFT   | 0.7625      |
| GOOGL  | 0.7008      |
| AAPL   | 0.6944      |
| NVDA   | 0.6249      |
| AMZN   | 0.6112      |
| META   | 0.5215      |

**Key finding**: During the 2020 COVID crash, rolling correlations spike sharply toward 1.0 across all assets simultaneously — confirming the well-known phenomenon that diversification fails exactly when it is needed most. Static correlation matrices used in mean-variance optimization do not capture this regime-dependent behavior.

---

## Phase 6 — Market Regime Detection

**Scripts**: `research/regimes/hmm.py`, `research/regimes/bayesian.py`, `research/regimes/detector.py`

### HMM (Hidden Markov Model)

- Library: `hmmlearn`
- Features: `log_return`, `vol_20`
- 4 regimes fit per ticker, labeled by mean return: `bear`, `neutral_bear`, `neutral_bull`, `bull`
- Outputs: regime label + probability per regime per day

**SPY HMM regime breakdown**:

| Regime       | Days | Mean Return | Avg Vol | Frequency |
|--------------|------|-------------|---------|-----------|
| bear         | 1446 | -0.000076   | 0.212   | 23.1%     |
| neutral_bear | 671  | -0.000068   | 0.386   | 10.7%     |
| neutral_bull | 1686 | 0.000263    | 0.143   | 26.9%     |
| bull         | 2466 | 0.000648    | 0.088   | 39.3%     |

**Note**: NVDA's "bear" regime has a positive mean return — for high-volatility individual stocks the HMM clusters by volatility more than direction. Trust regime probabilities over labels for these tickers.

### Bayesian State Estimation

- Features: `momentum_20`, `momentum_60`, `vol_ratio` (vol_20 / vol_100)
- 4 regimes: `bull`, `bear`, `high_vol`, `low_vol`
- Likelihood hand-coded per regime based on feature thresholds
- Priors updated each day with 85/15 smoothing to avoid getting stuck

### Unified Detector

`detector.py` provides a single `detect(ticker)` function that returns a DataFrame with:
- HMM regime label and probabilities
- Bayesian regime label and probabilities
- `hmm_size` — position multiplier from HMM (0.0 bear → 1.0 bull)
- `bayes_size` — position multiplier from Bayesian
- `regime_size` — average of both, used as position sizing input for strategies

**Position multipliers**:

| Regime       | Multiplier |
|--------------|------------|
| bull         | 1.00       |
| neutral_bull | 0.75       |
| low_vol      | 1.00       |
| neutral_bear | 0.25       |
| high_vol     | 0.50       |
| bear         | 0.00       |

---

## Phase 7 — Momentum Research

**Script**: `research/momentum.py`

**Methodology**: For each ticker, compute past returns over lookback windows of 1, 5, 20, 60 days. Split into long (positive momentum) and short (negative momentum) groups. Measure forward returns over lookahead windows of 1, 5, 20, 60 days. Compute spread (long_ret - short_ret) and win rate per group.

Results saved to `data/features/momentum.csv`.

**Top 10 momentum signals by spread**:

| Ticker | Lookback | Lookahead | Spread  | Long WR |
|--------|----------|-----------|---------|---------|
| NVDA   | 20       | 60        | 0.03866 | 68.86%  |
| NVDA   | 60       | 60        | 0.03163 | 67.47%  |
| NVDA   | 60       | 20        | 0.02310 | 60.95%  |
| NVDA   | 5        | 60        | 0.02132 | 67.22%  |
| META   | 20       | 20        | 0.02120 | 66.53%  |
| NVDA   | 20       | 20        | 0.01362 | 60.84%  |
| QQQ    | 60       | 60        | 0.01264 | 70.05%  |
| AAPL   | 5        | 60        | 0.01029 | 69.37%  |
| NVDA   | 5        | 20        | 0.00971 | 60.11%  |
| NVDA   | 1        | 60        | 0.00946 | 66.39%  |

**Key findings**:

- **NVDA dominates** momentum signals across almost every lookback/lookahead combination. The 20-day lookback / 60-day lookahead spread of 3.87% is the strongest signal found.
- **Indices (SPY, DIA, IWM) show negative or near-zero spreads** at most horizons — short-term momentum does not hold for broad market ETFs. DIA actually shows mean reversion behavior (negative momentum assets outperform).
- **60-day lookback is most consistent** for the tickers where momentum works — suggesting intermediate-term trend following rather than short-term continuation.
- **Short-term (1-day lookback)** spreads are negative for most tickers, consistent with mild short-term mean reversion / microstructure noise.
- **Win rates** for long momentum positions range from 50–71%, not high enough alone but meaningful when combined with sizing and regime filters.

---

## Key Findings So Far

1. **Markets are not Gaussian** — kurtosis is massively positive across all tickers. Tail events occur 80–140x more often than normal distribution predicts at 4σ.
2. **Chebyshev holds but is too loose** to be practically useful. Cantelli's one-sided inequality is more appropriate for asymmetric tail analysis.
3. **Correlations spike toward 1.0 in crises** — static correlation matrices underestimate true portfolio risk during market stress.
4. **Regime detection works** — HMM and Bayesian models identify distinct market environments with meaningfully different return and volatility profiles.
5. **Momentum exists but is asset-specific** — NVDA shows strong and consistent momentum across multiple horizons. Broad indices show weak or negative momentum at short horizons.

---

## Up Next

- Phase 8 — Mean Reversion Research
- Phase 9 — Expected Value Framework
- Phase 10 — Bayesian Updating
- Phase 11 — Machine Learning Models
- Phase 12 — Backtesting Engine
- Phase 13 — Out-of-Sample Validation
