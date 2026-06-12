"""
ml_signals.py — ML signal layer

Trains Random Forest and Gradient Boosting classifiers on the existing factor
feature set to predict 5-day forward return direction. Outputs predicted
probabilities as an additional signal column alongside the existing composite
z-score, and compares Information Coefficients across all three signals.

Design principles
-----------------
  - No look-ahead: target is computed with shift(-5), features are lagged
  - Hard train/test split: 2000-2018 train, 2019+ test. Models are never
    re-tuned on test data.
  - TimeSeriesSplit with a gap between folds during CV to prevent leakage
    from adjacent observations bleeding across the fold boundary.
  - Value/quality features excluded (static yfinance snapshots = look-ahead bias)
  - Walk-forward evaluation: re-fits model every RETRAIN_FREQ days on an
    expanding window, predicts the next window. This is how you'd use it live.

Output
------
  data/features/ml_signals.csv   — date × ticker with columns:
      ml_rf_prob    : RF predicted prob of positive 5d return
      ml_gbm_prob   : GBM predicted prob of positive 5d return
      ml_rf_signal  : long / short / neutral from RF prob threshold
      ml_gbm_signal : long / short / neutral from GBM prob threshold

Usage
-----
  python ml_signals.py                  # train + evaluate + save
  python ml_signals.py --train-only     # stop after training, skip test eval
  python ml_signals.py --feature-importance   # print extended feature importance
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from config import FEATURE_DIR
from utils.tickers import load_tickers


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_END      = "2018-12-31"
TEST_START     = "2019-01-01"
LOOKAHEAD      = 5          # predict direction of 5-day forward return
LONG_PROB      = 0.55       # prob threshold for long signal
SHORT_PROB     = 0.45       # prob threshold for short signal
RETRAIN_FREQ   = 63         # re-fit walk-forward model every ~quarter
CV_FOLDS       = 5
CV_GAP         = 10         # trading days gap between train and val in each fold

# Features fed to the ML models — excludes value/quality (look-ahead bias)
FEATURE_COLS = [
    "mom_20",
    "mom_60",
    "mom_120",
    "mom_60_skip1m",
    "rev_zscore_20",
    "rev_zscore_60",
    "rsi_mr",
    "vol_ratio",
    "vol_of_vol",
    "vol_rank_252",
    # Group scores from signals.py (already cross-sectionally normalized)
    "mom_score",
    "rev_score",
    "vol_score",
]

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=50,    # conservative — avoids overfitting on ~6k rows per ticker
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=50,
    random_state=42,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_factor_data(tickers: list[str]) -> pd.DataFrame:
    """
    Load per-ticker factor CSVs and stack into long format.
    Merges group scores from signals.csv if available.
    """
    frames = []
    for ticker in tickers:
        path = FEATURE_DIR / f"{ticker}_factors.csv"
        if not path.exists():
            print(f"  Warning: no factor file for {ticker}, skipping")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No factor files in {FEATURE_DIR}. Run factors.py first.")

    combined = pd.concat(frames)
    combined.index.name = "date"

    # Merge group scores from signals.csv if it exists
    sig_path = FEATURE_DIR / "signals.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path, index_col=0, parse_dates=True)
        score_cols = [c for c in sig.columns if c.endswith("_score")]
        if score_cols:
            sig_scores = sig[["ticker"] + score_cols].copy()
            combined = combined.merge(
                sig_scores,
                left_on=["date", "ticker"],
                right_on=["date", "ticker"],
                how="left",
                suffixes=("", "_sig"),
            )

    return combined


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a single ticker's DataFrame, build feature matrix and target.
    Target: 1 if 5-day forward return > 0, else 0.
    All features are already in df — just select and align.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 5:
        return pd.DataFrame()

    out = df[available].copy()

    # Forward return target — shift(-LOOKAHEAD) gives the return that starts tomorrow
    out["target"] = (df["daily_return"].rolling(LOOKAHEAD).sum().shift(-LOOKAHEAD) > 0).astype(int)
    out["fwd_return"] = df["daily_return"].rolling(LOOKAHEAD).sum().shift(-LOOKAHEAD)

    return out.dropna()


# ── Walk-forward model training ───────────────────────────────────────────────

def walk_forward_predict(
    dataset:       pd.DataFrame,
    model_type:    str,           # "rf" or "gbm"
    train_end:     str = TRAIN_END,
    test_start:    str = TEST_START,
    retrain_freq:  int = RETRAIN_FREQ,
) -> pd.Series:
    """
    Walk-forward prediction:
      1. Train on all data up to the current window boundary.
      2. Predict probabilities for the next retrain_freq days.
      3. Advance boundary by retrain_freq days and repeat.
      4. Only data through train_end is used to seed the initial model.
         Predictions on test data come from models trained on train + earlier test.

    Returns a Series of predicted probabilities indexed by date.
    """
    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    all_dates    = dataset.index.sort_values().unique()

    # Start predictions from the first date we have enough history
    min_train_rows = 252
    first_pred_idx = min_train_rows

    probs = pd.Series(np.nan, index=dataset.index)

    pred_start = first_pred_idx
    while pred_start < len(all_dates):
        train_dates = all_dates[:pred_start]
        pred_dates  = all_dates[pred_start : pred_start + retrain_freq]

        X_train = dataset.loc[train_dates, feature_cols]
        y_train = dataset.loc[train_dates, "target"]

        # Drop rows with NaN in features
        mask    = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(y_train) < min_train_rows or y_train.nunique() < 2:
            pred_start += retrain_freq
            continue

        # Build pipeline with scaler
        if model_type == "rf":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestClassifier(**RF_PARAMS)),
            ])
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingClassifier(**GBM_PARAMS)),
            ])

        clf.fit(X_train, y_train)

        X_pred = dataset.loc[pred_dates, feature_cols]
        valid  = X_pred.notna().all(axis=1)
        if valid.any():
            p = clf.predict_proba(X_pred[valid])[:, 1]
            probs.loc[pred_dates[valid]] = p

        pred_start += retrain_freq

    return probs


# ── Cross-validated AUC on training data ─────────────────────────────────────

def cv_auc(
    dataset:    pd.DataFrame,
    model_type: str,
    train_end:  str = TRAIN_END,
) -> float:
    """
    TimeSeriesSplit CV with a gap — evaluates model quality on training period
    without touching test data.
    """
    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    train        = dataset[dataset.index <= train_end].dropna(subset=feature_cols + ["target"])

    if len(train) < 500:
        return np.nan

    X = train[feature_cols].values
    y = train["target"].values

    tscv   = TimeSeriesSplit(n_splits=CV_FOLDS, gap=CV_GAP)
    aucs   = []

    for fold_train_idx, fold_val_idx in tscv.split(X):
        X_tr, X_val = X[fold_train_idx], X[fold_val_idx]
        y_tr, y_val = y[fold_train_idx], y[fold_val_idx]

        if len(np.unique(y_val)) < 2:
            continue

        if model_type == "rf":
            clf = Pipeline([("scaler", StandardScaler()),
                            ("model",  RandomForestClassifier(**RF_PARAMS))])
        else:
            clf = Pipeline([("scaler", StandardScaler()),
                            ("model",  GradientBoostingClassifier(**GBM_PARAMS))])

        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))

    return float(np.mean(aucs)) if aucs else np.nan


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(
    dataset:    pd.DataFrame,
    model_type: str,
    train_end:  str = TRAIN_END,
) -> pd.DataFrame:
    """Fit a single model on all training data and extract feature importances."""
    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    train        = dataset[dataset.index <= train_end].dropna(subset=feature_cols + ["target"])

    X = train[feature_cols].values
    y = train["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "rf":
        model = RandomForestClassifier(**RF_PARAMS)
    else:
        model = GradientBoostingClassifier(**GBM_PARAMS)

    model.fit(X_scaled, y)

    return pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


# ── Information Coefficient ───────────────────────────────────────────────────

def information_coefficient(
    signal:     pd.Series,
    fwd_return: pd.Series,
    period:     str = "full",
) -> float:
    """
    Rank IC: Spearman correlation between signal and forward return.
    This is the standard quant metric for signal quality.
    """
    aligned = pd.concat([signal, fwd_return], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


# ── Signal assignment ─────────────────────────────────────────────────────────

def prob_to_signal(
    prob:        pd.Series,
    long_prob:   float = LONG_PROB,
    short_prob:  float = SHORT_PROB,
) -> pd.Series:
    sig = pd.Series("neutral", index=prob.index)
    sig[prob >= long_prob]  = "long"
    sig[prob <= short_prob] = "short"
    return sig


# ── Printing ──────────────────────────────────────────────────────────────────

def print_ic_comparison(ic_table: pd.DataFrame):
    print(f"\n{'='*65}")
    print("  Information Coefficient Comparison (Spearman rank IC vs 5d fwd return)")
    print(f"{'='*65}")
    print(f"  {'Signal':<20} {'Train IC':>10} {'Test IC':>10}  {'AUC (CV)':>10}")
    print(f"  {'-'*52}")
    for _, row in ic_table.iterrows():
        train_ic = f"{row.train_ic:+.4f}" if not pd.isna(row.train_ic) else "   N/A"
        test_ic  = f"{row.test_ic:+.4f}"  if not pd.isna(row.test_ic)  else "   N/A"
        auc      = f"{row.auc:.4f}"       if not pd.isna(row.get("auc", np.nan)) else "   N/A"
        print(f"  {row.signal:<20} {train_ic:>10} {test_ic:>10}  {auc:>10}")


def print_feature_importance(imp_df: pd.DataFrame, model_label: str, top_n: int = 10):
    print(f"\n{'='*55}")
    print(f"  Feature Importance — {model_label}")
    print(f"{'='*55}")
    for _, row in imp_df.head(top_n).iterrows():
        bar = "█" * int(row.importance * 200)
        print(f"  {row.feature:<20}  {row.importance:.4f}  {bar}")


def print_signal_counts(signals_df: pd.DataFrame, col: str, label: str):
    counts = signals_df.groupby("ticker")[col].value_counts().unstack(fill_value=0)
    for c in ["long", "neutral", "short"]:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["long", "neutral", "short"]]
    counts["long%"]  = (counts["long"]  / counts.sum(axis=1) * 100).round(1)
    counts["short%"] = (counts["short"] / counts.sum(axis=1) * 100).round(1)
    print(f"\n  {label} signal counts:")
    print(counts.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def run(train_only: bool = False, show_feature_importance: bool = False):
    tickers = load_tickers()

    print("Loading factor data...")
    raw = load_factor_data(tickers)

    all_rf_probs  = []
    all_gbm_probs = []
    ic_rows       = []
    rf_imp_frames = []
    gbm_imp_frames = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        ticker_data = raw[raw["ticker"] == ticker].copy()

        if ticker_data.empty:
            print(f"  Skipping — no data")
            continue

        dataset = build_ml_dataset(ticker_data)
        if len(dataset) < 500:
            print(f"  Skipping — insufficient rows ({len(dataset)})")
            continue

        train_ds = dataset[dataset.index <= TRAIN_END]
        test_ds  = dataset[dataset.index >= TEST_START]

        print(f"  Train rows: {len(train_ds)}  |  Test rows: {len(test_ds)}")

        # CV AUC on training data
        rf_auc  = cv_auc(dataset, "rf",  TRAIN_END)
        gbm_auc = cv_auc(dataset, "gbm", TRAIN_END)
        print(f"  CV AUC — RF: {rf_auc:.4f}  GBM: {gbm_auc:.4f}")

        # Walk-forward predictions
        print(f"  Running walk-forward RF...")
        rf_probs  = walk_forward_predict(dataset, "rf")
        print(f"  Running walk-forward GBM...")
        gbm_probs = walk_forward_predict(dataset, "gbm")

        # Attach ticker
        rf_df  = rf_probs.rename("ml_rf_prob").to_frame()
        gbm_df = gbm_probs.rename("ml_gbm_prob").to_frame()
        rf_df["ticker"]  = ticker
        gbm_df["ticker"] = ticker

        all_rf_probs.append(rf_df)
        all_gbm_probs.append(gbm_df)

        # IC calculation
        fwd = dataset["fwd_return"]

        # Load composite_zscore from signals.csv for this ticker
        comp_z = None
        sig_path = FEATURE_DIR / "signals.csv"
        if sig_path.exists():
            sig_df = pd.read_csv(sig_path, index_col=0, parse_dates=True)
            ticker_sig = sig_df[sig_df["ticker"] == ticker]["composite_zscore"]
            comp_z = ticker_sig.reindex(dataset.index)

        for label, signal, auc in [
            ("composite_z",  comp_z,    np.nan),
            ("rf_prob",      rf_probs,  rf_auc),
            ("gbm_prob",     gbm_probs, gbm_auc),
        ]:
            if signal is None:
                continue
            sig_aligned = signal.reindex(dataset.index)
            train_ic = information_coefficient(
                sig_aligned[sig_aligned.index <= TRAIN_END],
                fwd[fwd.index <= TRAIN_END],
            )
            test_ic = information_coefficient(
                sig_aligned[sig_aligned.index >= TEST_START],
                fwd[fwd.index >= TEST_START],
            ) if not train_only else np.nan

            ic_rows.append({
                "ticker": ticker, "signal": label,
                "train_ic": train_ic, "test_ic": test_ic, "auc": auc,
            })

        # Feature importance (train data only)
        rf_imp  = get_feature_importance(dataset, "rf")
        gbm_imp = get_feature_importance(dataset, "gbm")
        rf_imp["ticker"]  = ticker
        gbm_imp["ticker"] = ticker
        rf_imp_frames.append(rf_imp)
        gbm_imp_frames.append(gbm_imp)

    if not all_rf_probs:
        print("\nNo tickers produced predictions. Check factor files.")
        return

    # ── Combine and save ──────────────────────────────────────────────────────
    print("\nCombining predictions...")

    rf_combined  = pd.concat(all_rf_probs).reset_index().rename(columns={"index": "date"})
    gbm_combined = pd.concat(all_gbm_probs).reset_index().rename(columns={"index": "date"})

    out_df = rf_combined.merge(gbm_combined, on=["date", "ticker"], how="outer")
    out_df = out_df.set_index("date").sort_index()

    out_df["ml_rf_signal"]  = prob_to_signal(out_df["ml_rf_prob"])
    out_df["ml_gbm_signal"] = prob_to_signal(out_df["ml_gbm_prob"])

    out_path = FEATURE_DIR / "ml_signals.csv"
    out_df.to_csv(out_path)
    print(f"  Saved → {out_path}")

    # ── IC comparison table ───────────────────────────────────────────────────
    ic_df = pd.DataFrame(ic_rows)

    # Aggregate across tickers
    ic_summary = (
        ic_df.groupby("signal")[["train_ic", "test_ic", "auc"]]
        .mean()
        .reset_index()
    )
    print_ic_comparison(ic_summary)

    # Per-ticker IC breakdown
    print(f"\n{'='*65}")
    print("  Per-Ticker IC (test period)")
    print(f"{'='*65}")
    ic_pivot = ic_df.pivot_table(
        index="ticker", columns="signal", values="test_ic"
    )
    print(ic_pivot.round(4).to_string())

    # ── Feature importance ────────────────────────────────────────────────────
    if show_feature_importance or True:   # always show summary
        rf_imp_all  = pd.concat(rf_imp_frames).groupby("feature")["importance"].mean().reset_index()
        gbm_imp_all = pd.concat(gbm_imp_frames).groupby("feature")["importance"].mean().reset_index()

        rf_imp_all  = rf_imp_all.sort_values("importance", ascending=False)
        gbm_imp_all = gbm_imp_all.sort_values("importance", ascending=False)

        print_feature_importance(rf_imp_all,  "Random Forest (avg across tickers)")
        print_feature_importance(gbm_imp_all, "Gradient Boosting (avg across tickers)")

        # Save importance tables
        rf_imp_all.to_csv(FEATURE_DIR / "feature_importance_rf.csv",  index=False)
        gbm_imp_all.to_csv(FEATURE_DIR / "feature_importance_gbm.csv", index=False)
        print(f"\n  Feature importance saved → {FEATURE_DIR}/feature_importance_*.csv")

    # ── Signal distribution ───────────────────────────────────────────────────
    print_signal_counts(out_df, "ml_rf_signal",  "RF")
    print_signal_counts(out_df, "ml_gbm_signal", "GBM")

    print(f"\n{'='*65}")
    print("  Done. To use ML signals in the backtester:")
    print("  Load ml_signals.csv and merge on (date, ticker),")
    print("  then replace 'signal' column with 'ml_rf_signal' or 'ml_gbm_signal'.")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only",          action="store_true")
    parser.add_argument("--feature-importance",  action="store_true")
    args = parser.parse_args()

    run(
        train_only=args.train_only,
        show_feature_importance=args.feature_importance,
    )