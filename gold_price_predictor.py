"""
GoldCast — Gold Price Prediction App
Machine learning application using Random Forest to predict gold closing prices.

Requirements:
    pip install yfinance pandas numpy scikit-learn matplotlib seaborn

Usage:
    python gold_price_predictor.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import yfinance as yf

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
TICKER        = "GC=F"          
START_DATE    = "2015-01-01"
END_DATE      = "2024-12-31"
FORECAST_DAYS = 5
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

def fetch_gold_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical gold futures data from Yahoo Finance."""
    print(f"Downloading gold price data ({start} to {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance. Check your internet connection.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    print(f"Downloaded {len(df)} trading days of data.\n")
    return df

def engineer_features(df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """
    Create technical indicator features from raw OHLCV data.
    All features are derived from historical prices only — no look-ahead bias.
    """
    data = df.copy()

    close = data["Close"]
    high  = data["High"]
    low   = data["Low"]
    vol   = data["Volume"]

    for lag in [1, 2, 3, 5, 10, 20]:
        data[f"close_lag_{lag}"] = close.shift(lag)

    for window in [5, 10, 20, 50]:
        data[f"ma_{window}"]         = close.rolling(window).mean()
        data[f"std_{window}"]        = close.rolling(window).std()
        data[f"high_roll_{window}"]  = high.rolling(window).max()
        data[f"low_roll_{window}"]   = low.rolling(window).min()

    data["return_1d"]  = close.pct_change(1)
    data["return_5d"]  = close.pct_change(5)
    data["return_20d"] = close.pct_change(20)

    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["volatility_20"] = data["return_1d"].rolling(20).std()

    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / (loss + 1e-9)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    data["macd"]   = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"]   = data["macd"] - data["macd_signal"]

    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    data["bb_upper"] = bb_mid + 2 * bb_std
    data["bb_lower"] = bb_mid - 2 * bb_std
    data["bb_pct"]   = (close - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"] + 1e-9
    )

    data["daily_range"]     = (high - low) / close
    data["open_close_diff"] = (data["Close"] - data["Open"]) / data["Open"]

    data["volume_ma_10"]    = vol.rolling(10).mean()
    data["volume_ratio"]    = vol / (data["volume_ma_10"] + 1e-9)

    data["day_of_week"]  = data.index.dayofweek
    data["month"]        = data.index.month
    data["quarter"]      = data.index.quarter

    data["target"] = close.shift(-forecast_days)

    data.dropna(inplace=True)

    return data


def split_data(data: pd.DataFrame, test_size: float):
    """
    Split into train and test sets preserving temporal order.
    Shuffling would leak future data into training — never do that with time series.
    """
    feature_cols = [c for c in data.columns
                    if c not in {"Open", "High", "Low", "Close", "Volume",
                                 "Adj Close", "target"}]

    X = data[feature_cols]
    y = data["target"]

    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, y_train):
    """Train a Random Forest regressor with time-series cross-validation."""
    print("Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    tscv   = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    cv_scores = cross_val_score(
        model, X_scaled, y_train,
        cv=tscv, scoring="r2", n_jobs=-1
    )
    print(f"  Cross-validation R² scores: {cv_scores.round(3)}")
    print(f"  Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")

    model.fit(X_scaled, y_train)
    return model, scaler


def train_gradient_boosting(X_train, y_train, scaler):
    """Train a Gradient Boosting model as a comparison baseline."""
    print("Training Gradient Boosting model (comparison)...")
    X_scaled = scaler.transform(X_train)

    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb_model.fit(X_scaled, y_train)
    print("  Done.\n")
    return gb_model

def evaluate_model(model, scaler, X_test, y_test, name="Random Forest"):
    """Compute and print regression metrics."""
    X_scaled = scaler.transform(X_test)
    preds    = model.predict(X_scaled)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    print(f"── {name} Test Results ─────────────────────")
    print(f"  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}\n")

    return preds, {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def forecast_next(model, scaler, data: pd.DataFrame, feature_cols: list,
                  forecast_days: int) -> float:
    """Use the most recent row of features to forecast the next price."""
    latest = data[feature_cols].iloc[[-1]]
    scaled = scaler.transform(latest)
    pred   = model.predict(scaled)[0]
    return pred

def plot_results(data, X_test, y_test, rf_preds, gb_preds,
                 rf_metrics, gb_metrics, feature_cols, model, forecast_days):

    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("GoldCast — Gold Price Prediction Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(4, 2, (1, 2))
    ax1.plot(data.index, data["Close"], color="#D4AF37", linewidth=1.2,
             label="Gold Close Price")
    split_date = X_test.index[0]
    ax1.axvline(split_date, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Train/Test split ({split_date.date()})")
    ax1.set_title("Historical Gold Price (GC=F)", fontsize=13)
    ax1.set_ylabel("Price (USD)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend()

    ax2 = fig.add_subplot(4, 2, (3, 4))
    ax2.plot(y_test.index, y_test.values, color="#D4AF37", linewidth=1.5,
             label="Actual Price", alpha=0.9)
    ax2.plot(y_test.index, rf_preds, color="#2196F3", linewidth=1.2,
             linestyle="--", label="Random Forest Prediction", alpha=0.85)
    ax2.plot(y_test.index, gb_preds, color="#4CAF50", linewidth=1.2,
             linestyle=":", label="Gradient Boosting Prediction", alpha=0.85)
    ax2.set_title(f"Predicted vs Actual Gold Price (Test Set, {forecast_days}-Day Ahead)",
                  fontsize=13)
    ax2.set_ylabel("Price (USD)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend()

    ax3 = fig.add_subplot(4, 2, 5)
    residuals = y_test.values - rf_preds
    ax3.scatter(rf_preds, residuals, alpha=0.4, color="#2196F3", s=10)
    ax3.axhline(0, color="crimson", linewidth=1)
    ax3.set_title("Residuals (Random Forest)", fontsize=12)
    ax3.set_xlabel("Predicted Price")
    ax3.set_ylabel("Residual")

    ax4 = fig.add_subplot(4, 2, 6)
    ax4.hist(residuals, bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
    ax4.axvline(0, color="crimson", linewidth=1.5)
    ax4.set_title("Residual Distribution", fontsize=12)
    ax4.set_xlabel("Residual (USD)")
    ax4.set_ylabel("Frequency")

    ax5 = fig.add_subplot(4, 2, (7, 8))
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(15)
    )
    colors = ["#D4AF37" if i >= len(fi_df) - 3 else "#B0B0B0"
              for i in range(len(fi_df))]
    ax5.barh(fi_df["feature"], fi_df["importance"], color=colors)
    ax5.set_title("Top 15 Feature Importances (Random Forest)", fontsize=12)
    ax5.set_xlabel("Importance Score")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("goldcast_results.png", dpi=150, bbox_inches="tight")
    print("Results chart saved to goldcast_results.png")
    plt.show()


def print_metrics_table(rf_metrics, gb_metrics):
    """Print a side-by-side metrics comparison."""
    print("\n┌─────────────────────────────────────────────┐")
    print("│          Model Comparison Summary            │")
    print("├──────────┬──────────────────┬────────────────┤")
    print("│ Metric   │  Random Forest   │ Grad. Boosting │")
    print("├──────────┼──────────────────┼────────────────┤")
    for key in ["MAE", "RMSE", "MAPE", "R2"]:
        rf_val = rf_metrics[key]
        gb_val = gb_metrics[key]
        unit   = "%" if key == "MAPE" else ("" if key == "R2" else "$")
        print(f"│ {key:<8} │ {rf_val:>10.4f} {unit:<5} │ {gb_val:>8.4f} {unit:<5} │")
    print("└──────────┴──────────────────┴────────────────┘\n")

def main():
    print("\n" + "=" * 52)
    print("  GoldCast — Gold Price Prediction App")
    print("=" * 52 + "\n")

    raw_df = fetch_gold_data(TICKER, START_DATE, END_DATE)

    print("Engineering features...")
    data = engineer_features(raw_df, FORECAST_DAYS)
    print(f"Dataset after feature engineering: {data.shape[0]} rows, "
          f"{data.shape[1]} columns.\n")

    X_train, X_test, y_train, y_test, feature_cols = split_data(data, TEST_SIZE)
    print(f"Training set : {len(X_train)} samples "
          f"({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"Test set     : {len(X_test)} samples "
          f"({X_test.index[0].date()} to {X_test.index[-1].date()})\n")

    rf_model, scaler  = train_random_forest(X_train, y_train)
    gb_model          = train_gradient_boosting(X_train, y_train, scaler)

    rf_preds, rf_metrics = evaluate_model(rf_model, scaler, X_test, y_test,
                                          "Random Forest")
    gb_preds, gb_metrics = evaluate_model(gb_model, scaler, X_test, y_test,
                                          "Gradient Boosting")
    print_metrics_table(rf_metrics, gb_metrics)

    next_price = forecast_next(rf_model, scaler, data, feature_cols, FORECAST_DAYS)
    last_price = float(data["Close"].iloc[-1])
    last_date  = data.index[-1].date()
    change_pct = (next_price - last_price) / last_price * 100

    print("┌─────────────────────────────────────────────┐")
    print("│              Price Forecast                  │")
    print("├─────────────────────────────────────────────┤")
    print(f"│  Last close  ({last_date}): ${last_price:,.2f}      │")
    print(f"│  Forecast ({FORECAST_DAYS}-day ahead)  : ${next_price:,.2f}      │")
    print(f"│  Expected change        : {change_pct:+.2f}%            │")
    print("└─────────────────────────────────────────────┘\n")

    plot_results(data, X_test, y_test, rf_preds, gb_preds,
                 rf_metrics, gb_metrics, feature_cols, rf_model, FORECAST_DAYS)


if __name__ == "__main__":
    main()
