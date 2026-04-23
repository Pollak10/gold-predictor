# GoldCast — Gold Price Prediction App

GoldCast is a machine learning application that predicts the future closing price of gold using historical price data and technical indicators. You run it, it pulls ten years of gold futures data from Yahoo Finance, trains a Random Forest model on engineered features, evaluates it against a held out test set, and prints a forecast for the next five trading days alongside a dashboard of charts.

---

## What it does

- Downloads historical gold futures data (ticker: `GC=F`) directly from Yahoo Finance  no manual dataset download needed
- Engineers 40+ technical indicator features from raw OHLCV data (moving averages, RSI, MACD, Bollinger Bands, momentum, volatility)
- Trains a Random Forest regressor and a Gradient Boosting model for comparison
- Uses time series cross validation to avoid data leakage
- Prints MAE, RMSE, MAPE, and R² for both models
- Forecasts the gold price 5 trading days ahead from the most recent data
- Saves a results dashboard chart to `goldcast_results.png`

---

## Algorithm

**Random Forest** was chosen as the primary model for three reasons. First, it handles the non linear relationships between technical indicators and price movement well without requiring extensive hyperparameter tuning. Second, it is naturally resistant to overfitting through bagging and feature subsampling. Third, it produces feature importance scores, which make it straightforward to understand which indicators are driving predictions useful for a financial application where explainability matters.

**Gradient Boosting** is included as a comparison baseline. It typically achieves slightly lower error on tabular data but is more sensitive to hyperparameters and slower to train.

Both models are regression models they predict a continuous price value, not a buy/sell signal.

---

## Dataset

| Property | Value |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Ticker | `GC=F` (Gold Futures, front month contract) |
| Date range | January 2015 – December 2024 |
| Records | ~2,500 trading days |
| Raw features | Open, High, Low, Close, Volume |
| Engineered features | 40+ (see below) |
| Target variable | Closing price 5 trading days ahead |

**Engineered features include:**

| Feature | Description |
|---|---|
| `close_lag_N` | Closing price N days ago (1, 2, 3, 5, 10, 20) |
| `ma_N` | Simple moving average over N days (5, 10, 20, 50) |
| `std_N` | Rolling standard deviation over N days |
| `return_Nd` | Percentage price change over N days |
| `volatility_N` | Rolling standard deviation of daily returns |
| `rsi_14` | Relative Strength Index (14-day) |
| `macd` / `macd_signal` / `macd_hist` | MACD line, signal line, histogram |
| `bb_pct` | Position within Bollinger Bands |
| `daily_range` | (High - Low) / Close |
| `volume_ratio` | Volume relative to 10-day average |
| `day_of_week` / `month` / `quarter` | Calendar features |

**Preprocessing steps:**
1. Raw OHLCV data downloaded and any rows with missing values dropped
2. Features engineered from rolling windows and price transformations
3. Rows with NaN (from warm up periods of rolling calculations) dropped
4. Data split 80/20 into training and test sets in chronological order no shuffling
5. Features scaled with `StandardScaler` fitted on training data only, then applied to test data

---

## Stack

| Library | Role |
|---|---|
| `yfinance` | Downloads historical gold price data from Yahoo Finance |
| `pandas` | Data manipulation, rolling calculations, DataFrame operations |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | Random Forest, Gradient Boosting, StandardScaler, metrics, TimeSeriesSplit |
| `matplotlib` | Results dashboard chart |
| `seaborn` | Chart styling |

---

## How it works

```
Yahoo Finance API
      │
      ▼
Raw OHLCV DataFrame (~2,500 rows)
      │
      ▼
Feature Engineering (40+ technical indicators)
      │
      ▼
Train/Test Split (80% train, 20% test — chronological)
      │
      ├──► Random Forest Regressor (300 trees, time-series CV)
      │           │
      └──► Gradient Boosting (200 estimators, comparison)
                  │
                  ▼
          Evaluation (MAE, RMSE, MAPE, R²)
                  │
                  ▼
          5-Day Ahead Forecast (from most recent row)
                  │
                  ▼
          goldcast_results.png (dashboard chart)
```

The key design decision is the temporal train/test split. Unlike standard ML problems where you can shuffle data, time series data must be split in order — training on data from 2015–2022, testing on 2023–2024. Shuffling would allow the model to "see the future" during training and produce misleadingly good metrics.

---

## Running it

**Requirements:** Python 3.9+

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/goldcast-predictor.git
cd goldcast-predictor

# Install dependencies
pip install -r requirements.txt

# Run
python gold_price_predictor.py
```

No API key needed. Data is pulled from Yahoo Finance which is free and public.

**Expected output:**
```
====================================================
  GoldCast — Gold Price Prediction App
====================================================

Downloading gold price data (2015-01-01 to 2024-12-31)...
Downloaded 2513 trading days of data.

Engineering features...
Dataset after feature engineering: 2453 rows, 48 columns.

Training set : 1962 samples (2015-07-15 to 2022-11-29)
Test set     :  491 samples (2022-11-30 to 2024-12-31)

Training Random Forest model...
  Cross-validation R² scores: [0.978 0.961 0.982 0.975 0.969]
  Mean CV R²: 0.973 (+/- 0.007)

── Random Forest Test Results ──────────────────────
  MAE  : $28.14
  RMSE : $41.22
  MAPE : 1.23%
  R²   : 0.9821

┌─────────────────────────────────────────────┐
│              Price Forecast                  |
├─────────────────────────────────────────────┤
│  Last close  (2024-12-31): $2,625.40        │
│  Forecast (5-day ahead)  : $2,638.70        │
│  Expected change         : +0.51%           │
└─────────────────────────────────────────────┘

Results chart saved to goldcast_results.png
```

The chart shows: full price history with train/test split marked, predicted vs actual prices on the test set, residual plot, residual distribution, and top 15 feature importances.

---

## Results

The Random Forest model achieved an R² of approximately 0.98 on the test set, with a mean absolute error around $28 per ounce. For context, gold was trading between $1,800 and $2,700 per ounce during the test period, so a $28 MAE represents roughly 1.2% average error.

Feature importance analysis consistently ranked lag features (yesterday's closing price, 5-day lag) and moving averages (20-day, 50-day) as the most predictive features. RSI and MACD contributed meaningfully but ranked lower. Calendar features (day of week, month) had minimal importance, which makes sense for a commodity driven by macroeconomic factors rather than seasonal retail patterns.

Gradient Boosting performed comparably, within a few dollars MAE of Random Forest.

---

## Limitations and potential improvements

The high R² is partly expected with financial time series because tomorrow's price is usually close to today's price a naive model that just predicts "tomorrow equals today" would also score reasonably well. A more meaningful evaluation would measure directional accuracy (did the model correctly predict whether price went up or down) rather than raw price error.

Things that would improve the model:

- **Macroeconomic features** — adding USD index (DXY), real interest rates, and inflation data would capture the macro drivers of gold price that technical indicators miss
- **Sentiment features** — gold is sensitive to geopolitical news; incorporating a news sentiment score could improve short term predictions
- **Longer forecast horizon evaluation** — currently predicting 5 days ahead; evaluating across multiple horizons (1, 5, 10, 20 days) would give a fuller picture of model decay
- **Walk-forward validation** — rather than a single train/test split, retraining the model monthly on a rolling window would better reflect real world deployment conditions

---

## References

Anthropic. (2024). *Claude API documentation*. https://docs.anthropic.com

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56. https://pandas.pydata.org

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://scikit-learn.org

Raschka, S. (2018). *Model evaluation, model selection, and algorithm selection in machine learning*. https://arxiv.org/abs/1811.12808

Ran Aroussi. (2019). *yfinance: Yahoo! Finance market data downloader* [Software]. https://github.com/ranaroussi/yfinance
