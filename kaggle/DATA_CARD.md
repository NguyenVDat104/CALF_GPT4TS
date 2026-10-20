# 🪙 BTCUSDT High-Frequency Time Series Benchmark (CALF Dataset)

[![License: CC0](https://img.shields.io/badge/License-CC0%201.0-blue.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Temporal Resolution](https://img.shields.io/badge/Frequency-15m%20%7C%201h%20%7C%204h%20%7C%201d-brightgreen.svg)](#)
[![Horizon Coverage](https://img.shields.io/badge/Coverage-2024--2026%20(2%20Years)-orange.svg)](#)
[![Domain](https://img.shields.io/badge/Domain-Financial%20Time%20Series%20%26%20LLM%20Forecasting-purple.svg)](#)

---

## 📌 1. Overview & Context

This dataset contains clean, high-frequency time series data for **Bitcoin (BTC/USDT)**, covering over **70,000 continuous 15-minute observations** spanning 2 full years (2024–2026), alongside multi-scale temporal aggregations (**1h, 4h, 1d**), high-dimensional **technical feature sets (28 indicators)**, and cross-asset companion benchmarks (**ETH/USDT**).

The dataset is curated specifically for testing and evaluating state-of-the-art Deep Learning models for Time Series, including:
* **Foundation Models for Time Series**: CALF (Cross-modal Alignment for LLM Forecasting), GPT4TS, PatchTST, TimesNet, DLinear.
* **Cross-Modal Vision & Language Distillation**: Transferring multimodal representations from Vision Transformers (**DINO-V3**) and Causal Language Models (**GPT-2**) into financial forecasting.
* **Long-Sequence Time-Series Forecasting (LSTF)**: Evaluating horizons $H \in \{96, 192, 336, 720\}$.

---

## 📁 2. Dataset Files & Structure

| File Name | Frequency | Observations | Columns | Size | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `BTCUSDT_15m_calf.csv` | **15 Minutes** | **70,177** | 8 | ~6.0 MB | Primary benchmark for high-frequency neural forecasting (CALF). |
| `BTCUSDT_15m_features_calf.csv` | **15 Minutes** | **70,177** | **28** | ~17.4 MB | High-dimensional multivariate time series with technical oscillators & cyclical embeddings. |
| `BTCUSDT_1h_calf.csv` | **1 Hour** | **17,545** | 8 | ~1.6 MB | Standard resolution for multi-horizon benchmarks & swing modeling. |
| `BTCUSDT_4h_calf.csv` | **4 Hours** | **4,387** | 8 | ~0.4 MB | Intermediate swing horizon forecasting. |
| `BTCUSDT_1d_calf.csv` | **1 Day** | **732** | 8 | ~0.1 MB | Macro regime detection & long-term trend forecasting. |
| `ETHUSDT_1h_calf.csv` | **1 Hour** | **1,000** | 8 | ~83 KB | Companion cross-asset benchmark (Ethereum) for transfer learning evaluation. |

---

## 📊 3. Column Data Dictionary

### Base OHLCV Files (`BTCUSDT_15m_calf.csv`, `BTCUSDT_1h_calf.csv`, etc.)
| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | `timestamp` | UTC | Timestamp indicating the start of the candlestick interval (YYYY-MM-DD HH:MM:SS). |
| `open` | `float64` | USDT | Opening price of Bitcoin during the 15m candle. |
| `high` | `float64` | USDT | Highest price reached during the interval. |
| `low` | `float64` | USDT | Lowest price reached during the interval. |
| `close` | `float64` | USDT | Closing price at interval end (Primary target for regression). |
| `volume` | `float64` | BTC | Total base asset volume traded (in BTC). |
| `quote_volume` | `float64` | USDT | Total quote asset volume traded (in USDT turnover). |
| `number_of_trades` | `int64` | Count | Number of discrete matched market fills executed during the bar. |

### Extended Feature Dataset (`BTCUSDT_15m_features_calf.csv`)
Includes all base columns plus:
* **Returns & Risk**: `log_return` ($\ln(P_t / P_{t-1})$), `volatility_24h` (Rolling 96-step annualized standard deviation).
* **Trend Indicators**: `sma_20`, `sma_50`, `sma_200`, `ema_21`.
* **Momentum & Oscillators**: `rsi_14` (Relative Strength Index), `macd`, `macd_signal`, `macd_hist`.
* **Volatility Bands**: `bollinger_upper`, `bollinger_lower`, `bollinger_width`, `atr_14` (Average True Range).
* **Volume Profile**: `volume_sma_20`, `volume_ratio` ($V_t / \overline{V}_{20}$).
* **Cyclical Temporal Encodings**: `hour_sin`, `hour_cos` (24h period), `day_sin`, `day_cos` (7-day period).

---

## 🧪 4. Data Quality & Preprocessing

* **Zero Missing Values**: 100% complete time sequence with 0 missing steps across all 70,177 intervals.
* **Cadence Verification**: Strict 15-minute interval continuity ($\Delta t = 15\text{m}$) confirmed without gaps or duplicates.
* **Instance Normalization**: Recommended for Deep Learning using **Reversible Instance Normalization (RevIN)**:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \hat{y} = \tilde{y} \cdot \sqrt{\sigma^2 + \epsilon} + \mu$$

---

## 📈 5. Exploratory Data Analysis & Visualization

An end-to-end, interactive Jupyter Notebook is provided in the **Code** tab:  
👉 **See Code Tab: `btcusdt-time-series-eda-and-visualization`**

The notebook provides:
1. **Interactive Candlestick Chart**: Plotly zoomable charts with SMA crossovers and volume indicators.
2. **Distribution & Heavy Tails**: Log-returns histogram, Gaussian fit, and Q-Q plots proving leptokurtic behavior (excess kurtosis).
3. **Temporal Heatmaps**: Activity heatmaps across UTC hours (0–23) and Days of the Week.
4. **Time Series Decomposition**: STL separation into Trend, 24h Seasonality, and Residuals.
5. **Autocorrelation & Stationarity**: Augmented Dickey-Fuller (ADF) tests and ACF/PACF plots up to lag 96.
6. **CALF Tokenization**: Visualizing the conversion of 1D continuous series into 2D patch matrices ($P=16, S=8$) for Vision Transformers.

---

## 🏷️ Recommended Tags for Kaggle
`cryptocurrency`, `finance`, `time-series-analysis`, `deep-learning`, `exploratory-data-analysis`, `bitcoin`

---

## 📜 Citation & Attribution
If using this dataset for research or project evaluation, please cite:
```bibtex
@dataset{datnguyen2026calf_btcusdt,
  author    = {Dat Nguyen},
  title     = {BTCUSDT 15m Time Series - CALF Forecasting Benchmark},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/iamdatpeo/btcusdt-15m-time-series-calf-dataset}
}
```
