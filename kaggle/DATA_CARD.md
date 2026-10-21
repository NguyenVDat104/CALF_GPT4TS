# 🪙 Multi-Asset Cryptocurrency High-Frequency Benchmark (CALF 1GB Suite)

[![License: CC0](https://img.shields.io/badge/License-CC0%201.0-blue.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Dataset Size](https://img.shields.io/badge/Size-1.06%20GB%20%7C%2011%20Files-blueviolet.svg)](#)
[![Temporal Resolution](https://img.shields.io/badge/Frequency-1m%20%7C%2015m%20%7C%201h%20%7C%201d-brightgreen.svg)](#)
[![Assets](https://img.shields.io/badge/Assets-BTC%20%7C%20ETH%20%7C%20SOL%20%7C%20BNB-orange.svg)](#)
[![Domain](https://img.shields.io/badge/Domain-Foundation%20Models%20%26%20LLM%20Forecasting-purple.svg)](#)

---

## 📌 1. Overview & Research Context

This dataset is a large-scale, enterprise-grade cryptocurrency time-series benchmark totaling **1.06 GB (over 8.4 million continuous high-frequency observations)**. It is specifically designed to pre-train, fine-tune, and rigorously benchmark modern Deep Learning and Foundation Models for Time Series Forecasting.

### Key Target Research Areas:
* **Foundation Models for Time Series**: CALF (Cross-modal Alignment for LLM Forecasting), GPT4TS, PatchTST, TimesNet, DLinear.
* **Cross-Modal ViT & LLM Distillation**: Transferring spatial-temporal representations from Vision Transformers (**DINO-V3**) and Causal Language Models (**GPT-2**) into financial sequence forecasting.
* **Long-Sequence Time-Series Forecasting (LSTF)**: Evaluating multi-horizon prediction spans $H \in \{96, 192, 336, 720\}$.
* **High-Frequency Market Microstructure**: Testing models under 1-minute high-volatility market dynamics.

---

## 📁 2. Dataset Files & Structure (1.06 GB Total)

| File Name | Asset | Frequency | Observations | Columns | File Size | Primary Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `BTCUSDT_1m_features_deep_learning.csv` | **BTC** | **1 Minute** | **1,051,200** | **26** | **206.7 MB** | Deep Learning feature matrix with 18 technical indicators + cyclical embeddings. |
| `ETHUSDT_1m_features_deep_learning.csv` | **ETH** | **1 Minute** | **1,051,200** | **26** | **203.6 MB** | Full feature-engineered high-dimensional series for Ethereum. |
| `BNBUSDT_1m_features_deep_learning.csv` | **BNB** | **1 Minute** | **1,051,200** | **26** | **192.0 MB** | Full feature-engineered high-dimensional series for Binance Coin. |
| `SOLUSDT_1m_features_deep_learning.csv` | **SOL** | **1 Minute** | **1,051,200** | **26** | **180.0 MB** | Full feature-engineered high-dimensional series for Solana. |
| `BTCUSDT_1m_high_frequency.csv` | **BTC** | **1 Minute** | **1,051,200** | 8 | **76.8 MB** | Pure 1-minute OHLCV Bitcoin raw high-frequency benchmark. |
| `ETHUSDT_1m_high_frequency.csv` | **ETH** | **1 Minute** | **1,051,200** | 8 | **77.2 MB** | Pure 1-minute OHLCV Ethereum raw high-frequency benchmark. |
| `BNBUSDT_1m_high_frequency.csv` | **BNB** | **1 Minute** | **1,051,200** | 8 | **71.8 MB** | Pure 1-minute OHLCV Binance Coin raw high-frequency benchmark. |
| `SOLUSDT_1m_high_frequency.csv` | **SOL** | **1 Minute** | **1,051,200** | 8 | **67.9 MB** | Pure 1-minute OHLCV Solana raw high-frequency benchmark. |
| `BTCUSDT_15m_calf.csv` | **BTC** | **15 Minutes** | **70,177** | 8 | **6.0 MB** | Core academic baseline for CALF 15m experiments. |
| `BTCUSDT_1h_calf.csv` | **BTC** | **1 Hour** | **17,545** | 8 | **1.6 MB** | Standard intermediate horizon benchmark. |
| `BTCUSDT_1d_calf.csv` | **BTC** | **1 Day** | **732** | 8 | **0.1 MB** | Macro daily trend & regime transition benchmark. |

**Total Size**: **1,083.6 MB (~1.06 GB)** across **11 files**.

---

## 📊 3. Column Data Dictionary

### Standard OHLCV Schema (`*_1m_high_frequency.csv`, `*_15m_calf.csv`, etc.)
| Column | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | `timestamp` | UTC | Interval open timestamp (`YYYY-MM-DD HH:MM:SS`). |
| `open` | `float64` | USDT | Opening price at interval start. |
| `high` | `float64` | USDT | Highest price reached during interval. |
| `low` | `float64` | USDT | Lowest price reached during interval. |
| `close` | `float64` | USDT | Closing price at interval end (Primary forecasting target). |
| `volume` | `float64` | Asset | Base currency volume traded (e.g. BTC, ETH). |
| `quote_volume` | `float64` | USDT | Dollar turnover turned over during interval. |
| `number_of_trades` | `int64` | Count | Number of discrete matched market fills executed. |

### High-Dimensional Feature Schema (`*_1m_features_deep_learning.csv`)
Includes all base OHLCV columns plus 18 advanced statistical and temporal features:
* **Log-Returns**: $\ln(P_t / P_{t-1})$ for stationary residual modeling.
* **Rolling Volatility**: 60-period annualized standard deviation of log returns.
* **Trend Ensembles**: `sma_20`, `sma_50`, `sma_200`, and `ema_21`.
* **Momentum & Oscillators**: `rsi_14` (Relative Strength Index), `macd`, `macd_signal`, and `macd_hist`.
* **Volatility Envelopes**: `bollinger_upper`, `bollinger_lower`, `bollinger_width`, and `atr_14` (Average True Range).
* **Volume Profiles**: `volume_sma_20`, and `volume_ratio` ($V_t / \overline{V}_{20}$).
* **Temporal Sin/Cos Embeddings**: Continuous cyclic time features for hour of day (`hour_sin`, `hour_cos`).

---

## 🧪 4. Data Quality & Preprocessing Standards

* **Zero Missing Values**: 100% complete time sequences with 0 missing steps across all 8.4M+ intervals.
* **Cadence Verification**: Strict interval continuity ($\Delta t = 1\text{m}, 15\text{m}, 1\text{h}$) confirmed without gaps or duplicates.
* **Instance Normalization**: Recommended for Deep Learning using **Reversible Instance Normalization (RevIN)**:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \hat{y} = \tilde{y} \cdot \sqrt{\sigma^2 + \epsilon} + \mu$$

---

## 📈 5. Exploratory Data Analysis & Visualization Notebook

An end-to-end, interactive Jupyter Notebook is provided in the **Code** tab:  
👉 **See Code Tab: `btcusdt-time-series-eda-and-visualization`**

The notebook covers:
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
@dataset{datnguyen2026calf_crypto_1gb,
  author    = {Dat Nguyen},
  title     = {Multi-Asset Cryptocurrency High-Frequency Benchmark (CALF 1GB Suite)},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/iamdatpeo/btcusdt-15m-time-series-calf-dataset}
}
```
