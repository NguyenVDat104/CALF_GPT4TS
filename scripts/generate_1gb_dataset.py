"""
CALF Large-Scale Benchmark Suite Generator (~1 GB)
Generates high-frequency 1-minute crypto time-series data & technical features.
"""

import os
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "datasets", "expanded_1gb")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_price_path(n_steps, start_price, end_price, vol_annual=0.65, seed=42):
    np.random.seed(seed)
    dt = 1.0 / (365.25 * 24 * 60)
    mu = np.log(end_price / start_price) / (n_steps * dt)
    
    # Geometric Brownian Motion with stochastic drift
    shocks = np.random.normal(0, np.sqrt(dt), n_steps)
    # Add occasional market jump processes (Poisson jumps)
    jumps = np.random.poisson(0.0005, n_steps) * np.random.normal(0, 0.008, n_steps)
    
    log_returns = (mu - 0.5 * vol_annual**2) * dt + vol_annual * shocks + jumps
    log_returns[0] = 0.0
    price_path = start_price * np.exp(np.cumsum(log_returns))
    return price_path

def generate_ohlcv_dataframe(dates, close_prices, base_vol=15.0, seed=42):
    np.random.seed(seed)
    n = len(close_prices)
    
    # Realistic candle formation
    noise_o = np.random.normal(0, 0.0005, n)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * (1 + noise_o[0])
    
    # High and Low bounds
    spread = np.abs(np.random.normal(0.0015, 0.0008, n)) * close_prices
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 0.5, n)) * spread
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 0.5, n)) * spread
    
    # Volume and trade dynamics correlated with volatility
    vol_factor = np.abs(close_prices - open_prices) / (open_prices + 1e-9)
    volume = np.maximum(0.1, np.random.lognormal(mean=2.2, sigma=0.8, size=n) * (1 + vol_factor * 120)) * base_vol
    quote_volume = volume * close_prices
    trades = np.maximum(10, (volume * np.random.uniform(15, 45, n)).astype(np.int64))
    
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d %H:%M:%S'),
        'open': np.round(open_prices, 2),
        'high': np.round(high_prices, 2),
        'low': np.round(low_prices, 2),
        'close': np.round(close_prices, 2),
        'volume': np.round(volume, 4),
        'quote_volume': np.round(quote_volume, 2),
        'number_of_trades': trades
    })
    return df

def generate_features_dataframe(df_ohlcv):
    print("   Engineering 18 technical & cross-modal features...")
    df = df_ohlcv.copy()
    close = df['close'].values
    vol = df['volume'].values
    n = len(close)
    
    # 1. Log returns & rolling volatility
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / (close[:-1] + 1e-9))
    df['log_return'] = np.round(log_ret, 6)
    
    s_ret = pd.Series(log_ret)
    df['volatility_60m'] = np.round(s_ret.rolling(60).std().bfill().values * np.sqrt(60 * 24 * 365), 4)
    
    # 2. Moving averages
    s_close = pd.Series(close)
    df['sma_20'] = np.round(s_close.rolling(20).mean().bfill().values, 2)
    df['sma_50'] = np.round(s_close.rolling(50).mean().bfill().values, 2)
    df['sma_200'] = np.round(s_close.rolling(200).mean().bfill().values, 2)
    df['ema_21'] = np.round(s_close.ewm(span=21, adjust=False).mean().values, 2)
    
    # 3. Oscillators (RSI & MACD)
    delta = s_close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().bfill()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().bfill()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = np.round(100 - (100 / (1 + rs)), 2)
    
    ema_12 = s_close.ewm(span=12, adjust=False).mean()
    ema_26 = s_close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = np.round(macd.values, 2)
    df['macd_signal'] = np.round(macd_signal.values, 2)
    df['macd_hist'] = np.round((macd - macd_signal).values, 2)
    
    # 4. Bollinger Bands & ATR
    r_std = s_close.rolling(20).std().bfill()
    df['bollinger_upper'] = np.round(df['sma_20'] + 2 * r_std, 2)
    df['bollinger_lower'] = np.round(df['sma_20'] - 2 * r_std, 2)
    df['bollinger_width'] = np.round((df['bollinger_upper'] - df['bollinger_lower']) / (df['sma_20'] + 1e-9), 4)
    
    h = df['high'].values
    l = df['low'].values
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(close, 1)), np.abs(l - np.roll(close, 1))))
    tr[0] = h[0] - l[0]
    df['atr_14'] = np.round(pd.Series(tr).rolling(14).mean().bfill().values, 2)
    
    # 5. Volume dynamics
    s_vol = pd.Series(vol)
    v_ma = s_vol.rolling(20).mean().bfill().values
    df['volume_sma_20'] = np.round(v_ma, 4)
    df['volume_ratio'] = np.round(vol / (v_ma + 1e-9), 2)
    
    # 6. Cyclical temporal encodings (Positional for ViT / Transformer)
    dt_series = pd.to_datetime(df['date'])
    hours = dt_series.dt.hour + dt_series.dt.minute / 60.0
    df['hour_sin'] = np.round(np.sin(2 * np.pi * hours / 24.0), 4)
    df['hour_cos'] = np.round(np.cos(2 * np.pi * hours / 24.0), 4)
    
    return df

def main():
    print("="*70)
    print("🚀 GENERATING 1 GB CRYPTO TIME-SERIES BENCHMARK DATASET")
    print(f"📁 Target Output: {OUT_DIR}")
    print("="*70)
    
    t_start = time.time()
    
    # 2 Full Years of 1-minute data = 1,051,200 intervals
    n_points = 1051200
    dates_1m = pd.date_range('2023-01-01 00:00:00', periods=n_points, freq='1min')
    print(f"Time Range: {dates_1m[0]} -> {dates_1m[-1]} ({n_points:,} 1m rows per asset)\n")
    
    # 1. BTCUSDT 1m Raw
    print("1/6 Generating BTCUSDT 1m Raw (~115 MB)...")
    btc_prices = generate_price_path(n_points, start_price=16500.0, end_price=96500.0, vol_annual=0.55, seed=101)
    df_btc = generate_ohlcv_dataframe(dates_1m, btc_prices, base_vol=2.5, seed=101)
    p_btc = os.path.join(OUT_DIR, "BTCUSDT_1m_high_frequency.csv")
    df_btc.to_csv(p_btc, index=False)
    print(f"   Saved: {p_btc} ({os.path.getsize(p_btc)/1024/1024:.2f} MB)")
    
    # 2. BTCUSDT 1m Features (26 columns ~280 MB)
    print("2/6 Generating BTCUSDT 1m Feature-Engineered (~280 MB)...")
    df_btc_feat = generate_features_dataframe(df_btc)
    p_btc_feat = os.path.join(OUT_DIR, "BTCUSDT_1m_features_deep_learning.csv")
    df_btc_feat.to_csv(p_btc_feat, index=False)
    print(f"   Saved: {p_btc_feat} ({os.path.getsize(p_btc_feat)/1024/1024:.2f} MB)")
    
    # 3. ETHUSDT 1m Raw
    print("3/6 Generating ETHUSDT 1m Raw (~115 MB)...")
    eth_prices = generate_price_path(n_points, start_price=1200.0, end_price=3650.0, vol_annual=0.68, seed=202)
    df_eth = generate_ohlcv_dataframe(dates_1m, eth_prices, base_vol=25.0, seed=202)
    p_eth = os.path.join(OUT_DIR, "ETHUSDT_1m_high_frequency.csv")
    df_eth.to_csv(p_eth, index=False)
    print(f"   Saved: {p_eth} ({os.path.getsize(p_eth)/1024/1024:.2f} MB)")
    
    # 4. ETHUSDT 1m Features (~280 MB)
    print("4/6 Generating ETHUSDT 1m Feature-Engineered (~280 MB)...")
    df_eth_feat = generate_features_dataframe(df_eth)
    p_eth_feat = os.path.join(OUT_DIR, "ETHUSDT_1m_features_deep_learning.csv")
    df_eth_feat.to_csv(p_eth_feat, index=False)
    print(f"   Saved: {p_eth_feat} ({os.path.getsize(p_eth_feat)/1024/1024:.2f} MB)")
    
    # 5. SOLUSDT 1m Raw (~115 MB)
    print("5/6 Generating SOLUSDT 1m Raw (~115 MB)...")
    sol_prices = generate_price_path(n_points, start_price=9.8, end_price=210.0, vol_annual=0.88, seed=303)
    df_sol = generate_ohlcv_dataframe(dates_1m, sol_prices, base_vol=120.0, seed=303)
    p_sol = os.path.join(OUT_DIR, "SOLUSDT_1m_high_frequency.csv")
    df_sol.to_csv(p_sol, index=False)
    print(f"   Saved: {p_sol} ({os.path.getsize(p_sol)/1024/1024:.2f} MB)")
    
    # 6. BNBUSDT 1m Raw (~115 MB)
    print("6/6 Generating BNBUSDT 1m Raw (~115 MB)...")
    bnb_prices = generate_price_path(n_points, start_price=245.0, end_price=660.0, vol_annual=0.58, seed=404)
    df_bnb = generate_ohlcv_dataframe(dates_1m, bnb_prices, base_vol=18.0, seed=404)
    p_bnb = os.path.join(OUT_DIR, "BNBUSDT_1m_high_frequency.csv")
    df_bnb.to_csv(p_bnb, index=False)
    print(f"   Saved: {p_bnb} ({os.path.getsize(p_bnb)/1024/1024:.2f} MB)")
    
    # 7. Copy original 15m and multi-timeframes for completeness
    src_custom = os.path.join(BASE_DIR, "datasets", "custom", "BTCUSDT_15m_calf.csv")
    if os.path.exists(src_custom):
        p_15m = os.path.join(OUT_DIR, "BTCUSDT_15m_calf.csv")
        import shutil
        shutil.copy2(src_custom, p_15m)
        print(f"   Copied: BTCUSDT_15m_calf.csv ({os.path.getsize(p_15m)/1024/1024:.2f} MB)")
    
    src_1h = os.path.join(BASE_DIR, "datasets", "expanded", "BTCUSDT_1h_calf.csv")
    if os.path.exists(src_1h):
        import shutil
        shutil.copy2(src_1h, os.path.join(OUT_DIR, "BTCUSDT_1h_calf.csv"))
        shutil.copy2(os.path.join(BASE_DIR, "datasets", "expanded", "BTCUSDT_1d_calf.csv"), os.path.join(OUT_DIR, "BTCUSDT_1d_calf.csv"))
    
    total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)) / (1024*1024)
    total_gb = total_mb / 1024.0
    elapsed = round(time.time() - t_start, 1)
    
    print("\n" + "="*70)
    print(f"🎉 1 GB BENCHMARK SUITE SUCCESSFULLY CREATED IN {elapsed}s!")
    print(f"📦 Total Files: {len(os.listdir(OUT_DIR))}")
    print(f"📊 Total Size : {total_gb:.2f} GB ({total_mb:.1f} MB)")
    print(f"📁 Directory  : {OUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
