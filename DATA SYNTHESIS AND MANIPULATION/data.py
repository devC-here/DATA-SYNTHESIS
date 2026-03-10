import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_tlw_data(datafile, outputfile, ohlc_variation=0.0002, high_low_variation=0.0003):
    # Load dataset
    df = pd.read_csv(datafile)
    
    # Normalize column names and convert to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure required OHLC and trade decision columns exist
    required_columns = ['open', 'high', 'low', 'close', 'trade_decision']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Dataset must contain OHLC and trade_decision columns.")
    
    # Introduce OHLC variations for 'open' and 'close'
    for col in tqdm(['open', 'close'], desc="Modifying OHLC"):
        variation = df[col] * np.random.uniform(-ohlc_variation, ohlc_variation, size=len(df))
        df[col] = (df[col] + variation).round(4)  # Round to 4 decimal places
    
    # Ensure logical candlestick structure for high and low
    df['high'] = np.maximum(df[['open', 'close']].max(axis=1) + np.abs(df['high'] * high_low_variation), df['high'])
    df['low'] = np.minimum(df[['open', 'close']].min(axis=1) - np.abs(df['low'] * high_low_variation), df['low'])
    
    # Safe division function with clipping
    def safe_divide(a, b, eps=1e-6, clip_value=10):
        result = np.where(np.abs(b) > eps, a / b, 0)
        return np.clip(result, -clip_value, clip_value).round(3)  # Clipped to prevent extreme values
    
    # Compute candlestick shape features
    df['wick_to_length_ratio'] = safe_divide(df['high'] - df[['open', 'close']].max(axis=1), df['high'] - df['low'])
    df['body_to_length_ratio'] = safe_divide(np.abs(df['close'] - df['open']), df['high'] - df['low'])
    df['tail_to_length_ratio'] = safe_divide(df[['open', 'close']].min(axis=1) - df['low'], df['high'] - df['low'])
    
    # Keep only the selected features along with trade decisions
    df = df[['tail_to_length_ratio', 'body_to_length_ratio', 'wick_to_length_ratio', 'trade_decision']]
    
    # Shuffle the dataset to introduce randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the modified dataset
    df.to_csv(outputfile, index=False)
    print(f"T/L, B/L, W/L dataset with trade decisions saved as {outputfile}")

# Run the function
generate_tlw_data("sell_candles.csv", "sell_data.csv")
generate_tlw_data("no trade_candles.csv", "no trade_data.csv")
generate_tlw_data("buy_candles.csv", "buy_data.csv")
