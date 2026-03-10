import pandas as pd
import numpy as np
import ta

def generate_synthetic_values(df):
    df['close'] = np.random.uniform(1800, 2000, len(df))
    df['high'] = df['close'] + np.random.uniform(1, 10, len(df))
    df['low'] = df['close'] - np.random.uniform(1, 10, len(df))
    df['open'] = df['close'] + np.random.uniform(-5, 5, len(df))
    df['volume'] = np.random.uniform(1000, 5000, len(df))
    return df

def calculate_indicators(df):
    df = df.copy()
    
    # Moving Average (MA)
    df['ma_14'] = ta.trend.sma_indicator(df['close'], window=14)
    
    # MACD
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_hist'] = ta.trend.macd_diff(df['close'])
    
    # Ensure MACD is not NaN
    df[['macd', 'macd_signal', 'macd_hist']] = df[['macd', 'macd_signal', 'macd_hist']].fillna(0)
    
    # Parabolic SAR
    df['sar'] = ta.trend.psar_up(df['high'], df['low'], df['close'])
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['upper_band'] = bb.bollinger_hband()
    df['middle_band'] = bb.bollinger_mavg()
    df['lower_band'] = bb.bollinger_lband()
    
    # Relative Strength Index (RSI)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    return df

def generate_synthetic_data(buy_file, sell_file, no_trade_file, output_file, total_samples=500000):
    buy_df = pd.read_csv(buy_file)
    sell_df = pd.read_csv(sell_file)
    no_trade_df = pd.read_csv(no_trade_file)
    
    buy_df = generate_synthetic_values(buy_df)
    sell_df = generate_synthetic_values(sell_df)
    no_trade_df = generate_synthetic_values(no_trade_df)
    
    buy_df = calculate_indicators(buy_df)
    sell_df = calculate_indicators(sell_df)
    no_trade_df = calculate_indicators(no_trade_df)
    
    for df in [buy_df, sell_df, no_trade_df]:
        df['tail_to_length_ratio_2'] = df['tail_to_length_ratio'].shift(1)
        df['tail_to_length_ratio_3'] = df['tail_to_length_ratio'].shift(2)
        df['wick_to_length_ratio_2'] = df['wick_to_length_ratio'].shift(1)
        df['wick_to_length_ratio_3'] = df['wick_to_length_ratio'].shift(2)
        df['body_to_length_ratio_2'] = df['body_to_length_ratio'].shift(1)
        df['body_to_length_ratio_3'] = df['body_to_length_ratio'].shift(2)
    
    buy_size = int(total_samples * 0.32)
    sell_size = int(total_samples * 0.32)
    no_trade_size = total_samples - (buy_size + sell_size)
    
    def generate_samples(df, condition, target_size, label):
        filtered_df = df[condition].copy()
        if len(filtered_df) >= target_size:
            selected_samples = filtered_df.sample(n=target_size, random_state=42)
        else:
            synthetic_samples = filtered_df.sample(n=target_size - len(filtered_df), replace=True, random_state=42)
            numeric_cols = synthetic_samples.select_dtypes(include=[np.number]).columns
            synthetic_samples[numeric_cols] *= np.random.uniform(0.98, 1.02, size=synthetic_samples[numeric_cols].shape)
            selected_samples = pd.concat([filtered_df, synthetic_samples], ignore_index=True)
        
        if label == 'Buy':
            selected_samples = selected_samples[selected_samples['rsi'] < 70]
        elif label == 'Sell':
            selected_samples = selected_samples[selected_samples['rsi'] > 30]
        elif label == 'No Trade':
            selected_samples = selected_samples[(selected_samples['rsi'] > 40) & (selected_samples['rsi'] < 60)]
        
        selected_samples['trade_decision'] = label
        return selected_samples
    
    conditions_buy = (
        ((buy_df['tail_to_length_ratio'] > 0.50) & (buy_df['tail_to_length_ratio'] > buy_df['body_to_length_ratio']) & (buy_df['body_to_length_ratio'] > buy_df['wick_to_length_ratio'])) |
        ((buy_df['tail_to_length_ratio'] > buy_df['wick_to_length_ratio']) & (buy_df['wick_to_length_ratio'] > buy_df['body_to_length_ratio']) & (buy_df['tail_to_length_ratio'] > 0.75))
    )
    
    conditions_sell = (
        ((sell_df['wick_to_length_ratio'] > 0.50) & (sell_df['wick_to_length_ratio'] > sell_df['body_to_length_ratio']) & (sell_df['body_to_length_ratio'] > sell_df['tail_to_length_ratio'])) |
        ((sell_df['wick_to_length_ratio'] > sell_df['tail_to_length_ratio']) & (sell_df['tail_to_length_ratio'] > sell_df['body_to_length_ratio']) & (sell_df['wick_to_length_ratio'] > 0.75))
    )
    
    conditions_no_trade = (
        (np.abs(no_trade_df['wick_to_length_ratio'] - no_trade_df['tail_to_length_ratio']) <= 0.10) & 
        (np.abs(no_trade_df['tail_to_length_ratio'] - no_trade_df['body_to_length_ratio']) <= 0.10) & 
        (np.abs(no_trade_df['wick_to_length_ratio'] - no_trade_df['body_to_length_ratio']) <= 0.10) &
        (no_trade_df['wick_to_length_ratio'] < 0.50) & 
        (no_trade_df['tail_to_length_ratio'] < 0.50) & 
        (no_trade_df['body_to_length_ratio'] < 0.50)
    )
    
    buy_selected = generate_samples(buy_df, conditions_buy, buy_size, 'Buy')
    sell_selected = generate_samples(sell_df, conditions_sell, sell_size, 'Sell')
    no_trade_selected = generate_samples(no_trade_df, conditions_no_trade, no_trade_size, 'No Trade')
    
    final_df = pd.concat([buy_selected, sell_selected, no_trade_selected], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(output_file, index=False)
    print(f"Generated dataset saved as {output_file}")

generate_synthetic_data('buy_data.csv', 'sell_data.csv', 'no trade_data.csv', 'synthetic_trade_data.csv', total_samples=500000)
