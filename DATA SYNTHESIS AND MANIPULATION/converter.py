import pandas as pd

def convert_sell_to_buy(df):
    # Ensure all column names are in lowercase
    df.columns = df.columns.str.lower()

    # Identify sell candles
    sell_candles = df['trade_decision'].str.lower() == 'sell'

    # Swap wick and tail while keeping open and close unchanged
    df.loc[sell_candles, 'high'], df.loc[sell_candles, 'low'] = (
        (df.loc[sell_candles, 'open'] + (df.loc[sell_candles, 'open'] - df.loc[sell_candles, 'low'])).round(3),
        (df.loc[sell_candles, 'close'] - (df.loc[sell_candles, 'high'] - df.loc[sell_candles, 'close'])).round(3)
    )

    # Convert trade decision from Sell to Buy
    df.loc[sell_candles, 'trade_decision'] = 'Buy'
    
    return df

# Load dataset (ensure the CSV does not have extra spaces in column names)
df = pd.read_csv("filtered_trades2.csv")

# Convert Sell to Buy
df_transformed = convert_sell_to_buy(df)

# Save new dataset
df_transformed.to_csv("transformed_dataset.csv", index=False)

print("Conversion complete. Transformed dataset saved as 'transformed_dataset.csv'.")
