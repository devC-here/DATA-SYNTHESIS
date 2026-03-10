import pandas as pd

def filter_trade_decisions(input_file, output_file, trade_type, num_rows):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Filter rows based on the trade decision type
    filtered_df = df[df['trade_decision'] == trade_type]
    
    # Select the specified number of rows
    selected_rows = filtered_df.head(num_rows)
    
    # Save to a new file
    selected_rows.to_csv(output_file, index=False)
    
    print(f"Saved {len(selected_rows)} rows of type '{trade_type}' to {output_file}")

# Example usage
input_file = "trade_decisions1.csv"
output_file = "filtered_trades2.csv"
trade_type = "Sell"  # Change to 'Sell' or 'No Trade' as needed
num_rows = 960  # Change to the number of rows you want to copy

filter_trade_decisions(input_file, output_file, trade_type, num_rows)
