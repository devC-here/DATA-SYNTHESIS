import pandas as pd

# Load dataset
df = pd.read_csv("trade_decisions2.csv")

# Count occurrences of each trade decision
decision_counts = df['trade_decision'].value_counts()

# Map trade decisions to numerical values if not already mapped
decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# Recalculate after mapping
decision_counts_numeric = df['trade_decision'].value_counts().sort_index()

# Compute percentages
total_samples = len(df)
decision_percentages = (decision_counts_numeric / total_samples) * 100

# Check for missing values
missing_values = df.isnull().sum()

# Summary statistics of numerical features
feature_stats = df.describe()

# Print results
print("Dataset Composition Analysis\n" + "="*40)
print("Class Distribution:")
for label, count in decision_counts_numeric.items():
    decision_label = list(decision_map.keys())[list(decision_map.values()).index(label)]
    print(f"{decision_label}: {count} samples ({decision_percentages[label]:.2f}%)")
print("\nMissing Values Per Column:\n", missing_values)
print("\nFeature Statistics:\n", feature_stats)
