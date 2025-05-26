import pandas as pd
import ast
from collections import Counter

# Load Excel file
df = pd.read_csv("data/bug_dataset.csv")

# Parse stringified lists into real Python lists
df["Labels"] = df["Labels"].apply(ast.literal_eval)

# Remove empty paths
df["Labels"] = df["Labels"].apply(lambda paths: [p for p in paths if p.strip() != ''])

# Count number of paths per bug
df["Num_Labels"] = df["Labels"].apply(len)

# Create distribution of labels per bug and print
path_distribution = Counter(df["Num_Labels"])
sorted_distribution = dict(sorted(path_distribution.items()))
print("Distribution of Labels per Bug:")
for num_paths, count in sorted_distribution.items():
    print(f"{num_paths} label(s): {count} bug(s)")

# Calculate the average number of paths per bug
average = df["Num_Labels"].mean()
print(f"\nAverage number of labels per bug: {average:.2f}")