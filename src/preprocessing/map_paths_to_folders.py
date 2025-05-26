import pandas as pd
from collections import Counter
import sys

print("Starting data processing and label filtering...")

# Configuration
INPUT_FILE_INFO_PATH = "paths_to_folders.xlsx"
INPUT_BUGS_PATH = "extracted_bugs.xlsx"
OUTPUT_CSV_PATH = "bug_dataset.csv"
MIN_LABEL_FREQUENCY = 10

# Load data
try:
    print(f"Loading file info from '{INPUT_FILE_INFO_PATH}'...")
    file_info_df = pd.read_excel(INPUT_FILE_INFO_PATH)
    print(f"Loading bug reports from '{INPUT_BUGS_PATH}'...")
    bugs_df = pd.read_excel(INPUT_BUGS_PATH)
    print("Data loaded successfully.")
except FileNotFoundError as e: 
    print(f"Error loading files: {e}")
    sys.exit(1)
except Exception as e: 
    print(f"Error during file loading: {e}")
    sys.exit(1)


# Preprocessing and initial label extraction
print("Normalizing paths and extracting initial labels...")
def normalize_path(path): 
    if not isinstance(path, str): return ""
    return path.strip().lower().replace("\\", "/")

def extract_labels(row, path_map): 
    paths_str = row.get('Paths', '')
    if pd.isna(paths_str): paths_str = ''
    paths = str(paths_str).split(',')
    normalized = [normalize_path(p) for p in paths if p]
    subfolders = set()
    for path in normalized:
        if path in path_map and pd.notna(path_map[path]) and str(path_map[path]).lower() != 'x':
            subfolders.add(str(path_map[path]))
    return sorted(list(subfolders))

file_info_df['File'] = file_info_df['File'].apply(normalize_path)
file_to_subfolder = dict(zip(file_info_df['File'], file_info_df['Label']))
bugs_df['Labels'] = bugs_df.apply(lambda row: extract_labels(row, file_to_subfolder), axis=1)
print("Initial label extraction complete.")


# Initial filtering based on label count per bug
print("Applying initial filters (1 to 5 labels per bug)...")
initial_bug_count = len(bugs_df)
bugs_df = bugs_df[bugs_df['Labels'].map(len).between(1, 5, inclusive='both')].reset_index(drop=True)
print(f"Filtered bugs based on label count: {initial_bug_count} -> {len(bugs_df)} rows remaining.")
if len(bugs_df) == 0: 
    print("Error: No bugs remaining after initial filtering.")
    sys.exit(1)


# Calculate overall label frequencies
print("Calculating overall label frequencies...")
all_labels_before_freq_filter = [label for sublist in bugs_df['Labels'] for label in sublist]
label_counts = Counter(all_labels_before_freq_filter)
original_unique_label_count = len(label_counts)
print(f"Found {original_unique_label_count} unique labels before frequency filtering.")

# Define valid labels based on frequency
print(f"\nDefining valid labels (frequency >= {MIN_LABEL_FREQUENCY})...")
valid_labels = {label for label, count in label_counts.items() if count >= MIN_LABEL_FREQUENCY}
final_unique_label_count = len(valid_labels)
if final_unique_label_count == 0:
    print(f"Warning: No labels meet the minimum frequency threshold of {MIN_LABEL_FREQUENCY}. Output will be empty.")
else:
    print(f"Identified {final_unique_label_count} labels to keep (out of {original_unique_label_count}).")


# Drop rare labels from each bug report
def drop_rare_labels(label_list):
    if isinstance(label_list, list):
        return [label for label in label_list if label in valid_labels]
    return []

bugs_df['Labels'] = bugs_df['Labels'].apply(drop_rare_labels)
print("Applied frequency filter to 'Labels' column.")

# Remove any bugs with no labels after filtering
print("Removing rows where all labels were filtered out...")
initial_bug_count_after_label_filter = len(bugs_df)
bugs_df = bugs_df[bugs_df['Labels'].map(len) > 0].reset_index(drop=True)
final_bug_count = len(bugs_df)
print(f"Filtered rows with empty labels: {initial_bug_count_after_label_filter} -> {final_bug_count} rows remaining.")

if final_bug_count == 0:
    print(f"Error: No bugs remaining after filtering labels with frequency < {MIN_LABEL_FREQUENCY}.")


# Final frequency check
print("\nRecalculating frequencies on final filtered data...")
all_labels_after_freq_filter = [label for sublist in bugs_df['Labels'] for label in sublist]
if all_labels_after_freq_filter:
    final_label_counts = Counter(all_labels_after_freq_filter)
    final_label_freq_df = pd.DataFrame(final_label_counts.items(), columns=['Label', 'Frequency'])
    final_label_freq_df.sort_values(by='Frequency', ascending=False, inplace=True)
    print(f"Final unique label count in data: {len(final_label_freq_df)}")
    print("Top 20 Label Frequencies (after filtering):")
    print(final_label_freq_df.head(20).to_string())
else:
    print("No labels remaining in the data after filtering.")

# Save filtered data
print(f"\nSaving final processed data to '{OUTPUT_CSV_PATH}'...")
try:
    existing_cols = bugs_df.columns.tolist()
    columns_to_save = ['Date', 'Bug ID', 'Bug Title', 'Priority', 'Description', 'Paths', 'Labels', 'processed_text']
    columns_to_save_final = [col for col in columns_to_save if col in existing_cols]
    bugs_df.to_csv(OUTPUT_CSV_PATH, columns=columns_to_save_final, index=False, encoding='utf-8')
    print("Data saved successfully.")
except Exception as e: print(f"Error saving data to CSV: {e}")

print("\nScript finished.")