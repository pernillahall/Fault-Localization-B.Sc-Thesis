import pandas as pd
import numpy as np
import random
import os
import sys
import ast
import joblib
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Configuration
INPUT_DATA_PATH = 'bug_dataset_preprocessed.csv' # <-- the csv file must contain columns: 'processed_text' and 'label_list'
OUTPUT_DIR = "data/prepared_data"
OUTPUT_TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_df.csv')
OUTPUT_VAL_FILE = os.path.join(OUTPUT_DIR, 'val_df.csv')
OUTPUT_TEST_FILE = os.path.join(OUTPUT_DIR, 'test_df.csv')
MLB_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'mlb.joblib')
Y_TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'y_train_bin.npy')
Y_VAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'y_val_bin.npy')
Y_TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'y_test_bin.npy')

SEED = 42
TEST_SET_SIZE = 0.1 
VALIDATION_SET_SIZE = 0.2 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

# Load the dataset
try:
    df = pd.read_csv(INPUT_DATA_PATH)
    required_columns = ['processed_text', 'label_list']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")
    
    # Convert 'label_list' from string representation of list to actual list
    if not df.empty and isinstance(df['label_list'].iloc[0], str):
        df['label_list'] = df['label_list'].apply(ast.literal_eval)

    df['processed_text'] = df['processed_text'].fillna('')

    initial_bug_count = len(df)
    df = df[df['label_list'].apply(len) > 0]
    if len(df) < initial_bug_count:
        print(f"Removed {initial_bug_count - len(df)} rows with empty label lists.")

    df['original_index'] = df.index  # Store original index for later reference

    print(f"Successfully loaded data from {INPUT_DATA_PATH}")

except FileNotFoundError:
    print(f"Error: File not found at {INPUT_DATA_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sys.exit(1)

if df.empty:
    print("Error: The dataset is empty.")
    sys.exit(1)

# Perform data splitting
print("Starting data splitting...")
train_df, val_df, test_df = None, None, None

try:
    # Temporarily binarize labels for guiding the stratification split
    unique_labels = sorted(list(set(label for sublist in df['label_list'] for label in sublist)))
    if len(unique_labels) == 0:
        raise ValueError("No unique labels found in the dataset.")
    
    temp_mlb_split = MultiLabelBinarizer(classes=unique_labels)
    temp_mlb_split.fit([unique_labels])
    y_binary = temp_mlb_split.transform(df['label_list'])

    # Reshape for compatibility with iterative_train_test_split
    X_indexes = df['original_index'].values.reshape(-1, 1)  
    if X_indexes.shape[0] != y_binary.shape[0]:
        raise ValueError("Mismatch between number of samples in indexes and labels.")

    # Split the data into train, validation, and test sets
    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(X_indexes, y_binary, test_size=TEST_SET_SIZE)
    val_relative_size = VALIDATION_SET_SIZE / (1 - TEST_SET_SIZE)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size=val_relative_size)

    # Extract indices for train, validation, and test sets
    train_indices = X_train[:, 0]
    val_indices = X_val[:, 0]
    test_indices = X_test[:, 0]

    # Convert back to DataFrame
    train_df = df.set_index('original_index').loc[train_indices].reset_index(drop=True)
    val_df = df.set_index('original_index').loc[val_indices].reset_index(drop=True)
    test_df = df.set_index('original_index').loc[test_indices].reset_index(drop=True)

    print(f"Data split completed.")

except Exception as e:
    print(f"Error during data splitting: {e}")
    sys.exit(1)

if train_df.empty or val_df.empty or test_df.empty:
    print("Error: One or more splits are empty.")
    sys.exit(1)

print(f"Final split sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# Save binary labels for train, validation, and test sets
print("Preparing and saving labels...")
try:
    # Fit MLB on final training data labels
    all_train_labels = sorted(list(set(label for sublist in train_df['label_list'] for label in sublist)))
    if len(all_train_labels) == 0:
        raise ValueError("No unique labels found in the training dataset.")
    print(f"Unique labels in training set: {len(all_train_labels)}")

    mlb = MultiLabelBinarizer(classes=all_train_labels)
    mlb.fit(train_df['label_list'])

    # Save fitted MLB for later use
    joblib.dump(mlb, MLB_OUTPUT_FILE)

    # Transform labels for all splits using the train-fitted mlb
    y_train_bin = mlb.transform(train_df['label_list'])
    y_val_bin = mlb.transform(val_df['label_list'])
    y_test_bin = mlb.transform(test_df['label_list'])

    # Save binary label arrays
    np.save(Y_TRAIN_OUTPUT_PATH, y_train_bin)
    np.save(Y_VAL_OUTPUT_PATH, y_val_bin)
    np.save(Y_TEST_OUTPUT_PATH, y_test_bin)

    print(f"Binary labels saved to {Y_TRAIN_OUTPUT_PATH}, {Y_VAL_OUTPUT_PATH}, {Y_TEST_OUTPUT_PATH}")
except Exception as e:
    print(f"Error saving binary labels: {e}")
    sys.exit(1)

print("Saving splits to CSV files...")

# Sava split data frames
try:
    train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
    val_df.to_csv(OUTPUT_VAL_FILE, index=False)
    test_df.to_csv(OUTPUT_TEST_FILE, index=False)
    print(f"Train split saved to {OUTPUT_TRAIN_FILE}")
    print(f"Validation split saved to {OUTPUT_VAL_FILE}")
    print(f"Test split saved to {OUTPUT_TEST_FILE}")
except Exception as e:
    print(f"Error saving CSV files: {e}")
    sys.exit(1)

print("Data splitting completed successfully.")
print("All files saved successfully.")
print("Preprocessing and splitting done. Ready for model training.")






