import pandas as pd
import numpy as np
import random
import os
import sys
import time
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from functools import partial

import eval_utils

SUFFIX = '_SR'
SEED = 42 # Random seed for reproducibility
PREPARED_DATA_DIR = "data/prepared_data"
INPUT_TRAIN_DF_PATH = os.path.join(PREPARED_DATA_DIR, f'train_df{SUFFIX}.csv')
INPUT_VAL_DF_PATH = os.path.join(PREPARED_DATA_DIR, f'val_df.csv')
INPUT_TEST_DF_PATH = os.path.join(PREPARED_DATA_DIR, f'test_df.csv')
MLB_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, f'mlb{SUFFIX}.joblib')
Y_TRAIN_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, f'y_train_bin{SUFFIX}.npy')
Y_VAL_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, f'y_val_bin{SUFFIX}.npy')
Y_TEST_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, f'y_test_bin{SUFFIX}.npy')

OUTPUT_DIR = f"results/rf_grid_search_tfidf{SUFFIX}"
CLASSIFIER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'rf_binary_relevance_model.joblib')
TFIDF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer_rf_baseline.joblib')
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rf_classification_report.txt")
RANKING_METRICS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rf_validation_ranking_metrics.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CV_FOLDS = 3 # Number of cross-validation folds

TFIDF_MAX_FEATURES_OPTIONS = [1000, 5000] # Limit vocabulary size -> TO TUNE
TFIDF_NGRAM_RANGE_OPTIONS = [(1, 1), (1, 2)] # Use unigrams and bigrams -> TO TUNE
TF_MIN_DF_OPTIONS = [2] # Minimum document frequency for TF-IDF -> TO TUNE

RF_N_ESTIMATORS_OPTIONS = [5, 10, 20, 100] # Number of trees -> TO TUNE
RF_MAX_DEPTH_OPTIONS = [None, 20, 100] # Maximum depth of trees -> TO TUNE
RF_MIN_SAMPLES_SPLIT_OPTIONS = [2, 5, 10] # Minimum number of samples required to split an internal node -> TO TUNE
RF_MIN_SAMPLES_LEAF_OPTIONS = [2, 5, 10, 20] # Minimum number of samples required to be at a leaf node -> TO TUNE
RF_CLASS_WEIGHT_OPTIONS = ['balanced_subsample'] # Helps with imbalance within each binary classifier

RANKING_K_VALUES = [1, 3, 5, 10] # K values for ranking evaluation

print("setting random seed to", SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load Preprocessed Data 
script_start_time = time.time()
print(f"\nLoading pre-split data and prepared labels")
try:
    train_df = pd.read_csv(INPUT_TRAIN_DF_PATH)
    val_df = pd.read_csv(INPUT_VAL_DF_PATH)
    test_df = pd.read_csv(INPUT_TEST_DF_PATH)
    mlb = joblib.load(MLB_LOAD_PATH)
    num_unique_labels = len(mlb.classes_)
    print(f"Number of labels model will predict: {num_unique_labels}")
    print(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples.")

    # Load binarized labels
    y_train_bin = np.load(Y_TRAIN_LOAD_PATH).astype(np.float32)
    y_val_bin = np.load(Y_VAL_LOAD_PATH).astype(np.float32)
    y_test_bin = np.load(Y_TEST_LOAD_PATH).astype(np.float32)

    if not all([len(train_df) == y_train_bin.shape[0],
                   len(val_df) == y_val_bin.shape[0],
                   len(test_df) == y_test_bin.shape[0],
                   y_train_bin.shape[1] == y_val_bin.shape[1] == y_test_bin.shape[1] == num_unique_labels]):
        raise ValueError("Mismatch between DataFrame and binarized label sizes.")

    print(f"Loaded datasets: Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print(f"Binarized Labels: Train: {y_train_bin.shape}, Val: {y_val_bin.shape}, Test: {y_test_bin.shape}")
except FileNotFoundError:
    print(f"Error: File not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading or processing data: {e}")
    sys.exit(1)

def top_k_accuracy(y_true, y_pred, k=3, needs_proba=False):
    ranked_indices = np.argsort(-y_pred, axis=1)
    num_samples = len(y_true)
    hits_at_k = 0

    if num_samples == 0:
        return 0.0

    for i in range(num_samples):
        true_positive_indices = np.where(y_true[i] == 1)[0]
        if len(true_positive_indices) > 0:
            top_k_preds = ranked_indices[i, :k]
            top_k_preds = top_k_preds[top_k_preds != -1]
            if len(top_k_preds) > 0:
                hits = np.intersect1d(true_positive_indices, top_k_preds)
                if len(hits) > 0:
                    hits_at_k += 1

    return hits_at_k / num_samples if num_samples > 0 else 0.0

def mean_average_precision(y_true_bin, y_pred, needs_proba=False):
    ranked_indices = np.argsort(-y_pred, axis=1)
    average_precisions = []

    for i in range(len(y_true_bin)):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        if len(true_positive_indices) == 0:
            continue  # No relevant labels for this sample

        ap = 0.0
        num_relevant = len(true_positive_indices)
        
        # For each relevant label, calculate precision at the rank position where it appears
        for rank, predicted_idx in enumerate(ranked_indices[i]):
            if predicted_idx in true_positive_indices:
                # Precision at this rank
                ap += len(np.intersect1d(true_positive_indices, ranked_indices[i, :rank+1])) / (rank+1)

        # Normalize by the number of relevant labels
        ap /= num_relevant
        average_precisions.append(ap)

    # Return the Mean Average Precision (MAP)
    return np.mean(average_precisions) if average_precisions else 0.0


# k_to_optimize = 3 
# custom_scorer = make_scorer(
#     partial(top_k_accuracy, k=k_to_optimize), 
#     greater_is_better=True, 
#     needs_proba=True
# )

custom_scorer = make_scorer(
    mean_average_precision, 
    greater_is_better=True, 
    needs_proba=True
)

# define pipeline and parameter grid
print("Defining pipeline and parameter grid...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1
    )))
])

param_grid = {
    'tfidf__max_features': TFIDF_MAX_FEATURES_OPTIONS,
    'tfidf__ngram_range': TFIDF_NGRAM_RANGE_OPTIONS,
    'tfidf__min_df': TF_MIN_DF_OPTIONS,
    'clf__estimator__n_estimators': RF_N_ESTIMATORS_OPTIONS,
    'clf__estimator__max_depth': RF_MAX_DEPTH_OPTIONS,
    'clf__estimator__min_samples_split': RF_MIN_SAMPLES_SPLIT_OPTIONS,
    'clf__estimator__min_samples_leaf': RF_MIN_SAMPLES_LEAF_OPTIONS,
    'clf__estimator__class_weight': RF_CLASS_WEIGHT_OPTIONS
}

print("Parameter grid defined.")
print(json.dumps(param_grid, indent=2))

# perform grid search with cross validaton
# fits on train data and use internal cross-validation to find best hyperparameters

print("Performing grid search with cross-validation...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=custom_scorer,
    cv=CV_FOLDS,
    verbose=2,
    n_jobs=-1
)

try:
    grid_search.fit(train_df['processed_text'], y_train_bin)
    print("Grid search completed.")
except Exception as e:
    print(f"Error during grid search: {e}")
    sys.exit(1)

# show best parameters and score
print("Best parameters found:")
print(grid_search.best_params_)

# save the best model
best_model = grid_search.best_estimator_

# Prediction on Validation Set
print("\nMaking predictions on the validation set...")
try:
    y_pred_bin = best_model.predict(val_df['processed_text'])
    y_pred_proba = best_model.predict_proba(val_df['processed_text'])
    print("Predictions generated.")
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# Evaluation on Validation set
print("\n--- Evaluating Model on Validation Set ---")
_ = eval_utils.evaluate_model_predictions(
    y_true_bin=y_val_bin,
    y_pred_bin=y_pred_bin,
    y_pred_scores=y_pred_proba,
    mlb=mlb,
    k_values=RANKING_K_VALUES,
    output_dir=OUTPUT_DIR,
    model_name="Random-Forest",
)

# Save Model & Vectorizer & Params
print(f"\nSaving trained objects...")
try:
    joblib.dump(best_model, CLASSIFIER_OUTPUT_PATH)
    print(f"  Classifier (pipeline) saved to {CLASSIFIER_OUTPUT_PATH}")
    
    tfidf_vectorizer = best_model.named_steps['tfidf']
    joblib.dump(tfidf_vectorizer, TFIDF_OUTPUT_PATH)
    print(f"  TF-IDF Vectorizer saved to {TFIDF_OUTPUT_PATH}")
    
    best_params_path = os.path.join(OUTPUT_DIR, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"  Best hyperparameters saved to {best_params_path}")
    
    config = {
        "random_seed": SEED,
        "cv_folds": CV_FOLDS,
        "param_grid": param_grid
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    print("  Config saved to config.json")

except Exception as e:
    print(f"Error saving objects: {e}")


script_end_time = time.time()
print(f"\n--- Full Random Forest Script Completed ---")
print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")
