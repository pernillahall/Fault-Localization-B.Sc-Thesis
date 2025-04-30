import pandas as pd
import numpy as np
import random
import os
import sys
import time
import joblib
from ..evaluation import eval_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration ---
print("Configuring environment for Linear SVM baseline...")
PREPARED_DATA_DIR = "data/prepared_data"
INPUT_TRAIN_DF_PATH = os.path.join(PREPARED_DATA_DIR, 'train_df.csv')
INPUT_VAL_DF_PATH = os.path.join(PREPARED_DATA_DIR, 'val_df.csv')
INPUT_TEST_DF_PATH = os.path.join(PREPARED_DATA_DIR, 'test_df.csv')
MLB_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, 'mlb.joblib')
Y_TRAIN_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, 'y_train_bin.npy')
Y_VAL_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, 'y_val_bin.npy')
Y_TEST_LOAD_PATH = os.path.join(PREPARED_DATA_DIR, 'y_test_bin.npy')

OUTPUT_DIR = "results/svm_validation"
CLASSIFIER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'svm_binary_relevance_model.joblib')
TFIDF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer_svm_baseline.joblib')
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "svm_classification_report.txt")
RANKING_METRICS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "svm_validation_ranking_metrics.txt")

SEED = 42

# TF-IDF Configuration
MAX_FEATURES = 10000 # Limit vocabulary size -> TO TUNE
NGRAM_RANGE = (1, 2) # Use unigrams and bigrams -> TO TUNE

# LinearSVC Configuration
SVC_C = 1.0 # Regularization parameter (inverse); smaller values specify stronger regularization -> TO TUNE
SVC_LOSS = 'squared_hinge' # Standard SVM loss, or 'hinge'
SVC_PENALTY = 'l2' # Standard regularization
SVC_DUAL = 'auto' # Or True (default) or False. False often faster when n_features > n_samples if supported by loss/penalty. 'auto' is new default.
SVC_CLASS_WEIGHT = 'balanced' # Helps with imbalance within each binary classifier
SVC_MAX_ITER = 2000 # Increase if convergence warnings appear

# Ranking Evaluation Configuration
RANKING_K_VALUES = [1, 3, 5, 10]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Setting random seed to {SEED}")
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

# Feature Extraction (TF-IDF)
print("\nExtracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE
)
try:
    print(f"Fitting TF-IDF on training data (max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE})...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_text'])
    print("Transforming validation and test data...")
    X_val_tfidf = tfidf_vectorizer.transform(val_df['processed_text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])
    print(f"TF-IDF transformation complete. Shape of Train TF-IDF: {X_train_tfidf.shape}")
except Exception as e:
    print(f"Error during TF-IDF feature extraction: {e}")
    sys.exit(1)

# Model Training (Binary Relevance with Linear SVM)
print("\nTraining Linear SVM model using OneVsRestClassifier...")
print(f"Using C={SVC_C}, loss='{SVC_LOSS}', penalty='{SVC_PENALTY}', dual='{SVC_DUAL}', class_weight='{SVC_CLASS_WEIGHT}'")
base_svm = LinearSVC(
    C=SVC_C,
    loss=SVC_LOSS,
    penalty=SVC_PENALTY,
    dual=SVC_DUAL,
    class_weight=SVC_CLASS_WEIGHT,
    random_state=SEED,
    max_iter=SVC_MAX_ITER
)
# Wrap with OneVsRestClassifier: trains one LinearSVC per label
classifier = OneVsRestClassifier(base_svm, n_jobs=-1)

try:
    training_start_time = time.time()
    classifier.fit(X_train_tfidf, y_train_bin)
    training_end_time = time.time()
    print(f"Training complete. Time taken: {training_end_time - training_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during model training: {e}")
    sys.exit(1)

# Prediction on Validation Set
print("\nMaking predictions on the validation set using Linear SVM...")
try:
    # Binary predictions (based on default thresholding within LinearSVC)
    y_pred_bin = classifier.predict(X_val_tfidf)

    # Scores for ranking (distance to hyperplane)
    # Higher values mean more confident positive prediction.
    y_pred_scores = classifier.decision_function(X_val_tfidf)
    print("Predictions and decision scores generated.")
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# Evaluation on Validation set
print("\n--- Evaluating Model on Validation Set ---")
_ = eval_utils.evaluate_model_predictions(
    y_true_bin=y_val_bin,
    y_pred_bin=y_pred_bin,
    y_pred_scores=y_pred_scores,
    mlb=mlb,
    k_values=RANKING_K_VALUES,
    output_dir=OUTPUT_DIR,
    model_name="SVM-Linear",
)

# Save Model & Objects
print(f"\nSaving trained objects...")
try:
    joblib.dump(classifier, CLASSIFIER_OUTPUT_PATH)
    joblib.dump(tfidf_vectorizer, TFIDF_OUTPUT_PATH)
    print(f"  Classifier saved to {CLASSIFIER_OUTPUT_PATH}")
    print(f"  TF-IDF Vectorizer saved to {TFIDF_OUTPUT_PATH}")
except Exception as e: print(f"Error saving objects: {e}")

script_end_time = time.time()
print(f"\n--- Full Linear SVM Script with Stratified Split Completed ---")
print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")