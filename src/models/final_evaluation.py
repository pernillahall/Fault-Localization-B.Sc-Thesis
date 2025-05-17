import pandas as pd
import numpy as np
import random
import os
import sys
import joblib
import json
import time

import eval_utils
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

print("Configuring final benchmark run")

PREPARED_DATA_DIR = "data/prepared_data"
OUTPUT_DIR = "results/final_benchmark"

MODEL_CONFIGS = {
    # LR
    "LR_TFIDF_OG": {
        "model_path": "results/lr/lr_grid_search_tfidf/lr_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_grid_search_tfidf/tfidf_vectorizer_lr_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR, "mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "TFIDF"
    },
    "LR_SBERT_OG": {
        "model_path": "results/lr/lr_sbert_grid_search/lr_sbert_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_sbert_grid_search/sbert_scaler_lr.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "SBERT"
    },
    "LR_TFIDF_RS": {
        "model_path": "results/lr/lr_grid_search_tfidf_RS/lr_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_grid_search_tfidf_RS/tfidf_vectorizer_lr_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "TFIDF"
    },
    "LR_SBERT_RS": {
        "model_path": "results/lr/lr_sbert_grid_search_RS/lr_sbert_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_sbert_grid_search_RS/sbert_scaler_lr.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "SBERT"
    },
    "LR_TFIDF_RS_underrep": {
        "model_path": "results/lr/lr_grid_search_tfidf_RS_underrep/lr_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_grid_search_tfidf_RS_underrep/tfidf_vectorizer_lr_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "TFIDF"
    },
    "LR_SBERT_RS_underrep": {
        "model_path": "results/lr/lr_sbert_grid_search_RS_underrep/lr_sbert_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_sbert_grid_search_RS_underrep/sbert_scaler_lr.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "SBERT"
    },
    "LR_TFIDF_SR": {
        "model_path": "results/lr/lr_grid_search_tfidf_SR/lr_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_grid_search_tfidf_SR/tfidf_vectorizer_lr_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "TFIDF"
    },
    "LR_SBERT_SR": {
        "model_path": "results/lr/lr_sbert_grid_search_SR/lr_sbert_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_sbert_grid_search_SR/sbert_scaler_lr.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "SBERT"
    },
    "LR_TFIDF_SR_underrep": {
        "model_path": "results/lr/lr_grid_search_tfidf_SR_underrep/lr_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_grid_search_tfidf_SR_underrep/tfidf_vectorizer_lr_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "TFIDF"
    },
    "LR_SBERT_SR_underrep": {
        "model_path": "results/lr/lr_sbert_grid_search_SR_underrep/lr_sbert_binary_relevance_model.joblib",
        "extractor_path": "results/lr/lr_sbert_grid_search_SR_underrep/sbert_scaler_lr.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "SBERT"
    },

    # SVM
    "SVM_TFIDF_OG": {
        "model_path": "results/svm/svm_grid_search_tfidf/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_grid_search_tfidf/tfidf_vectorizer_svm_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "TFIDF"
    },
    "SVM_SBERT_OG": {
        "model_path": "results/svm/svm_sbert_grid_search/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_sbert_grid_search/sbert_scaler_svm.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "SBERT"
    },
    "SVM_TFIDF_RS": {
        "model_path": "results/svm/svm_grid_search_tfidf_RS/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_grid_search_tfidf_RS/tfidf_vectorizer_svm_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "TFIDF"
    },
    "SVM_SBERT_RS": {
        "model_path": "results/svm/svm_sbert_grid_search_RS/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_sbert_grid_search_RS/sbert_scaler_svm.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "SBERT"
    },
    "SVM_TFIDF_RS_underrep": {
        "model_path": "results/svm/svm_grid_search_tfidf_RS_underrep/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_grid_search_tfidf_RS_underrep/tfidf_vectorizer_svm_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "TFIDF"
    },
    "SVM_SBERT_RS_underrep": {
        "model_path": "results/svm/svm_sbert_grid_search_RS_underrep/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_sbert_grid_search_RS_underrep/sbert_scaler_svm.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "SBERT"
    },
    "SVM_TFIDF_SR": {
        "model_path": "results/svm/svm_grid_search_tfidf_SR/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_grid_search_tfidf_SR/tfidf_vectorizer_svm_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "TFIDF"
    },
    "SVM_SBERT_SR": {
        "model_path": "results/svm/svm_sbert_grid_search_SR/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_sbert_grid_search_SR/sbert_scaler_svm.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "SBERT"
    },
    "SVM_TFIDF_SR_underrep": {
        "model_path": "results/svm/svm_grid_search_tfidf_SR_underrep/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_grid_search_tfidf_SR_underrep/tfidf_vectorizer_svm_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "TFIDF"
    },
    "SVM_SBERT_SR_underrep": {
        "model_path": "results/svm/svm_sbert_grid_search_SR_underrep/svm_binary_relevance_model.joblib",
        "extractor_path": "results/svm/svm_sbert_grid_search_SR_underrep/sbert_scaler_svm.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "SBERT"
    },

    # RF
    "RF_TFIDF_OG": {
        "model_path": "results/rf/rf_grid_search_tfidf/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_grid_search_tfidf/tfidf_vectorizer_rf_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "TFIDF"
    },
    "RF_SBERT_OG": {
        "model_path": "results/rf/rf_sbert_grid_search/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_sbert_grid_search/sbert_scaler_rf.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "SBERT"
    },
    "RF_TFIDF_RS": {
        "model_path": "results/rf/rf_grid_search_tfidf_RS/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_grid_search_tfidf_RS/tfidf_vectorizer_rf_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "TFIDF"
    },
    "RF_SBERT_RS": {
        "model_path": "results/rf/rf_sbert_grid_search_RS/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_sbert_grid_search_RS/sbert_scaler_rf.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "SBERT"
    },
    "RF_TFIDF_RS_underrep": {
        "model_path": "results/rf/rf_grid_search_tfidf_RS_underrep/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_grid_search_tfidf_RS_underrep/tfidf_vectorizer_rf_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "TFIDF"
    },
    "RF_SBERT_RS_underrep": {
        "model_path": "results/rf/rf_sbert_grid_search_RS_underrep/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_sbert_grid_search_RS_underrep/sbert_scaler_rf.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "SBERT"
    },
    "RF_TFIDF_SR": {
        "model_path": "results/rf/rf_grid_search_tfidf_SR/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_grid_search_tfidf_SR/tfidf_vectorizer_rf_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "TFIDF"
    },
    "RF_SBERT_SR": {
        "model_path": "results/rf/rf_sbert_grid_search_SR/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_sbert_grid_search_SR/sbert_scaler_rf.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "SBERT"
    },
    "RF_TFIDF_SR_underrep": {
        "model_path": "results/rf/rf_grid_search_tfidf_SR_underrep/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_grid_search_tfidf_SR_underrep/tfidf_vectorizer_rf_baseline.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "TFIDF"
    },
    "RF_SBERT_SR_underrep": {
        "model_path": "results/rf/rf_sbert_grid_search_SR_underrep/rf_binary_relevance_model.joblib",
        "extractor_path": "results/rf/rf_sbert_grid_search_SR_underrep/sbert_scaler_rf.joblib",
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "SBERT"
    },

    # Roberta base
    "RoBERTa_OG" : {
        "model_path": "results/optuna/HPO_roberta-base_UPDATED/RoBERTa_NoAug_FL_ValEval_20250513_125329/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "RoBERTa"
    },
    "RoBERTa_SR" : {
        "model_path": "results/optuna/HPO_roberta-base_SR/RoBERTa_NoAug_FL_ValEval_20250516_175935/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "RoBERTa"
    },
    "RoBERTa_SR_underrep" : {
        "model_path": "results/optuna/HPO_roberta-base_SR_underrep/RoBERTa_NoAug_FL_ValEval_20250516_163245/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "RoBERTa"
    },
    "RoBERTa_RS" : {
        "model_path": "results/optuna/HPO_roberta-base_RS/RoBERTa_NoAug_FL_ValEval_20250516_132553/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "RoBERTa"
    },
    "RoBERTa_RS_underrep" : {
        "model_path": "results/optuna/HPO_roberta-base_RS_underrep/RoBERTa_NoAug_FL_ValEval_20250516_150652/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "RoBERTa"
    },

    # Distil roberta 
    "DistilRoBERTa_OG" : {
        "model_path": "results/optuna/HPO_distil-roberta-base_UPDATED/RoBERTa_NoAug_FL_ValEval_20250513_140403/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin.npy"),
        "feature_type": "RoBERTa"
    },
    "DistilRoBERTa_SR" : {
        "model_path": "results/optuna/HPO_distil-roberta-base_SR/RoBERTa_NoAug_FL_ValEval_20250514_131628/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR.npy"),
        "feature_type": "RoBERTa"
    },
    "DistilRoBERTa_SR_underrep" : {
        "model_path": "results/optuna/HPO_distil-roberta-base_SR_underrep/RoBERTa_NoAug_FL_ValEval_20250514_121913/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_SR_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_SR_underrep.npy"),
        "feature_type": "RoBERTa"
    },
    "DistilRoBERTa_RS" : {
        "model_path": "results/optuna/HPO_distil-roberta-base_RS/RoBERTa_NoAug_FL_ValEval_20250514_090712/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS.npy"),
        "feature_type": "RoBERTa"
    },
    "DistilRoBERTa_RS_underrep" : {
        "model_path": "results/optuna/HPO_distil-roberta-base_RS_underrep/RoBERTa_NoAug_FL_ValEval_20250514_100128/best_roberta_model",
        "extractor_path": None,
        "mlb_path": os.path.join(PREPARED_DATA_DIR,"mlb_RS_underrep.joblib"),
        "y_test_bin_path": os.path.join(PREPARED_DATA_DIR,"y_test_bin_RS_underrep.npy"),
        "feature_type": "RoBERTa"
    }
}

SEED = 42
RANKING_K_VALUES = [1, 3, 5, 10]

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    SBERT_DEVICE = "cuda"
    HF_DEVICE = 0
else:
    SBERT_DEVICE = "cpu"
    HF_DEVICE = -1

# Load common test sets
script_start_time = time.time()
print("Loading common data...")
try:
    INPUT_TEST_DF_PATH = os.path.join(PREPARED_DATA_DIR, 'test_df.csv')
    test_df = pd.read_csv(INPUT_TEST_DF_PATH)
    test_df['processed_text'] = test_df["processed_text"].fillna('')
    print(f"Loaded {len(test_df)} test samples.")
except Exception as e: 
    print(f"Error loading test DataFrame: {e}")
    sys.exit(1)

# Evaluate each model configuration on the test set
all_test_results = {}

for model_key, config in MODEL_CONFIGS.items():
    print("\n" + "="*60)
    print(f"--- Evaluating Model on TEST SET: {model_key} ---")
    print(f"    Model Path: {config.get('model_path', 'N/A')}")
    print(f"    Extractor Path: {config.get('extractor_path', 'N/A')}")
    print(f"    MLB Path: {config.get('mlb_path', 'N/A')}")
    print(f"    y_test_bin Path: {config.get('y_test_bin_path', 'N/A')}")
    print(f"    Feature Type: {config.get('feature_type', 'N/A')}")
    print("="*60)

    # Load MLB and y_Test_bin for this specific models training data version
    try: 
        if not config.get('mlb_path') or not config.get('y_test_bin_path'):
            print(f"Warning: MLB path or y_test_bin path missing for {model_key}. Skipping.")
            all_test_results[model_key] = {"Error": "MLB or y_test_bin path missing in config"}
            continue

        current_mlb = joblib.load(config['mlb_path'])
        current_y_test_bin = np.load(config['y_test_bin_path']).astype(np.float32)
        current_num_labels = len(current_mlb.classes_)

        if len(test_df) != current_y_test_bin.shape[0] or current_y_test_bin.shape[1] != current_num_labels:
            raise ValueError(f"Shape mismatch for {model_key}: test_df ({len(test_df)}) vs y_test_bin ({current_y_test_bin.shape}) or MLB classes ({current_num_labels})")
    except FileNotFoundError as e_data:
        print(f"ERROR for {model_key}: Specific MLB or y_test_bin file not found: {e_data}. Check paths. Skipping.")
        all_test_results[model_key] = {"Error": f"Data file missing: {e_data}"}
        continue
    except Exception as e_load_specific:
        print(f"ERROR loading specific data for {model_key}: {e_load_specific}")
        all_test_results[model_key] = {"Error": f"Failed to load specific data: {e_load_specific}"}
        continue

    X_test_features_current_model = None
    predictor = None

    try:
        # Load model and feature extractor
        print("Loading model and feature extractor...")
        model_load_start = time.time()
        feature_type = config['feature_type']

        if feature_type == "TFIDF":
            predictor = joblib.load(config['model_path'])
            X_test_features_current_model = test_df["processed_text"]
            print(f"Loaded TFIDF extractor and {model_key} model.")

        elif feature_type == "SBERT":
            scaler = None
            if config.get('extractor_path'): 
                scaler = joblib.load(config['extractor_path'])

            print(f"Generating SBERT embeddings...")
            sbert_model_instance = SentenceTransformer("llms/all-mpnet-base-v2-local-files", device=SBERT_DEVICE)
            X_test_embeddings = sbert_model_instance.encode(test_df['processed_text'].tolist(), show_progress_bar=False)
            if scaler: 
                X_test_features_current_model = scaler.transform(X_test_embeddings)
            else: 
                X_test_features_current_model = X_test_embeddings
            X_test_features_current_model = X_test_features_current_model.astype(np.float32)

            predictor = joblib.load(config['model_path'])
            print(f"Generated SBERT features and loaded {model_key} model.")

        elif feature_type == "RoBERTa":
            print(f"Loading RoBERTa model and tokenizer from {config['model_path']}...")
            
            roberta_model = AutoModelForSequenceClassification.from_pretrained(config["model_path"])
            roberta_tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
            roberta_model.to(torch.device(SBERT_DEVICE)) # cuda or cpu
            roberta_model.eval()

            print("Loaded RoBERTa pipeline.")
        else:
            print(f"Warning: Unknown feature_type '{feature_type}' for model {model_key}. Skipping.")
            continue
        print(f"Model/Extractor loading took {time.time() - model_load_start:.2f}s")

        # Generate predictions
        print("Generating predictions on test set...")
        pred_start_time = time.time()
        y_pred_bin_test_current_model = None
        y_pred_scores_test_current_model = None

        if feature_type == "RoBERTa":
            test_texts = test_df["processed_text"].tolist()
            dummy_labels_for_predict = np.zeros((len(test_texts), current_num_labels), dtype=np.float32)
            hf_test_dataset = Dataset.from_dict({'text': test_texts, 'labels': dummy_labels_for_predict})

            def tokenize_for_predict(examples):
                return roberta_tokenizer(examples["text"], truncation=True, max_length=256, padding=False)
            
            tokenized_hf_test_dataset = hf_test_dataset.map(tokenize_for_predict, batched=True, remove_columns=["text"])

            temp_training_args = TrainingArguments(output_dir="./temp_training_predict_dir", per_device_eval_batch_size=16)
            trainer_for_predict = Trainer(
                model=roberta_model,
                args=temp_training_args,
                tokenizer=roberta_tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=roberta_tokenizer)
            )

            predictions_output = trainer_for_predict.predict(tokenized_hf_test_dataset)
            logits = predictions_output.predictions

            sigmoid = torch.nn.Sigmoid()
            y_pred_scores_test_current_model = sigmoid(torch.Tensor(logits)).numpy()
            threshold = 0.4
            y_pred_bin_test_current_model = (y_pred_scores_test_current_model >= threshold).astype(int)

        else: # Sklearn models/pipelines
            y_pred_bin_test_current_model = predictor.predict(X_test_features_current_model)
            if hasattr(predictor, "predict_proba"):
                y_pred_proba_list = predictor.predict_proba(X_test_features_current_model)
                if isinstance(y_pred_proba_list, list): # For MultiOutputClassifier
                    if y_pred_proba_list and isinstance(y_pred_proba_list[0], np.ndarray) and y_pred_proba_list[0].shape[1] == 2:
                        y_pred_scores_test_current_model = np.array([proba[:, 1] for proba in y_pred_proba_list]).T
                    else: y_pred_scores_test_current_model = y_pred_bin_test_current_model.astype(float)
                else: y_pred_scores_test_current_model = y_pred_proba_list
            elif hasattr(predictor, "decision_function"):
                y_pred_scores_test_current_model = predictor.decision_function(X_test_features_current_model)
            else: y_pred_scores_test_current_model = y_pred_bin_test_current_model.astype(float)
       
        if y_pred_bin_test_current_model is None or y_pred_scores_test_current_model is None:
             raise RuntimeError(f"Prediction generation failed for {model_key}")
        print(f"Prediction finished. Time: {time.time() - pred_start_time:.2f}s")

        # Evaluate
        print(f"Evaluating performance of {model_key} on TEST SET...")
        model_eval_output_dir = os.path.join(OUTPUT_DIR, model_key)
        os.makedirs(model_eval_output_dir, exist_ok=True)

        if y_pred_scores_test_current_model is not None:
            y_pred_scores_test_current_model = y_pred_scores_test_current_model.astype(np.float32)

        eval_metrics = eval_utils.evaluate_model_predictions(
                y_true_bin=current_y_test_bin,
                y_pred_bin=y_pred_bin_test_current_model,
                y_pred_scores=y_pred_scores_test_current_model,
                mlb=current_mlb,
                k_values=RANKING_K_VALUES,
                output_dir=model_eval_output_dir,
                model_name=f"{model_key}_FinalTest",
                threshold_for_report=0.5
            )
        all_test_results[model_key] = eval_metrics
        print(f"Evaluation complete for {model_key}.")

    except FileNotFoundError as e_fnf:
        print(f"ERROR for {model_key}: Model or extractor file not found: {e_fnf}. Check paths in MODEL_CONFIGS.")
        all_test_results[model_key] = {"Error": f"FileNotFound: {e_fnf}"}
    except Exception as e_proc:
        print(f"ERROR processing model {model_key}: {e_proc}")
        import traceback
        traceback.print_exc()
        all_test_results[model_key] = {"Error": str(e_proc)}

# Compile and Save Comparison
print("\n" + "="*60)
print("--- Final Benchmark Comparison (Test Set) ---")
print("="*60)
results_df = pd.DataFrame.from_dict(all_test_results, orient='index')
summary_cols = [ # From eval_utils.py
    'Top_1_Accuracy', 'Top_3_Accuracy', 'Top_5_Accuracy', 'Top_10_Accuracy',
    'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
    'MAP', 'MRR'
]
summary_cols_present = [col for col in summary_cols if col in results_df.columns]
if not summary_cols_present and not results_df.empty : # If specific cols not found, show all available
    results_df_summary = results_df
else:
    results_df_summary = results_df[summary_cols_present]


print(results_df_summary.to_string(float_format="%.4f"))

try:
    summary_path = os.path.join(OUTPUT_DIR, "final_benchmark_summary_TEST.csv")
    results_df_summary.to_csv(summary_path, float_format="%.4f")
    print(f"\nSummary results saved to: {summary_path}")

    full_results_path = os.path.join(OUTPUT_DIR, "final_benchmark_full_results_TEST.json")
    serializable_results = {}
    for model_name_res, metrics_dict_res in all_test_results.items():
        serializable_results[model_name_res] = {
            k: (float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v)
            for k, v in metrics_dict_res.items()
        }
    with open(full_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Full results dictionary saved to: {full_results_path}")
except Exception as e:
    print(f"Error saving summary results: {e}")

script_end_time = time.time()
print(f"\n--- Final Evaluation Script Completed ---")
print(f"Reports and summaries saved in: {OUTPUT_DIR}")
print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")

