# Imports
import pandas as pd
import numpy as np
import torch
import random
import os
import sys
import joblib
import time

# Hugging face libraries
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding,
    logging as hf_logging
)

# Custom modules
from focal_loss_trainer import FocalLossTrainer
import eval_utils 

# Show warnings only
hf_logging.set_verbosity_warning()

# Configuration
print("\nConfiguring environment for RoBERTa Fine-tuning...")
PREPARED_DATA_DIR = "data/prepared_data"
MODEL_PATH = "llms/roberta-base-local-files"
OUTPUT_DIR_BASE = "results/roberta_benchmarks"

RUN_NAME = f"RoBERTa_NoAug_FL_ValEval_{time.strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, RUN_NAME)
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")

SEED = 42
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
FP16_TRAINING = torch.cuda.is_available()

FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

RANKING_K_VALUES = [1, 3, 5, 10]
TRAINER_LOG_K_VALUES = [1, 3, 5]
EVAL_TARGET = 'validation' # Set to 'test' for test set evaluation

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
print(f"Setting random seed to {SEED}")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(SEED)

# Load pre-split data and prepared labels
script_start_time = time.time()
print("\nLoading pre-split data and prepared labels...")

try:
    train_df = pd.read_csv(os.path.join(PREPARED_DATA_DIR, 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(PREPARED_DATA_DIR, 'val_df.csv'))
    test_df = pd.read_csv(os.path.join(PREPARED_DATA_DIR, 'test_df.csv'))
    mlb = joblib.load(os.path.join(PREPARED_DATA_DIR, 'mlb.joblib'))
    num_labels = len(mlb.classes_)
    y_train_bin = np.load(os.path.join(PREPARED_DATA_DIR, 'y_train_bin.npy')).astype(np.float32)
    y_val_bin = np.load(os.path.join(PREPARED_DATA_DIR, 'y_val_bin.npy')).astype(np.float32)
    y_test_bin = np.load(os.path.join(PREPARED_DATA_DIR, 'y_test_bin.npy')).astype(np.float32)
    print(f"Loaded data. Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}. Labels={num_labels}")
    if not (len(train_df) == y_train_bin.shape[0] and y_train_bin.shape[1] == num_labels): 
        raise ValueError("Data shape mismatch.")
except Exception as e: 
    print(f"Error loading pre-split data: {e}")
    sys.exit(1)

# Data augmentation goes here
train_texts_final = train_df['processed_text'].tolist()
y_train_bin_final = y_train_bin

# Prepare hugging face datasets
print("\nPreparing Hugging Face Datasets...")
train_dataset = Dataset.from_dict({'text': train_texts_final, 'labels': y_train_bin_final})
val_dataset = Dataset.from_dict({'text': val_df['processed_text'].tolist(), 'labels': y_val_bin})
test_dataset = Dataset.from_dict({'text': test_df['processed_text'].tolist(), 'labels': y_test_bin})
print(f"Created HF Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Load RoBERTa tokenizer and model for multi-label classification
print(f"\nLoading Tokenizer and Model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels, problem_type="multi_label_classification")
    print("Model and Tokenizer loaded.")
except Exception as e: 
    print(f"Error loading model/tokenizer: {e}")
    sys.exit(1)

# Tokenize datasets
print("\nTokenizing datasets...")
def tokenize_function(examples): 
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Tokenization complete.")

# Pad the tokenized inputs
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation metrics function for monitoring during training
# EvalPrediction contains predictions and true labels
def compute_ranking_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels_true_bin = p.label_ids.astype(np.int32)
   
    metrics = {}

    try:
        # Scores for ranking metrics need to be probabilities, use sigmoid
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(torch.Tensor(logits)).cpu().numpy()
        
        # Sort labels in descending order of scores
        ranked_indices = np.argsort(-scores, axis=1)

        # Calculate ranking metrics
        for k_val in TRAINER_LOG_K_VALUES:
            metrics[f'recall_at_{k_val}'] = eval_utils.recall_at_k(labels_true_bin, ranked_indices, k_val)
            metrics[f'top_{k_val}_accuracy'] = eval_utils.top_k_accuracy_score(labels_true_bin, ranked_indices, k_val)

        metrics['MAP'] = eval_utils.mean_average_precision(labels_true_bin, ranked_indices)
        metrics['MRR'] = eval_utils.mean_reciprocal_rank(labels_true_bin, ranked_indices)

    # Return defaults for all expected keys if error occurs
    except Exception as e:
        print(f"Error in compute_ranking_metrics_for_trainer: {e}")
        for k_val in TRAINER_LOG_K_VALUES:
            metrics[f'recall_at_{k_val}'] = 0.0
            metrics[f'top_{k_val}_accuracy'] = 0.0
        metrics['MAP'] = 0.0
        metrics['MRR'] = 0.0
    return metrics


# Set Training Arguments
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_MAP", # Monitor 'MAP' for best model.
    greater_is_better = True, # For MAP, MRR, and LRAP
    fp16=FP16_TRAINING,
    seed=SEED,
    report_to="none",
    save_total_limit=1, # Keep only the best model
)

# Initialize Trainer
print("Initializing Trainer...")
trainer_kwargs = {}
trainer_kwargs['focal_loss_alpha'] = FOCAL_LOSS_ALPHA
trainer_kwargs['focal_loss_gamma'] = FOCAL_LOSS_GAMMA

trainer = FocalLossTrainer(
    model=model, 
    args=training_args,
    train_dataset=tokenized_train_dataset, 
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_ranking_metrics, # use custom ranking metrics
    **trainer_kwargs
)

# Train the model in a loop
print(f"\nStarting Fine-tuning for {EPOCHS} epochs...")
try:
    train_start_time = time.time()
    train_result = trainer.train()
    train_end_time = time.time()
    print(f"Training finished. Time: {train_end_time - train_start_time:.2f}s.")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
except Exception as e: 
    print(f"Error during training: {e}")
    sys.exit(1)


# Evaluation on chosen set (using eval_utils)
print(f"\n--- Evaluating Best Model on {EVAL_TARGET.upper()} ---")

if EVAL_TARGET == 'validation':
    eval_dataset_final = tokenized_val_dataset
    y_eval_bin_final = y_val_bin
else:
    eval_dataset_final = tokenized_test_dataset
    y_eval_bin_final = y_test_bin

try:
    print("Generating predictions...")
    predictions_output = trainer.predict(eval_dataset_final)

    # Extract logits and convert to numpy array
    logits = predictions_output.predictions
    logits_cpu = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    true_labels_cpu = y_eval_bin_final

    # Apply sigmoid to logits to get binary predictions
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(torch.Tensor(logits_cpu)).numpy()
    threshold = 0.4
    y_pred_bin_final = (probabilities >= threshold).astype(int)

    print(f"\n--- Full Evaluation Report for {EVAL_TARGET.upper()} Set ---")
    model_id_str = f"RoBERTa_FL_{EVAL_TARGET}"

    # Evaluate using custom eval_utils
    eval_results = eval_utils.evaluate_model_predictions(
        y_true_bin=true_labels_cpu, 
        y_pred_bin=y_pred_bin_final,
        y_pred_scores=probabilities, 
        mlb=mlb, 
        k_values=RANKING_K_VALUES,
        output_dir=OUTPUT_DIR, 
        model_name=model_id_str, 
        threshold_for_report=threshold
    )

    # Save results to JSON
    try:
        eval_metrics_file = os.path.join(OUTPUT_DIR, f"final_{EVAL_TARGET}_eval_utils_metrics.json")
        serializable_results = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k, v in eval_results.items()}
        pd.Series(serializable_results).to_json(eval_metrics_file, indent=4)
        print(f"Summary evaluation metrics from eval_utils saved to {eval_metrics_file}")
    except Exception as e_save: print(f"Warning: could not save summary metrics json: {e_save}")
except Exception as e: 
    print(f"Error during final evaluation: {e}")
    import traceback
    traceback.print_exc()

# Save final model & objects
print("\nSaving the best model and associated objects...")
final_model_save_path = os.path.join(OUTPUT_DIR, "best_roberta_model")
try:
    trainer.save_model(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    joblib.dump(mlb, os.path.join(final_model_save_path, 'mlb.joblib'))
    print(f"Best model, tokenizer, and MLB saved to {final_model_save_path}")
    config_save_path = os.path.join(OUTPUT_DIR, "run_config.txt")
    with open(config_save_path, "w") as f:
         f.write(f"MODEL_TYPE=RoBERTa\nBASE_MODEL_PATH={MODEL_PATH}\n")
         f.write(f"PREPARED_DATA_DIR={PREPARED_DATA_DIR}\nSEED={SEED}\nEPOCHS={EPOCHS}\nLR={LEARNING_RATE}\n")
         f.write(f"BATCH_SIZE={TRAIN_BATCH_SIZE}\nGRAD_ACCUM={GRADIENT_ACCUMULATION_STEPS}\nMAX_LENGTH={MAX_LENGTH}\n")
         f.write(f"ALPHA={FOCAL_LOSS_ALPHA}\nGAMMA={FOCAL_LOSS_GAMMA}\n")
         f.write(f"EVAL_TARGET={EVAL_TARGET}\n")
    print(f"Run configuration saved to {config_save_path}")
except Exception as e: 
    print(f"Error saving final model/objects: {e}")

script_end_time = time.time()
print(f"\n--- Fine-tuning script completed ---")
print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")