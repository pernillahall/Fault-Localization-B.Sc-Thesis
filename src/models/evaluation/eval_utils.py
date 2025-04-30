import numpy as np
import os
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    classification_report, label_ranking_average_precision_score
)

# Define Ranking Metric Functions
def recall_at_k(y_true_bin, ranked_indices, k):
    """Calculates Recall at k for multi-label"""
    recalls = []
    for i in range(len(y_true_bin)):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        if len(true_positive_indices) == 0: 
            continue
        top_k_preds = ranked_indices[i, :k]; hits = np.intersect1d(true_positive_indices, top_k_preds);
        recall = len(hits) / len(true_positive_indices); recalls.append(recall);
    return np.mean(recalls) if recalls else 0.0

def precision_at_k(y_true_bin, ranked_indices, k):
    """Calculates Precision at k for multi-label"""
    precisions = []
    if k == 0: 
        return 0.0
    for i in range(len(y_true_bin)):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        top_k_preds = ranked_indices[i, :k]; hits = np.intersect1d(true_positive_indices, top_k_preds);
        precision = len(hits) / k; precisions.append(precision);
    return np.mean(precisions) if precisions else 0.0

def top_k_accuracy_score(y_true_bin, ranked_indices, k):
    """Calculates Top-k Accuracy for multi-label"""
    hits_at_k = 0
    num_samples = len(y_true_bin)
    if num_samples == 0: return 0.0

    for i in range(num_samples):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        # Check only if there are true labels to be found for this sample
        if len(true_positive_indices) > 0:
            top_k_preds = ranked_indices[i, :k]
            # Filter out potential padding indices (-1) if used
            top_k_preds = top_k_preds[top_k_preds != -1]
            if len(top_k_preds) > 0:
                hits = np.intersect1d(true_positive_indices, top_k_preds)
                if len(hits) > 0:
                    hits_at_k += 1 # Increment if at least one true label was found
    return hits_at_k / num_samples if num_samples > 0 else 0.0

def mean_reciprocal_rank(y_true_bin, ranked_indices):
    """Calculates Mean Reciprocal Rank (MRR)."""
    reciprocal_ranks = []
    num_samples = len(y_true_bin)
    if num_samples == 0: return 0.0

    for i in range(num_samples):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        # Only calculate MRR for samples that have at least one true label
        if len(true_positive_indices) > 0:
            preds = ranked_indices[i]
            preds = preds[preds != -1] # Remove padding if necessary
            first_correct_rank = 0 # Use 0 if no correct label is found
            for rank, predicted_idx in enumerate(preds):
                 if predicted_idx in true_positive_indices:
                     first_correct_rank = rank + 1 # Rank is 1-based
                     break # Stop after finding the first correct one

            if first_correct_rank > 0:
                 reciprocal_ranks.append(1.0 / first_correct_rank)
            else:
                 reciprocal_ranks.append(0.0) # Assign 0 if no correct label found in ranked list

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

# Main Evaluation Function
# This function evaluates the model predictions using various metrics and saves the results.
def evaluate_model_predictions(
    y_true_bin, y_pred_bin, y_pred_scores, mlb, k_values, output_dir,
    model_name="Model", threshold_for_report=0.5
    ):
    """Calculates, prints, and saves threshold and ranking metrics."""

    print(f"\n--- Evaluating '{model_name}' ---")
    report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    ranking_path = os.path.join(output_dir, f"{model_name}_ranking_metrics.txt")

    # Threshold-Based Metrics
    print(f"\n--- Threshold-Based Metrics (@ threshold {threshold_for_report}) ---")
    eval_metrics_threshold = {}
    try:
        eval_metrics_threshold['Subset Accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
        eval_metrics_threshold['Hamming Loss'] = hamming_loss(y_true_bin, y_pred_bin)
        eval_metrics_threshold['F1 Micro'] = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
        eval_metrics_threshold['F1 Macro'] = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
        eval_metrics_threshold['F1 Weighted'] = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
        print(f"  Subset Accuracy (Exact Match): {eval_metrics_threshold['Subset Accuracy']:.4f}")
        print(f"  Hamming Loss:                 {eval_metrics_threshold['Hamming Loss']:.4f}")
        print(f"  F1 Score (Micro):             {eval_metrics_threshold['F1 Micro']:.4f}")
        print(f"  F1 Score (Macro):             {eval_metrics_threshold['F1 Macro']:.4f}")
        print(f"  F1 Score (Weighted):          {eval_metrics_threshold['F1 Weighted']:.4f}")

        print(f"\n--- Detailed Classification Report (@ threshold {threshold_for_report}) ---")
        if hasattr(mlb, 'classes_') and len(mlb.classes_) == y_true_bin.shape[1]:
            report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0)
            print(report)
            try:
                report_content = f"Evaluation Metrics ({model_name}, @ {threshold_for_report} Threshold):\n"
                for key, value in eval_metrics_threshold.items(): report_content += f"  {key}: {value:.4f}\n"
                report_content += f"\nDetailed Classification Report:\n{report}"
                with open(report_path, "w", encoding='utf-8') as f: f.write(report_content)
                print(f"Classification report saved to {report_path}")
            except Exception as e_save: print(f"Warning: Could not save classification report: {e_save}")
        else: print("Warning: Could not generate classification report (mlb.classes_ missing or mismatch).")

    except Exception as e: print(f"Error calculating threshold metrics: {e}")

    #  Ranking Metrics
    print("\n--- Ranking-Based Metrics ---")
    ranking_metrics = {}
    if y_pred_scores is not None:
        try:
            scores = y_pred_scores
            ranked_indices = np.argsort(-scores, axis=1) # Higher score is better
            true_labels_bin = y_true_bin # Use consistent name

            print("Calculating R@k, P@k, Top-k Accuracy...")
            for k in k_values: # Use the k_values list passed to the function
                r_at_k = recall_at_k(true_labels_bin, ranked_indices, k=k)
                p_at_k = precision_at_k(true_labels_bin, ranked_indices, k=k)
                top_k_acc = top_k_accuracy_score(true_labels_bin, ranked_indices, k=k) # Calculate Top-k Acc
                ranking_metrics[f'Recall@{k}'] = r_at_k
                ranking_metrics[f'Precision@{k}'] = p_at_k
                ranking_metrics[f'Top_{k}_Accuracy'] = top_k_acc # Add Top-k Acc
                print(f"  Recall@{k:<2}: {r_at_k:.4f}")
                print(f"  Precision@{k:<2}: {p_at_k:.4f}")
                print(f"  Top-{k} Accuracy: {top_k_acc:.4f}") # Print Top-k Acc

            lrap = label_ranking_average_precision_score(true_labels_bin, scores)
            mrr = mean_reciprocal_rank(true_labels_bin, ranked_indices) # Calculate MRR
            ranking_metrics['LRAP (MAP-like)'] = lrap
            ranking_metrics['MRR'] = mrr # Add MRR
            print(f"  LRAP (MAP-like): {lrap:.4f}")
            print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}") # Print MRR

            # Save ranking metrics (including new ones)
            try:
                with open(ranking_path, "w") as f:
                    f.write(f"Ranking Metrics ({model_name}):\n")
                    for key, value in ranking_metrics.items():
                        f.write(f"  {key.replace('@', ' at ').replace('_', '-')}: {value:.4f}\n") # Format key
                print(f"Ranking metrics saved to {ranking_path}")
            except Exception as e_save: print(f"Warning: Could not save ranking metrics: {e_save}")

        except Exception as e: print(f"Error during ranking metric calculation: {e}")
    else: print("Skipping ranking metrics as scores were not provided.")

    # Combine results
    all_metrics = {**eval_metrics_threshold, **ranking_metrics}
    return all_metrics