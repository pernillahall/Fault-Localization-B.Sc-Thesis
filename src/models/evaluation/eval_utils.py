import numpy as np
import os
from sklearn.metrics import classification_report

# Define Ranking Metric Functions (Including MAP)
def recall_at_k(y_true_bin, ranked_indices, k):
    """Calculates Recall at k for multi-label"""
    recalls = []
    for i in range(len(y_true_bin)):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        if len(true_positive_indices) == 0: 
            continue
        top_k_preds = ranked_indices[i, :k]
        hits = np.intersect1d(true_positive_indices, top_k_preds)
        recall = len(hits) / len(true_positive_indices)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0

def precision_at_k(y_true_bin, ranked_indices, k):
    """Calculates Precision at k for multi-label"""
    precisions = []
    if k == 0: 
        return 0.0
    for i in range(len(y_true_bin)):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        top_k_preds = ranked_indices[i, :k]
        hits = np.intersect1d(true_positive_indices, top_k_preds)
        precision = len(hits) / k
        precisions.append(precision)
    return np.mean(precisions) if precisions else 0.0

def top_k_accuracy_score(y_true_bin, ranked_indices, k):
    """Calculates Top-k Accuracy for multi-label"""
    hits_at_k = 0
    num_samples = len(y_true_bin)
    if num_samples == 0: return 0.0

    for i in range(num_samples):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        if len(true_positive_indices) > 0:
            top_k_preds = ranked_indices[i, :k]
            top_k_preds = top_k_preds[top_k_preds != -1]
            if len(top_k_preds) > 0:
                hits = np.intersect1d(true_positive_indices, top_k_preds)
                if len(hits) > 0:
                    hits_at_k += 1
    return hits_at_k / num_samples if num_samples > 0 else 0.0

def mean_reciprocal_rank(y_true_bin, ranked_indices):
    """Calculates Mean Reciprocal Rank (MRR)."""
    reciprocal_ranks = []
    num_samples = len(y_true_bin)
    if num_samples == 0: return 0.0

    for i in range(num_samples):
        true_positive_indices = np.where(y_true_bin[i] == 1)[0]
        if len(true_positive_indices) > 0:
            preds = ranked_indices[i]
            preds = preds[preds != -1]
            first_correct_rank = 0
            for rank, predicted_idx in enumerate(preds):
                if predicted_idx in true_positive_indices:
                    first_correct_rank = rank + 1
                    break

            if first_correct_rank > 0:
                 reciprocal_ranks.append(1.0 / first_correct_rank)
            else:
                 reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def mean_average_precision(y_true_bin, ranked_indices):
    """Calculates Mean Average Precision (MAP) for multi-label classification."""
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

# Main Evaluation Function (including MAP)
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

    # Ranking Metrics
    print("\n--- Ranking-Based Metrics ---")
    ranking_metrics = {}
    if y_pred_scores is not None:
        try:
            scores = y_pred_scores
            ranked_indices = np.argsort(-scores, axis=1)  # Higher score is better
            true_labels_bin = y_true_bin  # Use consistent name

            print("Calculating R@k, P@k, Top-k Accuracy...")
            for k in k_values:
                r_at_k = recall_at_k(true_labels_bin, ranked_indices, k=k)
                p_at_k = precision_at_k(true_labels_bin, ranked_indices, k=k)
                top_k_acc = top_k_accuracy_score(true_labels_bin, ranked_indices, k=k)
                ranking_metrics[f'Recall@{k}'] = r_at_k
                ranking_metrics[f'Precision@{k}'] = p_at_k
                ranking_metrics[f'Top_{k}_Accuracy'] = top_k_acc
                print(f"  Recall@{k:<2}: {r_at_k:.4f}")
                print(f"  Precision@{k:<2}: {p_at_k:.4f}")
                print(f"  Top-{k} Accuracy: {top_k_acc:.4f}")

            mrr = mean_reciprocal_rank(true_labels_bin, ranked_indices)
            ranking_metrics['MRR'] = mrr
            print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")

            # Calculate and print MAP
            map_score = mean_average_precision(true_labels_bin, ranked_indices)
            ranking_metrics['MAP'] = map_score
            print(f"  MAP (Mean Average Precision): {map_score:.4f}")

            # Save ranking metrics
            try:
                with open(ranking_path, "w") as f:
                    f.write(f"Ranking Metrics ({model_name}):\n")
                    for key, value in ranking_metrics.items():
                        f.write(f"  {key.replace('@', ' at ').replace('_', '-')}: {value:.4f}\n")
                print(f"Ranking metrics saved to {ranking_path}")
            except Exception as e_save: print(f"Warning: Could not save ranking metrics: {e_save}")

        except Exception as e: print(f"Error during ranking metric calculation: {e}")
    else: print("Skipping ranking metrics as scores were not provided.")

    # Combine results
    all_metrics = {**eval_metrics_threshold, **ranking_metrics}
    return all_metrics