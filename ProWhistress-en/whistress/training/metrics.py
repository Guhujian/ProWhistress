# metrics.py implements evaluation metrics for the WhiStress model,
# including accuracy, precision, recall, and F1.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Metrics class encapsulating commonly used evaluation scores.
class WhiStressMetrics:
    def __init__(self):
        # Use local sklearn metrics without relying on the online evaluate package.
        pass

    def compute_metrics(self, pred):
        # Ignore masked labels (-100) and compute metrics only on valid positions.
        def ignore_masked_predictions(pred_ids, label_ids, pad_token_id):
            # Create a mask where label_ids is not equal to pad_token_id
            mask_label_ids = label_ids != pad_token_id

            # Flatten the tensors to process them as one-dimensional arrays
            pred_ids_flat = pred_ids[mask_label_ids].flatten()
            label_ids_flat = label_ids[mask_label_ids].flatten()

            return pred_ids_flat, label_ids_flat

        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
        preds, labels = ignore_masked_predictions(pred_ids, label_ids, -100)

        metrics = {}
        # Compute accuracy
        metrics["accuracy"] = accuracy_score(labels, preds)
        
        # Compute precision
        metrics["precision"] = precision_score(labels, preds, pos_label=1, zero_division=0)
        
        # Compute recall
        metrics["recall"] = recall_score(labels, preds, pos_label=1, zero_division=0)
        
        # Compute F1 score
        metrics["f1"] = f1_score(labels, preds, pos_label=1, zero_division=0)

        return metrics
