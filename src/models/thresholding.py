
import numpy as np
from sklearn.metrics import f1_score
from typing import Tuple, Optional

def find_best_f1_threshold(errors: np.ndarray,
                           labels: np.ndarray,
                           num_thresholds: int = 1000,
                           return_all: bool = False
                           ) -> Tuple[float, Optional[np.ndarray]]:
    """
    Select threshold based on maximum F1 score on a labeled validation set.

    Parameters:
        errors: Anomaly scores (reconstruction errors).
        labels: Ground truth labels (0: normal, 1: anomaly).
        num_thresholds: Number of candidate thresholds to evaluate.
        return_all: Whether to return full F1 scores for all thresholds.

    Returns:
        best_threshold: Threshold giving the maximum F1.
        f1_scores (optional): Array of F1 scores for each threshold.
    """
    thresholds = np.linspace(errors.min(), errors.max(), num_thresholds)
    f1_scores = []

    for t in thresholds:
        preds = (errors > t).astype(int)
        score = f1_score(labels, preds)
        f1_scores.append(score)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    if return_all:
        return best_threshold, np.array(f1_scores)
    else:
        return best_threshold, None

def apply_threshold(errors, threshold):
    """
    Apply threshold to reconstruction errors to produce binary predictions.

    Parameters:
        errors (np.ndarray): Reconstruction errors.
        threshold (float): Selected threshold.

    Returns:
        np.ndarray: Binary predictions (0 or 1).
    """
    return (errors > threshold).astype(int)
