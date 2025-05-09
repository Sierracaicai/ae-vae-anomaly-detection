import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    roc_curve,
    auc
)

def compute_reconstruction_error(model, X):
    """
    Compute per-sample reconstruction error (MSE).

    Parameters:
        model: trained autoencoder
        X: input data

    Returns:
        errors: array of reconstruction MSE per sample
    """
    X_pred = model.predict(X)
    return np.mean(np.square(X - X_pred), axis=1)

def evaluate_anomaly_detection(errors, y_true, threshold=None):
    """
    Evaluate anomaly detection performance from reconstruction errors.

    Parameters:
        errors (np.array): reconstruction errors
        y_true (np.array): true binary labels (0=normal, 1=anomaly)
        threshold (float): optional manual threshold

    Returns:
        dict: performance metrics including best threshold, f1, auc, confusion matrix
    """
    precision, recall, thresholds = precision_recall_curve(y_true, errors)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = threshold or thresholds[best_idx]

    preds = (errors >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    auc_score = roc_auc_score(y_true, errors)

    return {
        "threshold": best_thresh,
        "precision": precision[best_idx],
        "recall": recall[best_idx],
        "f1": f1_scores[best_idx],
        "roc_auc": auc_score,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    }

def plot_reconstruction_error_distribution(errors, y_true, threshold=None):
    """
    Plot histogram of reconstruction error for normal vs anomaly samples.

    Parameters:
        errors (np.array): reconstruction errors
        y_true (np.array): true labels
        threshold (float): optional threshold line
    """
    plt.figure(figsize=(10,5))
    plt.hist(errors[y_true==0], bins=50, alpha=0.6, label='Normal', color='blue')
    plt.hist(errors[y_true==1], bins=50, alpha=0.6, label='Anomaly', color='red')
    if threshold:
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_reconstruction(model, X, n=5):
    """
    Plot original vs reconstructed feature vectors for a few samples.

    Parameters:
        model: trained autoencoder
        X: input data
        n: number of samples to visualize
    """
    preds = model.predict(X)
    idx = np.random.choice(len(X), n, replace=False)
    plt.figure(figsize=(15, n * 2))
    for i, j in enumerate(idx):
        plt.subplot(n, 2, 2*i+1)
        plt.plot(X[j], label='Original')
        plt.title(f"Sample {j} - Original")
        plt.subplot(n, 2, 2*i+2)
        plt.plot(preds[j], label='Reconstructed')
        plt.title(f"Sample {j} - Reconstructed")
    plt.tight_layout()
    plt.show()
