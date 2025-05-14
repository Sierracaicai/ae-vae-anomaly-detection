# `thresholding.py` Documentation

This module provides utility functions for selecting thresholds and converting anomaly scores into binary predictions, typically used in unsupervised anomaly detection tasks with reconstruction-based models like Autoencoders or Variational Autoencoders.

---

## 📌 Functions

### `find_best_f1_threshold`

```python
def find_best_f1_threshold(errors: np.ndarray,
                           labels: np.ndarray,
                           num_thresholds: int = 1000,
                           return_all: bool = False
                           ) -> Tuple[float, Optional[np.ndarray]]:
```

**Description**:  
Selects the best threshold that maximizes the F1 score on a labeled validation set.

**Parameters**:
- `errors`: `np.ndarray` — Reconstruction errors or anomaly scores.
- `labels`: `np.ndarray` — Ground truth labels (0 for normal, 1 for anomaly).
- `num_thresholds`: `int` — Number of candidate thresholds to scan.
- `return_all`: `bool` — Whether to return the full list of F1 scores.

**Returns**:
- `best_threshold`: `float` — Threshold value giving the highest F1.
- `f1_scores`: `Optional[np.ndarray]` — List of F1 scores (if `return_all=True`).

**Usage Example**:
```python
best_thresh, f1_scores = find_best_f1_threshold(recon_errors, y_val, return_all=True)
```

---

### `apply_threshold`

```python
def apply_threshold(errors, threshold):
```

**Description**:  
Applies a threshold to reconstruction errors to generate binary classification results.

**Parameters**:
- `errors`: `np.ndarray` — Array of reconstruction errors.
- `threshold`: `float` — Pre-determined threshold value.

**Returns**:
- `np.ndarray`: Binary prediction labels (0 or 1).

**Usage Example**:
```python
y_pred = apply_threshold(recon_errors, best_thresh)
```

---

## ✅ Typical Use Case

These functions are used after training an AE/VAE model:

1. Compute reconstruction errors on a labeled validation set.
2. Use `find_best_f1_threshold` to find the optimal threshold.
3. Apply this threshold to test set errors using `apply_threshold`.
