# 🧪 Autoencoder Evaluation Module

This module provides core evaluation and visualization utilities for trained Autoencoder (AE) models in anomaly detection settings.

---

## 📁 File
`ae_evaluation.py`

---

## ✅ Functions Overview

### 1. `compute_reconstruction_error(model, X)`
Compute mean squared error (MSE) between inputs and their AE reconstructions.

**Parameters:**
- `model`: Trained Keras autoencoder
- `X`: Input samples (NumPy array or DataFrame converted via `.to_numpy()`)

**Returns:**  
- `np.array`: Per-sample reconstruction MSE

---

### 2. `evaluate_anomaly_detection(errors, y_true, threshold=None)`
Compute key classification metrics and determine an optimal threshold based on F1 score.

**Parameters:**
- `errors`: Reconstruction errors (from `compute_reconstruction_error`)
- `y_true`: True binary labels (0 = normal, 1 = anomaly)
- `threshold`: Optional manual threshold; if None, selects based on best F1

**Returns:**
- Dictionary containing:
  - Best threshold
  - Precision, recall, F1 score
  - ROC AUC
  - Confusion matrix (`TP`, `FP`, `TN`, `FN`)

---

### 3. `plot_reconstruction_error_distribution(errors, y_true, threshold=None)`
Plot histogram of reconstruction errors, separated by label.

**Highlights:**
- Helps visually inspect overlap between normal and anomaly distributions
- Optional threshold line displayed

---

### 4. `visualize_reconstruction(model, X, n=5)`
Visualizes a few input-output reconstruction pairs.

**Parameters:**
- `model`: Trained AE
- `X`: Input features
- `n`: Number of samples to display

**Returns:**
- Side-by-side line plots comparing original and reconstructed features

---

## 🧪 Recommended Usage

```python
from ae_evaluation import *

errors = compute_reconstruction_error(model, X_test)
results = evaluate_anomaly_detection(errors, y_test)

print(results)
plot_reconstruction_error_distribution(errors, y_test, threshold=results['threshold'])
visualize_reconstruction(model, X_test.to_numpy(), n=5)
```

---

## 🗂 Suggested Location

Place this module in `src/utils/` and import it wherever evaluation is needed. Ensure input data is properly normalized and matches the training format of the AE.

---

## 📎 Notes

- Supports both manual and automatic thresholding strategies.
- Plots are based on Matplotlib, and run best in Jupyter/Colab.
- For VAE, similar logic applies with latent sampling considered separately.

---
