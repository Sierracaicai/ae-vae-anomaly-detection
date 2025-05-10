
# Bottleneck Dimension Tuning Experiment

This experiment evaluates the effect of different bottleneck dimensions on the performance of a shallow Autoencoder model for anomaly detection on the UNSW-NB15 dataset.

## 🔧 Experiment Setup

- Dataset: UNSW-NB15 (Preprocessed, scaled, encoded)
- Model: Shallow Autoencoder
- Evaluation Metric: Precision, Recall, F1 Score, ROC-AUC
- Bottleneck Dimensions Tested: 4, 8, 16, 32

## 📊 Results Summary

| Bottleneck Dim | Precision | Recall | F1 Score | ROC-AUC | TP   | FP   | TN     | FN   |
|----------------|-----------|--------|----------|---------|------|------|--------|------|
| 4              | 0.6162    | 0.5154 | 0.5613   | 0.9786  | 1472 | 917  | 124385 | 1384 |
| 8              | 0.5591    | 0.6208 | 0.5884   | 0.9868  | 1773 | 1398 | 123904 | 1083 |
| 16             | **0.6046**| **0.9860** | **0.7496** | 0.9910  | 1127 | 737  | 49383  | 16   |
| 32             | 0.5706    | 0.9667 | 0.7176   | **0.9912**  | 2761 | 2078 | 123224 | 95   |

## 📈 Key Observations

- **Bottleneck = 16** provides the best balance of high recall and strong F1 score, making it the optimal choice in most cases.
- **Bottleneck = 32** also performs well, slightly improving recall and AUC at the cost of more false positives.
- Smaller dimensions (4 or 8) show signs of underfitting, losing anomaly detection ability due to excessive compression.

## ✅ Recommendation

- Use **16 dimensions** as the default setting for bottleneck size.
- Consider **8 dimensions** for lightweight deployment.
- Avoid **4 dimensions** unless for speed testing only.

---

_Report generated automatically from CSV and visual analysis._
