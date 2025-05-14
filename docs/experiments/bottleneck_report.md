
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
| 4              | 0.5297    | 0.9862 | 0.6892   | **0.9898**  | 285 | 253  | 12274 | 4 |
| 8              | **0.5504**    | 0.9827 | **0.7056**   | 0.9889  | 284 | 232 | 12295 | 5 |
| 16             | 0.5274| **1.0** | 0.6906 | 0.9893  | 289 | 259  | 12268  | **0**   |
| 32             | 0.5062    | 0.9862 | 0.6690   | 0.9886  | 285 | 278 | 12249 | 4  |

## 📈 Key Observations

- **Bottleneck = 16** provides the best balance of high recall and strong F1 score, making it the optimal choice in most cases.
- **Bottleneck = 8** also performs well, slightly improving Precision and F1 score at the cost of more false negatives.
- Other two dimensions (4 or 32) show signs of underfitting, losing anomaly detection ability due to excessive compression.

## ✅ Recommendation

- Use **16 dimensions** as the default setting for bottleneck size.
- Consider **8 dimensions** for lightweight deployment.
- Avoid **4 dimensions** unless for speed testing only.

---
