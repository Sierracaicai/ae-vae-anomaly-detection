
# 🔬 Activation Function Experiment Report

This report summarizes the performance of different activation functions on the shallow Autoencoder (AE) model using the same training, validation, and test splits. The goal is to evaluate which activation function provides the best trade-off between precision, recall, and overall reconstruction-based anomaly detection performance.

---

## 📊 Experiment Summary Table

| Activation  | Precision | Recall  | F1     | AUC     | FP  | FN | Total Errors |
|-------------|-----------|---------|--------|---------|-----|----|---------------|
| **tanh**     | 0.602     | 0.993   | **0.749** | **0.9908** | 190 | 2  | **192** ✅ |
| **selu**     | 0.599     | 0.993   | 0.747  | 0.9907 | 192 | 2  | 194 |
| relu        | 0.582     | **0.997** | 0.735  | 0.9906 | 207 | 1  | 208 |
| leaky_relu  | 0.511     | 0.993   | 0.675  | 0.9885 | 275 | 2  | 277 |
| elu         | 0.502     | 0.972   | 0.662  | 0.9876 | 279 | 8  | 287 |

---

## ✅ Conclusion

- **Best Overall**: `tanh`
  - Achieved the highest F1-score (0.749) and one of the best AUC scores (0.9908).
  - Balanced between high recall and relatively low false positives.
- **Runner-up**: `selu`
  - Very close to `tanh` in all metrics and a good alternative.
- **relu** achieved perfect recall but suffered in precision, leading to more false positives.
- `leaky_relu` and `elu` performed noticeably worse due to more false positives and lower precision.

---

## 🔍 Interpretation

- `tanh` is ideal in this normalized input setting due to its smooth gradient behavior and ability to capture nuanced input changes.
- `selu` also works well thanks to its self-normalizing properties.

We recommend using `tanh` as the default activation for future AE experiments.

