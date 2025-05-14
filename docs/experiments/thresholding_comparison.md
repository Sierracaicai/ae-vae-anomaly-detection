
# 🔎 Thresholding Strategy Comparison Report

This report compares two threshold selection strategies for anomaly detection using Autoencoder reconstruction errors:

---

## 📌 1. Percentile-based Threshold (Baseline)

**Method:**  
Select threshold based on a fixed percentile (e.g., 95%) of reconstruction error.

**Result:**
```
Confusion Matrix:
TP: 2814
FP: 1807
TN: 123495
FN:   42
```

- **Precision**: 0.6088  
- **Recall**: 0.9853  
- **F1 Score**: 0.7497

---

## 📌 2. Best F1 Threshold on Validation Set

**Method:**  
Evaluate a range of thresholds on a labeled validation set. Choose the one that maximizes the F1 score.

**Result:**
```
Confusion Matrix:
TP: 2250
FP: 1479
TN: 98763
FN:   35
```

- **Precision**: 0.6034  
- **Recall**: 0.9847  
- **F1 Score**: 0.7442

---

## ✅ Conclusion

While both methods perform comparably, the percentile-based method shows **slightly better F1** and **higher recall**.

Thus, **for stability and generalization**, the **percentile method is recommended as the default**, while **best-F1 thresholding** can be retained as an optional evaluation tool for validation experiments.

---
