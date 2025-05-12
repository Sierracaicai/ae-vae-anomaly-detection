
# 📊 Optimizer Comparison Report (Basic AE with MSE Loss)

## 🔧 Experiment Setup

- **Model**: Shallow Autoencoder  
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizers Evaluated**: Adam, AdamW, SGD  
- **Evaluation Metric**: Based on best F1 score using fixed threshold selection  

---

## 📈 Results Summary

| Optimizer | Threshold | Precision | Recall | F1 Score | ROC-AUC | TP / FP / FN |
|-----------|-----------|-----------|--------|----------|---------|--------------------------|
| **Adam**   | 0.00256   | 0.515     | 0.993  | 0.677     | 0.9887  | 287 / 272 / 2            |
| **AdamW**  | 0.00452   | **0.581** | **1.000** | **0.735** | **0.9903** | **289 / 208 / 0**     |
| **SGD**    | 0.01065   | 0.489     | 0.837  | 0.622     | 0.9818  | 242 / 247 / 47           |

---

## ✅ Analysis

- **AdamW** outperforms all other optimizers with the **highest F1 score (0.735)** and **perfect recall (1.000)**, indicating strong generalization with minimal false negatives (FN=0).
- **Adam** offers strong recall (0.993) but slightly lower precision, leading to more false positives (FP=272).
- **SGD** underperforms with the **lowest recall (0.837)** and highest number of false negatives (FN=47), making it less suitable for anomaly detection tasks where recall is critical.

---

## 🧪 Recommendation

> ✅ **AdamW is the most balanced and effective optimizer** for this AE configuration.  
> ⚠️ Avoid using **SGD** in sensitive tasks where missing anomalies is costly.
