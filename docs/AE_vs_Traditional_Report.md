# Anomaly Detection Model Comparison Report

## ✅ Final AutoEncoder (AE) Model (Selected)

- **Architecture**: Shallow AE  
- **Activation**: `tanh`
- **Optimizer**: `AdamW`
- **Thresholding**: Percentile method  
- **Training Data**: Full dataset

### 📈 Performance

| Metric      | Value     |
|-------------|-----------|
| Precision   | 0.6090    |
| Recall      | 0.9853    |
| F1 Score    | 0.7527    |
| ROC-AUC     | 0.9908    |
| Confusion Matrix | TP: 2814, FP: 1807, TN: 123495, FN: 42 |

---

## 📊 Traditional Anomaly Detection Models

| Method               | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix |
|----------------------|-----------|--------|----------|---------|------------------|
| One-Class SVM        | 0.5539    | 0.5452 | 0.5495   | 0.7676  | TP: 1557, FP: 1254, TN: 124048, FN: 1299 |
| Local Outlier Factor | 0.5539    | 0.5431 | 0.5484   | 0.7665  | TP: 1551, FP: 1249, TN: 124053, FN: 1305 |
| Elliptic Envelope    | 0.4389    | 0.3407 | 0.3836   | 0.6654  | TP: 973,  FP: 1244, TN: 124058, FN: 1883 |
| Isolation Forest     | 0.3260    | 0.2122 | 0.2571   | 0.6011  | TP: 606,  FP: 1253, TN: 124049, FN: 2250 |

---

## 🧠 Key Insights

- **AutoEncoder** achieved the **highest F1 score** and **ROC-AUC**, particularly excelling in **recall** (near-perfect detection of anomalies).
- **One-Class SVM** and **LOF** were the best among traditional methods but still significantly underperform AE.
- **Elliptic Envelope** and **Isolation Forest** struggled with either precision or recall, reflecting their assumptions not fitting well for this dataset.

---

## ✅ Conclusion

The selected AutoEncoder model (shallow, tanh, AdamW) consistently outperforms traditional methods in both **precision-recall tradeoff** and **overall detection quality**. It is the recommended solution for anomaly detection on this dataset.