# 🧪 Autoencoder Evaluation Report

**Date:** 2025-05-09  
**Model:** Basic Autoencoder  
**Dataset:** UNSW-NB15 (cleaned subset)  
**Samples:** 640,788 → used subset for test  
**Features:** 178

---

## ✅ Evaluation Metrics

| Metric        | Value     |
|---------------|-----------|
| Threshold     | 0.00389   |
| Precision     | 0.6046    |
| Recall        | 0.9860    |
| F1 Score      | 0.7496    |
| ROC AUC       | 0.9910    |

### Confusion Matrix

|        | Pred: Normal | Pred: Anomaly |
|--------|--------------|----------------|
| **Actual: Normal** | TN = 49,383   | FP = 737         |
| **Actual: Anomaly**| FN = 16       | TP = 1,127       |

---

## 📈 Reconstruction Error Distribution

- **Normal samples**: mostly low error, tight cluster near zero
- **Anomalies**: higher reconstruction errors, right-shifted tail
- **Threshold**: 0.00389 chosen via F1 score optimization
- ✅ Clear distinction with acceptable overlap

---

## 🧩 Sample Reconstructions

- 5 randomly selected samples plotted (Original vs Reconstructed)
- Most samples show:
  - Good preservation of shape
  - Accurate reconstruction of high-value peaks
- Some deviations indicate potential anomaly learnability

---

## 📌 Analysis Summary

- AE model **successfully detects majority of anomalies** (recall 98.6%)
- Slightly high false positives due to overlapping low-error outliers
- Suitable for:
  - Pre-screening
  - Network anomaly detection
  - Risk prioritization

---

## 🛠️ Recommendations

- Add regularization (L1/L2)
- Tune encoding dimension or deepen the architecture
- Explore:
  - Sample weighting
  - Post-filtering on FP cases
  - VAE for probabilistic interpretation

---

## 🔗 Visuals

- `ae_training_plot.png`: Loss/MAE curve  
- `recon_error_hist.png`: Reconstruction error distribution  
- `reconstructed_samples.png`: Input vs Output plots

---

_This report was automatically generated from current experiment results._
