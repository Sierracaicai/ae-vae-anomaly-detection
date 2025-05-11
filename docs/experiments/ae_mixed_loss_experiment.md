
# 🔬 Experiment: Mixed Loss Function in Autoencoder

## 📌 Objective
To evaluate whether introducing a mixed reconstruction loss (standard MSE + feature-weighted MSE) can improve anomaly detection performance in AE models.

## ⚙️ Loss Function
We used a custom loss:

\[
\text{Loss} = \text{MSE}(x, \hat{x}) + \alpha \cdot \text{WeightedMSE}(x, \hat{x})
\]

Where `WeightedMSE` is the squared error weighted by the true feature values.

- `alpha = 0.3`
- Base: Mean Squared Error (MSE)
- Weighted Term: Emphasizes higher-valued features

## 🧪 Experimental Setup

| Component     | Configuration                  |
|---------------|-------------------------------|
| Model Type    | Shallow Autoencoder            |
| Encoder/Decoder | [64 → 32]                     |
| Bottleneck Dim | 16                            |
| Activation    | tanh                           |
| Dropout       | 0.2                            |
| Optimizer     | Adam (lr=1e-3)                 |
| Batch Size    | 64                             |
| Epochs        | 100 (EarlyStopping enabled)    |
| Dataset       | UNSW-NB15 (Preprocessed)       |
| Split         | Normal (Train/Val), Mixed (Test) |

---

## 📊 Evaluation Metrics

| Metric         | Mixed Loss AE| Basic AE |
|----------------|--------------|---------------|
| Threshold      | 0.00439      | 0.00454       |
| Precision      | 0.6013       | **0.6050**    |
| Recall         | **0.9965**   | **0.9965**    |
| F1-Score       | 0.7500       | **0.7529**    |
| ROC-AUC        | 0.9903       | **0.9907**    |
| TP             | 288          | 288           |
| FP             | 191          | **188**       |
| TN             | 12336        | **12339**     |
| FN             | 1            | 1             |

---

## 📈 Visualizations

> (Insert the reconstruction error histogram, confusion matrix, ROC curve, or loss curve images here if uploading to GitHub. e.g.)

```bash
docs/experiments/ae_mixed_loss_eval/
├── reconstruction_hist.png
├── confusion_matrix.png
├── roc_curve.png
```

---

## 📌 Conclusion

- **Basic AE** showed slightly better **precision**, **F1**, and **AUC**, while recall is the same as the mixed loss AE.
- The improvement is **slight**, suggesting that weighted features contribute marginally in this context.
- **Recommendation**: Keep using basic AE and will use Mixed loss when guided by domain knowledge (e.g. feature importance via SHAP).

