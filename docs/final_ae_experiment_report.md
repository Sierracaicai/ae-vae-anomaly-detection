# 📊 Autoencoder (AE) Experiment Summary Report

This report summarizes all experimental findings from the Autoencoder (AE) based anomaly detection system, including architectural design, training strategies, and final evaluation metrics.

---

## ✅ Final Selected AE Configuration

- **Model Type**: Shallow Autoencoder
- **Encoder/Decoder Layers**: [64, 32] (symmetric)
- **Bottleneck Dimension**: 16
- **Activation Function**: `tanh`
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: `AdamW`
- **Thresholding Method**: Quantile-based (95th percentile)
- **Training Data**: Full cleaned dataset (~640,000 rows)

---

## 🧪 Optimization Strategy & Experiments

### 1. 🔍 Architecture Experiments
- Compared **shallow vs deep AE**
- Shallow AE consistently produced **higher recall** and more balanced performance.
- Best depth: 2-layer encoder and decoder with symmetric structure.

### 2. 🎯 Bottleneck Dimension
- Tested: `[4, 8, 16, 32]`
- Bottleneck = 16 gave the best F1-recall tradeoff.

### 3. ⚡ Activation Functions
- Compared: `relu`, `tanh`, `elu`, `selu`, `leaky_relu`
- `tanh` gave the **best F1 score and stability**.

### 4. 🧠 Loss Function
- Mixed MSE (MSE + weighted MSE) was tested.
- Minor improvements in some metrics, but base MSE performed more reliably across data splits.

### 5. 🛠 Optimizers
- Compared: `Adam`, `AdamW`, `SGD`, `Ranger`
- `AdamW` consistently yielded **best balance of precision/recall** and smooth convergence.

### 6. 🔎 Thresholding Strategy
- Quantile-based (default 95th percentile) used as baseline.
- F1-optimized threshold and PR-curve methods were tested but **offered no consistent improvement**.

---

## 📈 Final Model Evaluation

| Metric         | Value       |
|----------------|-------------|
| **Threshold**  | 0.00407     |
| **Precision**  | 0.609       |
| **Recall**     | 0.985       |
| **F1 Score**   | 0.753       |
| **ROC AUC**    | 0.991       |
| **TP / FP**    | 2814 / 1807 |
| **TN / FN**    | 123495 / 42 |

---

## 📌 Conclusion

After iterative optimization, the final AE model demonstrates strong anomaly detection performance with excellent recall and solid precision. The model is:
- **Lightweight** (shallow, fast to train)
- **Robust** to noise
- **Well-suited** to real-world anomaly detection tasks.