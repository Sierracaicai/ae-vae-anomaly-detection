# 📦 Beta-VAE Model

This module defines a custom implementation of a β-VAE (Beta Variational Autoencoder), which introduces a weighting factor β on the KL-divergence to control disentanglement in the latent space.

---

## 🔧 Architecture

### 🧠 Encoder

- `Dense(64)` + `ReLU` → `BatchNorm` → `Dropout`
- `Dense(32)` + `ReLU`
- Latent:
  - `z_mean`: `Dense(encoding_dim)`
  - `z_log_var`: `Dense(encoding_dim)`
  - `z`: Sampled via `Sampling` layer using reparameterization trick

### 🧪 Decoder

- `Dense(32)` + `ReLU` → `BatchNorm` → `Dropout`
- `Dense(64)` + `ReLU`
- `Dense(input_dim)` with linear activation for reconstruction

---

## 🧮 Loss Function

The total loss is defined as:

\[
\mathcal{L} = \text{Reconstruction Loss} + \beta \cdot \text{KL Divergence}
\]

- **Reconstruction Loss**: MSE
- **KL Divergence**:
\[
-0.5 \sum(1 + \log(\sigma^2) - \mu^2 - \sigma^2)
\]

---

## ⚙️ Key Components

### 🔹 `Sampling` Layer

Reparameterization trick:
```python
z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

### 🔹 `BetaVAE` Class

Inherits from `tf.keras.Model`, and overrides:
- `call()`: For inference
- `train_step()` and `test_step()`: To apply custom loss with weighted KL term

Tracks:
- `total_loss`
- `reconstruction_loss`
- `kl_loss`

### 🔹 `KLAnnealing` Class

Supports warm-up strategy for β:
```python
beta = start_beta + (target_beta - start_beta) * (epoch / total_epochs)
```

---

## 🏋️ Training Utilities

### 🔸 `train_vae(...)`

- Compiles and trains the model with callbacks:
  - Early stopping
  - Model checkpoint
  - Learning rate scheduler

### 🔸 `plot_vae_history(...)`

- Visualizes loss curves:
  - Total Loss
  - Reconstruction Loss
  - KL Loss

---

## 📁 Output

- Best model saved to `.h5` file
- Loss curve saved to `vae_loss_plot.png`

---

## ✅ Summary

The β-VAE improves control over latent variable disentanglement by weighting the KL divergence term. This implementation supports:
- Custom β values
- β warm-up scheduling
- Custom training loop for loss monitoring
- Visualization of training dynamics
