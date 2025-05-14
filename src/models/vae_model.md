
# Variational Autoencoder (VAE) with KL Annealing

This document explains the implementation of a Variational Autoencoder (VAE) model with support for adjustable KL divergence weight (β-VAE) and KL annealing.

---

## 🧠 Model Structure

The model consists of:

- **Encoder**: Two fully connected hidden layers with ReLU activations, batch normalization, and dropout.
- **Latent Space**: Outputs `z_mean` and `z_log_var` used by a custom `Sampling` layer.
- **Decoder**: Symmetric to the encoder.
- **Losses**:
  - **Reconstruction Loss**: MSE between input and output.
  - **KL Divergence Loss**: Weighted by configurable `beta`.

---

## 🔧 Key Classes & Functions

### `Sampling`
```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```
Samples latent variable `z` from Gaussian distribution.

---

### `VAE(Model)`
```python
class VAE(tf.keras.Model):
    def __init__(self, input_dim, encoding_dim=16, beta=1.0)
```
- Custom `train_step` & `test_step` handle loss calculation.
- Supports dynamic `beta` adjustment for β-VAE or KL annealing.

---

## 🔁 KL Annealing

```python
class KLAnnealing:
    def __init__(self, start_beta=0.0, target_beta=1.0, n_epochs=10)
```
- Returns increasing beta values per epoch.
- Plug into training using a callback.

---

## 🚀 Training Function

```python
def train_vae(model, X_train, X_val, batch_size=64, epochs=100, ...)
```
- EarlyStopping, ModelCheckpoint, ReduceLROnPlateau support.
- KL annealing enabled with optional `kl_scheduler`.

---

## 📊 Visualization

```python
def plot_vae_history(history, save_path='vae_loss_plot.png')
```
- Plots total loss, reconstruction loss, and KL divergence over epochs.

---

## 📎 Usage Example

```python
model = VAE(input_dim=178, encoding_dim=16, beta=1.0)
scheduler = KLAnnealing(start_beta=0.0, target_beta=1.0, n_epochs=20)
history, model = train_vae(model, X_train, X_val, kl_scheduler=scheduler)
plot_vae_history(history)
```

---

✅ Designed for anomaly detection with interpretability and flexible regularization.
