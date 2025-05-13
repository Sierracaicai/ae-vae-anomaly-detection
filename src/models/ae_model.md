
# Autoencoder (AE) Model

This module implements flexible Autoencoder architectures with customizable depth, activation functions, loss configurations, and training strategies, tailored for anomaly detection tasks.

---

## 🔧 Functions

### `build_ae(...)`
Construct a modular Autoencoder model.

**Parameters**:
- `input_dim`: Number of input features.
- `encoding_dim`: Size of bottleneck latent layer (default: 16).
- `hidden_dims`: List of hidden layer sizes for encoder and decoder (default: [64, 32]).
- `dropout_rate`: Dropout rate applied after each layer (default: 0.2).
- `use_batchnorm`: Whether to apply BatchNormalization after layers.
- `activation`: Activation function for each layer; supports `'relu'`, `'tanh'`, `'elu'`, `'selu'`, `'leaky_relu'`.

### `build_ae_deep(...)`
Build a deeper Autoencoder architecture with hard-coded deeper layers for comparative experiments.

---

## 🏋️ Training Utilities

### `train_autoencoder(...)`
Train AE using standard MSE loss and Adam optimizer. Includes:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

### `train_autoencoder_with_optimizer(...)`
Support custom optimizer (e.g., AdamW, Ranger). Usage is otherwise similar to `train_autoencoder`.

### `train_autoencoder_mixed_loss(...)`
Train AE with custom loss function:  
> **Loss = MSE + α × (Weighted MSE)**

Where α is a tunable scalar (default = 0.3) for emphasizing certain input features.

---

## 📊 Visualization

### `plot_training_history(...)`
Plot loss and MAE curves over training epochs.

---

## 🔍 Notes

- All models use `'linear'` activation for the output layer.
- Encoders and decoders are symmetric by default.
- Supports dropout and batch normalization, adjustable per experiment.
- Recommended for use in combination with anomaly thresholding methods defined elsewhere.

---

✅ Designed for reproducible and interpretable anomaly detection workflows.
