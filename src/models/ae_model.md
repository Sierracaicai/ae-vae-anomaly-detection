# AE Model Module (`ae_model.py`)

This module provides a reusable, customizable implementation of a basic Autoencoder (AE) model for anomaly detection tasks.

## 📦 Functions

### `build_ae(...)`

Creates a configurable Autoencoder model.

**Parameters:**
- `input_dim` (int): Number of features in the input.
- `encoding_dim` (int): Bottleneck dimension.
- `hidden_dims` (list): List of hidden layer sizes for encoder/decoder (e.g., `[64, 32]`).
- `dropout_rate` (float): Dropout ratio applied after batchnorm.
- `use_batchnorm` (bool): Whether to include batch normalization layers.
- `activation` (str): Activation function (default: `'relu'`).

**Returns:** Keras `Model`

---

### `train_autoencoder(...)`

Trains the AE model using MSE loss and Adam optimizer, with early stopping and learning rate scheduler.

**Parameters:**
- `model`: Keras AE model.
- `X_train`: Normal training samples.
- `X_val`: Normal validation samples.
- `batch_size`, `epochs`, `learning_rate`: Training hyperparameters.
- `save_path`: Path to save best model weights.

**Returns:** `(history, model)`

---

### `plot_training_history(...)`

Plots and saves training loss and MAE over epochs.

---

## 🔁 Example Usage

```python
from models.ae_model import build_ae, train_autoencoder, plot_training_history

model = build_ae(input_dim=X_train.shape[1], encoding_dim=16)
history, model = train_autoencoder(model, X_train, X_val)
plot_training_history(history)
```

---

## 📁 Location

Place this file in:

```
src/
└── models/
    └── ae_model.py
```

Then import from `models.ae_model` in your notebooks.

---

## 🧠 Why This Module?

This separates the modeling logic from your notebooks, enabling:

- Easy reuse across experiments
- Consistent architecture trials (depth, dropout, batchnorm)
- Clean training/plotting interface
- Simpler GitHub structure
