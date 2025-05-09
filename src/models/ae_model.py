"""
ae_model.py

Modular Autoencoder (AE) model builder, trainer, and visualizer for anomaly detection.
"""

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


def build_ae(input_dim: int,
             encoding_dim: int = 16,
             hidden_dims: list = [64, 32],
             dropout_rate: float = 0.2,
             use_batchnorm: bool = True,
             activation: str = 'relu') -> Model:
    """
    Build an Autoencoder (AE) model with configurable depth, dropout, and batchnorm.

    Parameters:
        input_dim (int): Number of input features.
        encoding_dim (int): Dimension of the bottleneck layer.
        hidden_dims (list): List of hidden layer sizes for encoder and decoder.
        dropout_rate (float): Dropout rate (applied after batchnorm).
        use_batchnorm (bool): Whether to include BatchNormalization layers.
        activation (str): Activation function used in all layers (except final).

    Returns:
        keras.Model: Compiled AE model (untrained).
    """
    inp = Input((input_dim,))
    x = inp

    # Encoder
    for dim in hidden_dims:
        x = Dense(dim, activation=activation)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    bottleneck = Dense(encoding_dim, activation=activation, name='bottleneck')(x)

    # Decoder
    x = bottleneck
    for dim in reversed(hidden_dims):
        x = Dense(dim, activation=activation)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    out = Dense(input_dim, activation='linear')(x)

    return Model(inp, out)


def train_autoencoder(model,
                      X_train,
                      X_val,
                      batch_size: int = 64,
                      epochs: int = 100,
                      learning_rate: float = 1e-3,
                      save_path: str = 'best_ae.h5'):
    """
    Train the Autoencoder model with early stopping and LR scheduling.

    Parameters:
        model (keras.Model): Compiled AE model.
        X_train (np.array): Training data (only normal samples).
        X_val (np.array): Validation data (only normal samples).
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        save_path (str): Path to save the best model (.h5 file).

    Returns:
        history (keras.callbacks.History): Training history.
        model (keras.Model): Trained model with best weights restored.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history, model


def plot_training_history(history, save_path: str = 'training_plot.png') -> None:
    """
    Plot training loss and MAE curves and save the figure.

    Parameters:
        history: Output from model.fit().
        save_path (str): Output file name to save the plot.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Over Epochs')
    plt.legend()

    # MAE curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Training plot saved as: {save_path}")
