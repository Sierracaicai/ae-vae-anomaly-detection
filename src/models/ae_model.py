"""
ae_model.py

Modular Autoencoder (AE) model builder, trainer, and visualizer for anomaly detection.
"""
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, LeakyReLU
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
        activation (str): One of ['relu', 'tanh', 'elu', 'selu', 'leaky_relu'].

    Returns:
        keras.Model: Compiled AE model (untrained).
    """
    def apply_activation(x, act):
        if act == 'leaky_relu':
            return LeakyReLU(alpha=0.1)(x)
        else:
            return tensorflow.keras.layers.Activation(act)(x)

    inp = Input(shape=(input_dim,))
    x = inp

    # Encoder
    for dim in hidden_dims:
        x = Dense(dim)(x)
        x = apply_activation(x, activation)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    bottleneck = Dense(encoding_dim)(x)
    bottleneck = apply_activation(bottleneck, activation)

    # Decoder
    x = bottleneck
    for dim in reversed(hidden_dims):
        x = Dense(dim)(x)
        x = apply_activation(x, activation)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    out = Dense(input_dim, activation='linear')(x)

    return Model(inputs=inp, outputs=out)


def build_ae_deep(input_dim: int, encoding_dim: int = 16, dropout_rate: float = 0.2) -> Model:
    """
    Build a deeper Autoencoder model with more hidden layers and optional BatchNorm/Dropout.

    Parameters:
        input_dim (int): Number of input features.
        encoding_dim (int): Dimension of bottleneck latent space.
        dropout_rate (float): Dropout rate after each layer (default 0.2).

    Returns:
        keras.Model: Compiled autoencoder model.
    """
    inp = Input(shape=(input_dim,), name="input")

    # Encoder
    x = Dense(128, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(x)

    # Decoder
    x = Dense(32, activation='relu')(bottleneck)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    out = Dense(input_dim, activation='linear', name="output")(x)

    return Model(inp, out, name="DeepAutoencoder")


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

def train_autoencoder_with_optimizer(model,
                      X_train,
                      X_val,
                      optimizer,
                      batch_size: int = 64,
                      epochs: int = 100,
                      save_path: str = 'best_ae_optimizer.h5'):
    """
    Train the Autoencoder model with early stopping and LR scheduling.

    Parameters:
        model (keras.Model): Compiled AE model.
        X_train (np.array): Training data (only normal samples).
        X_val (np.array): Validation data (only normal samples).
        optimizer: Keras optimizer instance (e.g., Adam, AdamW, Ranger).
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.
        save_path (str): Path to save the best model (.h5 file).

    Returns:
        history (keras.callbacks.History): Training history.
        model (keras.Model): Trained model with best weights restored.
    """
    model.compile(
        optimizer=optimizer,
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




def train_autoencoder_mixed_loss(model, 
                                  X_train, 
                                  X_val, 
                                  alpha=0.3, 
                                  learning_rate=1e-3,
                                  batch_size=64,
                                  epochs=100,
                                  save_path='best_ae_mixed.h5'):
    """
    Train Autoencoder with combined reconstruction loss:
        loss = MSE + α × weighted MSE

    Parameters:
        model (keras.Model): AE model to train
        X_train (np.ndarray): Training features (normal only)
        X_val (np.ndarray): Validation features (normal only)
        alpha (float): Weight for additional MSE term
        save_path (str): File to save best model (.h5)

    Returns:
        history: training history object
        model: trained model (best checkpoint)
    """

    def mixed_mse_loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        weighted = K.mean(K.square(y_true - y_pred) * y_true, axis=1)
        return mse + alpha * weighted

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=mixed_mse_loss,
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        batch_size=batch_size,
        epochs=epochs,
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
