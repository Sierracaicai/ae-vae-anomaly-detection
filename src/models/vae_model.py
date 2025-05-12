import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, input_dim, encoding_dim=16, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.beta = beta
        self.encoder_hidden = layers.Dense(64, activation='relu')
        self.encoder_bn = layers.BatchNormalization()
        self.encoder_dropout = layers.Dropout(0.2)
        self.encoder_hidden2 = layers.Dense(32, activation='relu')
        self.z_mean_layer = layers.Dense(encoding_dim, name='z_mean')
        self.z_log_var_layer = layers.Dense(encoding_dim, name='z_log_var')
        self.sampling = Sampling()

        self.decoder_hidden = layers.Dense(32, activation='relu')
        self.decoder_bn = layers.BatchNormalization()
        self.decoder_dropout = layers.Dropout(0.2)
        self.decoder_hidden2 = layers.Dense(64, activation='relu')
        self.decoder_output = layers.Dense(input_dim, activation='linear')

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x = self.encoder_hidden(inputs)
        x = self.encoder_bn(x)
        x = self.encoder_dropout(x)
        x = self.encoder_hidden2(x)

        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling((z_mean, z_log_var))

        x = self.decoder_hidden(z)
        x = self.decoder_bn(x)
        x = self.decoder_dropout(x)
        x = self.decoder_hidden2(x)
        return self.decoder_output(x)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            x = self.encoder_hidden(data)
            x = self.encoder_bn(x)
            x = self.encoder_dropout(x)
            x = self.encoder_hidden2(x)

            z_mean = self.z_mean_layer(x)
            z_log_var = self.z_log_var_layer(x)
            z = self.sampling((z_mean, z_log_var))

            x_decoded = self.decoder_hidden(z)
            x_decoded = self.decoder_bn(x_decoded)
            x_decoded = self.decoder_dropout(x_decoded)
            x_decoded = self.decoder_hidden2(x_decoded)
            output = self.decoder_output(x_decoded)

            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - output), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        x = self.encoder_hidden(data)
        x = self.encoder_bn(x, training=False)
        x = self.encoder_dropout(x, training=False)
        x = self.encoder_hidden2(x)

        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling((z_mean, z_log_var))

        x_decoded = self.decoder_hidden(z)
        x_decoded = self.decoder_bn(x_decoded, training=False)
        x_decoded = self.decoder_dropout(x_decoded, training=False)
        x_decoded = self.decoder_hidden2(x_decoded)
        output = self.decoder_output(x_decoded)

        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - output), axis=1))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

def train_vae(model, X_train, X_val,
              batch_size=64,
              epochs=100,
              learning_rate=1e-3,
              save_path='best_vae.h5',
              kl_scheduler=None):

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    class BetaCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if kl_scheduler:
                model.beta = kl_scheduler.get_beta(epoch)

    callbacks = [es, mc, lr_scheduler, BetaCallback()] if kl_scheduler else [es, mc, lr_scheduler]

    history = model.fit(
        X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )

    return history, model

def plot_vae_history(history, save_path='vae_loss_plot.png'):
    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, h['loss'],     label='Train Total Loss')
    plt.plot(epochs, h['val_loss'], label='Val Total Loss')
    plt.title('VAE Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, h['recon_loss'],     label='Train Recon Loss')
    plt.plot(epochs, h['val_recon_loss'], label='Val Recon Loss')
    plt.plot(epochs, h['kl_loss'],        label='Train KL Loss')
    plt.plot(epochs, h['val_kl_loss'],    label='Val KL Loss')
    plt.title('Reconstruction & KL Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f'✅ VAE loss 曲线已保存到 {save_path}')

class KLAnnealing:
    def __init__(self, start_beta=0.0, target_beta=1.0, n_epochs=10):
        self.start_beta = start_beta
        self.target_beta = target_beta
        self.n_epochs = n_epochs

    def get_beta(self, epoch):
        if epoch >= self.n_epochs:
            return self.target_beta
        return self.start_beta + (self.target_beta - self.start_beta) * (epoch / self.n_epochs)
