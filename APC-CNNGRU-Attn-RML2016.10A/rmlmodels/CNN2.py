"""CLDNNLike model for RadioML with Two-Head Temporal Attention."""
import os
from keras import backend as K
from keras.models import Model
from keras.layers import GRU, Input, Dense, Dropout, Conv2D, MaxPool2D, SeparableConv2D, Activation, \
    Layer, Lambda, GlobalAveragePooling2D, GaussianDropout, Reshape, LayerNormalization, GlobalAveragePooling1D
from keras.layers import Bidirectional, Flatten, Multiply
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate
from tensorflow.keras import regularizers

class TwoHeadTemporalAttention(layers.Layer):
    def __init__(self, units=64, heads=2, **kwargs):
        super().__init__(**kwargs)
        if units % heads != 0:
            raise ValueError(f"units ({units}) must be divisible by heads ({heads})")
        self.units = units
        self.heads = heads
        self.head_dim = units // heads


        self.W_q = layers.Dense(units)
        self.W_k = layers.Dense(units)
        self.W_v = layers.Dense(units)
        self.W_o = layers.Dense(units)


        self.attention_weights = None

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]

        q = self.W_q(inputs)
        k = self.W_k(inputs)
        v = self.W_v(inputs)


        q = tf.reshape(q, (batch_size, T, self.heads, self.head_dim))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (batch_size, T, self.heads, self.head_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, (batch_size, T, self.heads, self.head_dim))
        v = tf.transpose(v, perm=[0, 2, 1, 3])


        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))


        attention_weights = tf.nn.softmax(scores, axis=-1)
        self.attention_weights = tf.identity(attention_weights, name="attention_weights")

        output = tf.matmul(self.attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, T, self.units))
        return self.W_o(output)

    def get_attention_weights(self):

        return self.attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "heads": self.heads})
        return config

def phase_correction(x):
    i_signal = x[:, 0, :]
    q_signal = x[:, 1, :]
    iq_complex = tf.complex(i_signal, q_signal)

    phase = tf.math.angle(iq_complex)
    mean_phase = tf.reduce_mean(phase, axis=-1, keepdims=True)
    correction_factor = tf.exp(-1j * tf.cast(mean_phase, tf.complex64))
    corrected_complex = iq_complex * correction_factor

    i_real = tf.math.real(corrected_complex)
    q_imag = tf.math.imag(corrected_complex)
    corrected_iq = tf.stack([i_real, q_imag], axis=1)
    corrected_iq = tf.cast(corrected_iq, dtype=tf.float32)
    return corrected_iq

def CNN2(weights=None, input_shape=(2, 128), classes=11, **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The weights argument should be either None or a valid path.')

    input = layers.Input(shape=input_shape)
    x = layers.Lambda(phase_correction)(input)
    x = layers.Permute((2, 1))(x)


    x = layers.SeparableConv1D(64, 3, padding='same', activation='relu', depthwise_initializer='he_normal')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GaussianDropout(0.4)(x)


    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = layers.LayerNormalization()(x)

    x = layers.SeparableConv1D(128, 3, padding='same', activation='relu', depthwise_initializer='he_normal')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GaussianDropout(0.4)(x)


    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = layers.LayerNormalization()(x)


    x = TwoHeadTemporalAttention(units=128, heads=2)(x)
    x = layers.LayerNormalization()(x)


    x = layers.GlobalAveragePooling1D()(x)


    x = layers.Dense(classes, activation='softmax', kernel_initializer='lecun_normal')(x)

    model = Model(inputs=input, outputs=x)
    if weights is not None:
        model.load_weights(weights)
    return model