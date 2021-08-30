import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

from tensorflow.python.keras.utils import conv_utils




# classes

class LayerNorm(Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """

    def __init__(self, scale_initializer='ones', bias_initializer='zeros', **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = 1e-6
        self.scale_initializer = initializers.get(scale_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],),
                                     initializer=self.scale_initializer,
                                     trainable=True,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    trainable=True,
                                    name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        norm = (x - mean) * (1 / (std + self.epsilon))
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape



class SincConv1D(Layer):

    def __init__(
            self,
            N_filt,
            Filt_dim,
            fs,
            stride=128,
            padding="SAME",
            **kwargs):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.stride = stride
        self.padding = padding

        super(SincConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name='filt_b1',
            shape=(self.N_filt, 1),
            initializer='uniform',
            trainable=True)
        self.filt_band = self.add_weight(
            name='filt_band',
            shape=(self.N_filt, 1),
            initializer='uniform',
            trainable=True)

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.B1 = np.expand_dims(b1, axis=-1)
        self.B2 = np.expand_dims(b2, axis=-1)
        self.freq_scale = self.fs * 1.0

        t_right = tf.constant(tf.linspace(1.0, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2)) / self.fs, tf.float32)
        self.T_Right = tf.tile(tf.expand_dims(t_right, axis=0), (self.N_filt, 1))

        n = tf.linspace(0, self.Filt_dim - 1, self.Filt_dim)
        window = 0.54 - 0.46 * tf.cos(2 * math.pi * n / self.Filt_dim)
        window = tf.cast(window, tf.float32)
        self.Window = tf.tile(tf.expand_dims(window, axis=0), (self.N_filt, 1))


        self.set_weights([self.B1 / self.freq_scale,
                         (self.B2 - self.B1) / self.freq_scale])
        


        super(SincConv1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        
        min_freq = 50.0;
        min_band = 50.0;
        
        filt_beg_freq = tf.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (tf.abs(self.filt_band) + min_band / self.freq_scale)

        

        low_pass1 = 2 * filt_beg_freq * self.sinc(filt_beg_freq * self.freq_scale)
        low_pass2 = 2 * filt_end_freq * self.sinc(filt_end_freq * self.freq_scale)
        band_pass = (low_pass2 - low_pass1)
        band_pass = band_pass / tf.reduce_max(band_pass, axis=1, keepdims=True)
        windowed_band_pass = band_pass * self.Window
        
        filters = tf.transpose(windowed_band_pass)
        filters = tf.reshape(filters, (self.Filt_dim, 1, self.N_filt))

        # Do the convolution.
        out = tf.nn.conv1d(
            x,
            filters=filters,
            stride=self.stride,
            padding=self.padding
        )

        return out

    def sinc(self, band):
        y_right = tf.sin(2 * math.pi * band * self.T_Right) / (2 * math.pi * band * self.T_Right)
        y_left = tf.reverse(y_right, [1])
        y = tf.concat([y_left, tf.ones((self.N_filt, 1)), y_right], axis=1)
        return y

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.Filt_dim,
            padding=self.padding.lower(),
            stride=self.stride,
            dilation=1)
        return (input_shape[0],) + (new_size,) + (self.N_filt,)
