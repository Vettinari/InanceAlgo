import tensorflow as tf
import numpy as np
import torch


class VolumeProfile(tf.keras.layers.Layer):
    def __init__(self, bins, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins

    def call(self, inputs, **kwargs):
        # Separate the prices and volumes
        prices, volumes = np.split(inputs, 2, axis=-1)
        price_bins = np.linspace(min(prices.flatten()), max(prices.flatten()),
                                 self.bins)  # adjust bin width as necessary

        volume_profile, _ = np.histogram(prices, bins=price_bins, weights=volumes)

        return tf.convert_to_tensor(volume_profile, dtype=tf.int64)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.bins


class TimeSeriesConv2D(tf.keras.layers.Conv2D):
    def __init__(self, features, current_timeframe, target_timeframe, filters, activation=None):
        tf_window_size = int(target_timeframe / current_timeframe)

        super().__init__(filters=filters,
                         kernel_size=(tf_window_size, features),
                         dilation_rate=1,
                         strides=tf_window_size,
                         data_format='channels_first',
                         activation=activation)


class TimeFrameBlock(tf.keras.Model):
    def __init__(self, features, target_timeframe, filters, activation):
        super().__init__()
        self.conv = TimeSeriesConv2D(features=features, current_timeframe=1, target_timeframe=target_timeframe,
                                     filters=filters, activation=activation)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.gru = tf.keras.layers.GRU(units=32,
                                       return_sequences=False,
                                       kernel_regularizer=tf.keras.regularizers.L1(0.005),
                                       kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.gru(x)

        return x


class CustomModel(tf.keras.Model):
    def __init__(self, features, output):
        super().__init__()
        self.vp = VolumeProfile(bins=50)
        self.vp_gru = tf.keras.layers.GRU(units=32, return_sequences=False,
                                          kernel_regularizer=tf.keras.regularizers.L1(0.005),
                                          kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.tf_5 = TimeFrameBlock(features=features, target_timeframe=5, filters=64, activation=None)
        self.tf_15 = TimeFrameBlock(features=features, target_timeframe=15, filters=64, activation=None)
        self.tf_30 = TimeFrameBlock(features=features, target_timeframe=30, filters=64, activation=None)
        self.tf_60 = TimeFrameBlock(features=features, target_timeframe=60, filters=64, activation=None)
        self.out = tf.keras.layers.Dense(output, activation='linear')

    def call(self, inputs, training=None, mask=None):
        vp = self.vp(inputs)
        vp_gru = self.vp_gru(vp)
        tf_5 = self.tf_5(inputs)
        tf_15 = self.tf_15(inputs)
        tf_30 = self.tf_30(inputs)
        tf_60 = self.tf_60(inputs)
        out = self.dense_1(tf.concat([vp_gru, tf_5, tf_15, tf_30, tf_60], axis=-1))

        return out


class TimeFrameConv2d(torch.nn.Conv2d):
    def __init__(self, features, current_timeframe, target_timeframe, in_channels, out_channels):
        tf_window_size = int(target_timeframe / current_timeframe)
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=(tf_window_size, features),
                         padding=0,
                         dilation=1,
                         stride=tf_window_size,
                         groups=1)
