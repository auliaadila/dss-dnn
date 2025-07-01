# extractor.py

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LeakyReLU,
)


class WatermarkExtractor(Model):
    def __init__(self, time_len=2400, bits_per_frame=64, use_global_pool=False):
        """
        Args:
            time_len (int or None): Input time dimension. Use None for variable length input.
            bits_per_frame (int): Output bit length.
            use_global_pool (bool): If True, use GlobalAveragePooling instead of Flatten.
        """
        super(WatermarkExtractor, self).__init__()
        self.time_len = time_len
        self.bits_per_frame = bits_per_frame
        self.use_global_pool = use_global_pool

        self.conv_layers = [
            Conv1D(32, 5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1D(32, 5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1D(64, 5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1D(64, 5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1D(128, 5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1D(128, 5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
        ]

        self.pool_or_flatten = (
            GlobalAveragePooling1D(name="global_pool") if use_global_pool else Flatten()
        )
        self.output_dense = Dense(
            bits_per_frame, activation="sigmoid", name="bits_pred"
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x, training=training)
        x = self.pool_or_flatten(x)
        return self.output_dense(x)


if __name__ == "__main__":
    # Example usage and summary
    model = WatermarkExtractor(time_len=2400, bits_per_frame=64, use_global_pool=False)

    # Build with specified input shape
    model.build(input_shape=(None, 2400, 1))
    model.summary()

    # Test dummy inference
    dummy = tf.zeros((128, 2400, 1))
    out = model(dummy)
    print("output shape:", out.shape)  # (128, 64)
