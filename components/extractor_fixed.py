import tensorflow as tf


class DSSExtractorFixed(tf.keras.Model):
    """Enhanced 1-D CNN extractor for LP-DSS payload.

    Key improvements:
    - High-pass residual preprocessing to match embedder
    - Increased capacity with doubled base_filters
    - Reduced aggressive pooling
    - Better architecture for 64-bit classification

    Parameters
    ----------
    payload_bits : int
        Length of the payload vector (e.g. 64 or 8).
    context_len  : int
        Number of time-samples per training window (frame_size * frames_per_payload).
    base_filters : int, optional
        # of filters in the first Conv layer (doubled from original).
    """

    def __init__(
        self,
        payload_bits: int,
        context_len: int,
        base_filters: int = 64,  # Doubled from 32
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bits = payload_bits  # P: 64
        self.len = context_len  # 2400
        self.base_filters = base_filters  # Store for get_config
        F = base_filters

        # Re-enable high-pass filter to extract watermark signal
        def _pre(x): 
            return x - 0.95 * tf.pad(x, [[0,0],[1,0],[0,0]])[:,:-1,:]
        self.highpass = tf.keras.layers.Lambda(_pre, name='hp')

        self.backbone = tf.keras.Sequential(
            [
                # Progressive dilation stack: (1,2,4,8,16,32) for dense pattern recognition
                tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=4),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=8),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=16),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=32),
                tf.keras.layers.LayerNormalization(),
                # RED 2: Use stride=64 for pooling to match the chip pattern
                tf.keras.layers.AveragePooling1D(
                    64, strides=64
                ),  # Pool with stride matching the chip period
                tf.keras.layers.Conv1D(F * 2, 3, padding="same", activation="relu"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Conv1D(F * 2, 3, padding="same", activation="relu"),
                tf.keras.layers.LayerNormalization(),
                # Third block - keep moderate pooling
                tf.keras.layers.AveragePooling1D(
                    2
                ),  # L/128 (after first pooling of 64)
                tf.keras.layers.Conv1D(F * 4, 3, padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(F * 4, 3, padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GlobalAveragePooling1D(),  # (B, F*4)
            ]
        )

        # Enhanced classifier for 64-bit output
        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(F * 8, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(payload_bits, activation="sigmoid"),
            ]
        )

    def call(self, x, training=False):
        """x : (B, context_len, 1) float32 in [-1,1]"""
        # Apply high-pass filter to extract watermark signal  
        x_hp = self.highpass(x)

        # Extract features
        h = self.backbone(x_hp, training=training)

        # Classify bits
        return self.classifier(h, training=training)
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update({
            'payload_bits': self.bits,
            'context_len': self.len,
            'base_filters': self.base_filters,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration, with defaults for missing parameters."""
        # Handle old models that don't have these parameters saved
        payload_bits = config.get('payload_bits', 64)  # Default to 64
        context_len = config.get('context_len', 2400)  # Default to 2400 (15*160)
        base_filters = config.get('base_filters', 64)  # Default to 64
        
        # Remove these from config to avoid passing them twice
        config = config.copy()
        config.pop('payload_bits', None)
        config.pop('context_len', None)
        config.pop('base_filters', None)
        
        return cls(
            payload_bits=payload_bits,
            context_len=context_len,
            base_filters=base_filters,
            **config
        )
