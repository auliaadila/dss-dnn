import tensorflow as tf
import numpy as np

class DeltaPESQ(tf.keras.layers.Layer):
    def __init__(self, fs=16000, **kw):
        super().__init__(trainable=False, **kw); self.fs=fs
    def call(self, inputs):
        ref, deg = inputs                         # (B,T,1) clean & wm
        # def _pesq_pair(r,d):
        #     from pesq import pesq
        #     return np.float32(
        #         pesq(self.fs, r, r, 'wb') - pesq(self.fs, r, d, 'wb'))
        def _pesq_pair(ref, deg):
            from pesq import pesq
            # Make sure they are 1D
            ref = np.asarray(ref).flatten()
            deg = np.asarray(deg).flatten()
            return np.float32(pesq(self.fs, ref, ref, 'wb') - pesq(self.fs, ref, deg, 'wb'))
        delta = tf.numpy_function(_pesq_pair,[tf.squeeze(ref),tf.squeeze(deg)],
                                  tf.float32)
        self.add_metric(delta, name='delta_pesq', aggregation='mean')
        # also expose as a loss term (value only, no grad)
        self.add_loss(delta)
        return deg                                # pass-through audio
    def get_config(self):
        config = super().get_config()
        config.update({
            "fs": self.fs
        })
        return config

class PESQLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, log_prefix="run"):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.log_prefix = log_prefix
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        losses = self.model.get_layer("pesq_layer").losses
        if losses:
            delta_pesq = float(tf.reduce_mean(losses[0]))
            with self.writer.as_default():
                tf.summary.scalar(f"{self.log_prefix}/delta_pesq", delta_pesq, step=epoch)
        self.epoch += 1

class DeltaPESQMetric(tf.keras.metrics.Metric):
    def __init__(self, fs=16000, name='delta_pesq', **kwargs):
        super().__init__(name=name, **kwargs)
        self.fs = fs
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        def pesq_np(ref, deg):
            import numpy as np
            from pesq import pesq
            ref = np.asarray(ref).flatten()
            deg = np.asarray(deg).flatten()
            return np.float32(pesq(self.fs, ref, deg, 'wb'))

        delta = tf.numpy_function(
            lambda ref, deg: pesq_np(ref, deg),
            [tf.squeeze(y_true), tf.squeeze(y_pred)],
            tf.float32
        )
        self.total.assign_add(delta)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)