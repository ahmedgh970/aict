from functools import partial

import tensorflow as tf
from tensorflow.keras import layers as L


class PatchMerging(L.Layer):
    """Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
    """

    def __init__(
            self,
            dim,
            out_dim=None,
            norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer()
        self.reduction = L.Dense(self.out_dim, use_bias=False)

    def call(self, x):
        """
        x: B, H, W, C
        """
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # [B, H/2, W/2, 4*C]
        x = tf.reshape(x, (-1, H//2*W//2, 4 * C))  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)
        x = tf.reshape(x, (-1, H//2, W//2, self.out_dim))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "out_dim": self.out_dim,
                "norm": self.norm,
            }
        )
        return config
