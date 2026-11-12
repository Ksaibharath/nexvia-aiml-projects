"""Stage-2 Deep Ranking architectures: DCN-V2 (Variant B) and Wide & Deep (Variant A)."""
import tensorflow as tf

try:
    import tensorflow_recommenders as tfrs
    HAS_TFRS = True
except ImportError:
    tfrs = None
    HAS_TFRS = False

L = tf.keras.layers


class DCNCrossLayer(L.Layer):
    """Explicit DCN-V2 full-matrix feature cross layer: x_{l+1} = x_0 * (W_l x_l + b_l) + x_l."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj = None

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.proj = L.Dense(dim, use_bias=True, name="cross_dense")
        super().build(input_shape)

    def call(self, x0, x):
        return x0 * self.proj(x) + x


def _make_cross_layer(idx: int):
    if HAS_TFRS:
        return tfrs.layers.dcn.Cross(name=f"cross_{idx}")
    return DCNCrossLayer(name=f"cross_{idx}")


class DCNRanker(tf.keras.Model):
    def __init__(self, n_users: int, n_items: int, emb: int = 32, n_cross: int = 3, deep=(256, 128, 64)):
        super().__init__()
        self.ue = L.Embedding(n_users + 1, emb, name="user_emb")
        self.ie = L.Embedding(n_items + 1, emb, name="item_emb")
        self.id_drop = L.Dropout(0.25, name="id_drop")
        self.norm = L.BatchNormalization(name="dense_bn")
        self.cross = [_make_cross_layer(i) for i in range(n_cross)]
        self.deep = tf.keras.Sequential(
            [L.Dense(h, activation="relu", name=f"deep_{i}") for i, h in enumerate(deep)],
            name="deep_mlp",
        )
        self.out = L.Dense(1, name="logit_out")

    def call(self, f, training: bool = False):
        ids = self.id_drop(tf.concat([self.ue(f["user_idx"]), self.ie(f["item_idx"])], axis=-1), training=training)
        x0 = tf.concat([ids, self.norm(f["dense"], training=training)], axis=-1)
        x = x0
        for c in self.cross:
            x = c(x0, x)
        return self.out(tf.concat([x, self.deep(x0)], axis=-1))


class WideDeepRanker(tf.keras.Model):
    def __init__(self, n_users: int, n_items: int, emb: int = 32, deep=(256, 128, 64)):
        super().__init__()
        self.ue = L.Embedding(n_users + 1, emb, name="user_emb")
        self.ie = L.Embedding(n_items + 1, emb, name="item_emb")
        self.id_drop = L.Dropout(0.25, name="id_drop")
        self.norm = L.BatchNormalization(name="dense_bn")
        self.wide = L.Dense(1, name="wide_linear")
        self.deep = tf.keras.Sequential(
            [L.Dense(h, activation="relu", name=f"deep_{i}") for i, h in enumerate(deep)]
            + [L.Dense(1, name="deep_out")],
            name="deep_mlp",
        )

    def call(self, f, training: bool = False):
        d = self.norm(f["dense"], training=training)
        ids = self.id_drop(tf.concat([self.ue(f["user_idx"]), self.ie(f["item_idx"])], axis=-1), training=training)
        x = tf.concat([ids, d], axis=-1)
        return self.wide(d) + self.deep(x)
