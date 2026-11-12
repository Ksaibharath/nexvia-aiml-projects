"""Two-Tower neural retrieval architecture (compatible with TFRS 0.7.x and Keras 3 / TF 2.16+)."""
import numpy as np
import tensorflow as tf

try:
    import tensorflow_recommenders as tfrs
    HAS_TFRS = True
except ImportError:
    tfrs = None
    HAS_TFRS = False

L = tf.keras.layers


class UserTower(tf.keras.Model):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.user_emb = L.Embedding(n_users + 1, dim, name="user_emb")
        self.hist_emb = L.Embedding(n_items + 1, dim, name="hist_emb")
        self.mlp = tf.keras.Sequential(
            [
                L.Dense(128, activation="relu", name="u_dense_0"),
                L.Dense(dim, name="u_dense_1"),
            ],
            name="user_mlp",
        )

    def call(self, f):
        u = self.user_emb(f["user_idx"])                             # (B, dim)
        h = self.hist_emb(f["history"])                              # (B, HIST, dim)
        mask = tf.cast(f["history"] > 0, tf.float32)[..., None]      # ignore padding 0
        h_mean = tf.reduce_sum(h * mask, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        return tf.math.l2_normalize(self.mlp(tf.concat([u, h_mean], axis=-1)), axis=-1)


class ItemTower(tf.keras.Model):
    def __init__(self, n_items: int, genre_matrix: np.ndarray, dim: int = 64):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.emb = L.Embedding(n_items + 1, dim, name="item_emb")
        self.genres = tf.constant(genre_matrix, dtype=tf.float32)
        self.mlp = tf.keras.Sequential(
            [
                L.Dense(128, activation="relu", name="i_dense_0"),
                L.Dense(dim, name="i_dense_1"),
            ],
            name="item_mlp",
        )

    def call(self, item_idx):
        x = tf.concat([self.emb(item_idx), tf.gather(self.genres, item_idx)], axis=-1)
        return tf.math.l2_normalize(self.mlp(x), axis=-1)


if HAS_TFRS:
    class TwoTower(tfrs.Model):
        def __init__(self, user_tower: UserTower, item_tower: ItemTower, all_item_idx: np.ndarray,
                     item_probs: np.ndarray | None = None, temperature: float = 0.05):
            super().__init__()
            self.user_tower = user_tower
            self.item_tower = item_tower
            self.item_probs = tf.constant(item_probs, dtype=tf.float32) if item_probs is not None else None
            candidates = tf.data.Dataset.from_tensor_slices(all_item_idx).batch(2048).map(item_tower)
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(candidates=candidates, ks=[10, 50, 100]),
                temperature=temperature,
            )

        def compute_loss(self, features, training: bool = False):
            u = self.user_tower(features)
            i = self.item_tower(features["item_idx"])
            cand_prob = tf.gather(self.item_probs, features["item_idx"]) if self.item_probs is not None else None
            return self.task(u, i, candidate_sampling_probability=cand_prob, compute_metrics=not training)

else:
    class TwoTower(tf.keras.Model):
        """In-batch softmax Two-Tower retrieval with LogQ frequency correction & accidental-hit masking."""

        def __init__(self, user_tower: UserTower, item_tower: ItemTower, all_item_idx: np.ndarray,
                     item_probs: np.ndarray | None = None, temperature: float = 0.07):
            super().__init__()
            self.user_tower = user_tower
            self.item_tower = item_tower
            self.all_item_idx = tf.constant(all_item_idx, dtype=tf.int32)
            self.temperature = temperature
            self.log_q = (
                tf.constant(np.log(np.clip(item_probs, 1e-7, 1.0)), dtype=tf.float32)
                if item_probs is not None
                else None
            )
            self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.acc_top10 = tf.keras.metrics.Mean(name="inbatch_recall_at_10")

        @property
        def metrics(self):
            return [self.loss_tracker, self.acc_top10]

        def _batch_loss(self, data, training: bool):
            u = self.user_tower(data, training=training)              # (B, D)
            item_ids = data["item_idx"]                               # (B,)
            v = self.item_tower(item_ids, training=training)          # (B, D)

            logits = tf.matmul(u, v, transpose_b=True) / self.temperature  # (B, B)
            if self.log_q is not None:
                # Partial LogQ correction balances personalization vs popularity bias
                lq = tf.gather(self.log_q, item_ids)                  # (B,)
                logits = logits - 0.45 * lq[None, :]

            # Mask accidental duplicate positives off the diagonal
            same_item = tf.equal(item_ids[:, None], item_ids[None, :])
            diag = tf.eye(tf.shape(item_ids)[0], dtype=tf.bool)
            accidental = tf.logical_and(same_item, tf.logical_not(diag))
            logits = tf.where(accidental, tf.fill(tf.shape(logits), -1e9), logits)

            labels = tf.range(tf.shape(item_ids)[0])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            )

            k = tf.minimum(10, tf.shape(logits)[1])
            top_k = tf.math.top_k(logits, k=k).indices
            in_top = tf.reduce_any(tf.equal(top_k, labels[:, None]), axis=-1)
            return loss, tf.reduce_mean(tf.cast(in_top, tf.float32))

        def train_step(self, data):
            with tf.GradientTape() as tape:
                loss, r10 = self._batch_loss(data, training=True)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.loss_tracker.update_state(loss)
            self.acc_top10.update_state(r10)
            return {"total_loss": self.loss_tracker.result(), "inbatch_recall_at_10": self.acc_top10.result()}

        def test_step(self, data):
            loss, r10 = self._batch_loss(data, training=False)
            self.loss_tracker.update_state(loss)
            self.acc_top10.update_state(r10)
            return {"total_loss": self.loss_tracker.result(), "inbatch_recall_at_10": self.acc_top10.result()}
