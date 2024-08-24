from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from utils import GradientBasedPolicyDataset  # utilsモジュールが適用可能か確認してください


@dataclass
class IPSBasedGradientPolicyLearner:
    """勾配ベースのアプローチに基づく、メール配信経由の視聴時間を最大化するオフ方策学習."""
    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        if self.activation == "tanh":
            activation_layer = layers.Activation("tanh")
        elif self.activation == "relu":
            activation_layer = layers.Activation("relu")
        elif self.activation == "elu":
            activation_layer = layers.Activation("elu")

        # Keras Sequentialモデルの作成
        self.model = models.Sequential()
        for h in self.hidden_layer_size:
            self.model.add(layers.Dense(h, input_dim=input_size))
            self.model.add(activation_layer)
            input_size = h
        self.model.add(layers.Dense(self.num_actions))
        self.model.add(layers.Softmax())

        # Optimizerの設定
        if self.solver == "adagrad":
            self.optimizer = optimizers.Adagrad(learning_rate=self.learning_rate_init)
        elif self.solver == "adam":
            self.optimizer = optimizers.Adam(learning_rate=self.learning_rate_init)
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        self.naive_value = []
        self.cate_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        pscore, pi_0 = dataset["pscore"], dataset["pi_0"]

        # データセットの準備
        training_data_loader = self._create_train_data_for_opl(
            x, a, r, pscore, pi_0, pi_0
        )

        # ポリシートレーニングの開始
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            for batch in training_data_loader:
                x_batch, a_batch, r_batch, p_batch, _, _ = batch
                with tf.GradientTape() as tape:
                    pi = self.model(x_batch, training=True)
                    loss = -tf.reduce_mean(self._estimate_policy_gradient(a_batch, r_batch, p_batch, pi))
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            pi = self.predict(dataset_test)
            self.naive_value.append(np.mean(np.sum(pi * q_x_a_1, axis=1)))
            self.cate_value.append(np.mean(np.sum(pi * q_x_a_1 + (1.0 - pi) * q_x_a_0, axis=1)))

    def _create_train_data_for_opl(
        self, x: np.ndarray, a: np.ndarray, r: np.ndarray, pscore: np.ndarray, q_hat: np.ndarray, pi_0: np.ndarray
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((x, a, r, pscore, q_hat, pi_0))
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _estimate_policy_gradient(
        self, a: tf.Tensor, r: tf.Tensor, pscore: tf.Tensor, pi: tf.Tensor
    ) -> tf.Tensor:
        current_pi = pi
        log_prob = tf.math.log(pi + self.log_eps)
        idx = tf.range(a.shape[0])

        iw = tf.gather(current_pi, idx, axis=1, batch_dims=1) / pscore
        estimated_policy_grad_arr = iw * r * tf.gather(log_prob, idx, axis=1, batch_dims=1)

        return estimated_policy_grad_arr

    def predict(self, dataset_test: dict) -> np.ndarray:
        x = dataset_test["x"]
        pi = self.model(x, training=False)
        return pi.numpy()


@dataclass
class CateBasedGradientPolicyLearner(IPSBasedGradientPolicyLearner):
    """勾配ベースのアプローチに基づく、プラットフォーム全体の視聴時間を最大化するオフ方策学習."""
    
    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        a_mat, r_mat, pscore_mat = dataset["a_mat"], dataset["r_mat"], dataset["pscore_mat"]

        training_data_loader = self._create_train_data_for_opl(
            x, a, r, pscore_mat, a_mat, r_mat
        )

        # ポリシートレーニングの開始
        q_x_a_1 = dataset_test["q_x_a_1"]
        q_x_a_0 = dataset_test["q_x_a_0"]
        for _ in range(self.max_iter):
            for batch in training_data_loader:
                x_batch, a_batch, r_batch, pscore_mat_batch, a_mat_batch, r_mat_batch = batch
                with tf.GradientTape() as tape:
                    pi = self.model(x_batch, training=True)
                    loss = -tf.reduce_mean(
                        self._estimate_policy_gradient(a_batch, a_mat_batch, r_batch, r_mat_batch, pscore_mat_batch, pi)
                    )
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            pi = self.predict(dataset_test)
            self.naive_value.append(np.mean(np.sum(pi * q_x_a_1, axis=1)))
            self.cate_value.append(np.mean(np.sum(pi * q_x_a_1 + (1.0 - pi) * q_x_a_0, axis=1)))

    def _estimate_policy_gradient(
        self, a: tf.Tensor, a_mat: tf.Tensor, r: tf.Tensor, r_mat: tf.Tensor, pscore_mat: tf.Tensor, pi: tf.Tensor
    ) -> tf.Tensor:
        current_pi = pi
        log_prob1 = tf.math.log(pi + self.log_eps)
        log_prob2 = tf.math.log(1.0 - pi + self.log_eps)
        idx = tf.range(a.shape[0])

        estimated_policy_grad_arr = (
            tf.gather(current_pi, idx, axis=1, batch_dims=1) * r / tf.gather(pscore_mat, idx, axis=1, batch_dims=1)
        ) * tf.gather(log_prob1, idx, axis=1, batch_dims=1)
        estimated_policy_grad_arr += (
            (1 - a_mat) * (1.0 - current_pi) * (r_mat / pscore_mat) * log_prob2
        ).sum(1)

        return estimated_policy_grad_arr