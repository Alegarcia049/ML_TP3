import numpy as np

class MLPClassifier:
    """
    Multi-layer perceptron classifier implemented with NumPy.

    The network supports an arbitrary number of hidden layers, ReLU activations
    in hidden layers, softmax output, cross-entropy loss, and standard gradient
    descent optimization.
    """
    def __init__(
        self,
        layer_sizes,
        use_batch_norm=False,
        bn_momentum=0.9,
        bn_epsilon=1e-5
        ):
        """
        Initialize the MLP model.

        Parameters
        ----------
        layer_sizes : list[int]
            Network architecture. Example: [784, 128, 64, 47].
        use_batch_norm : bool, default=False
            Whether to apply batch normalization to hidden layers.
        bn_momentum : float
            Momentum used for running mean and variance.
        bn_epsilon : float
            Numerical stability constant for batch normalization.
        """
        self.layer_sizes = layer_sizes

        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self.weights = []
        self.biases = []
        self.cache = {}

        self.bn_gamma = []
        self.bn_beta = []
        self.bn_running_mean = []
        self.bn_running_var = []

        self._initialize_parameters()

        if self.use_batch_norm:
            self._initialize_batch_norm_parameters()

    def _initialize_parameters(self):
        """
        Initialize weights and biases.

        He initialization is used because hidden layers use ReLU activations.
        """
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            W = np.random.normal(
                loc=0.0,
                scale=np.sqrt(2 / fan_in),
                size=(fan_in, fan_out)
            )
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

    def _initialize_batch_norm_parameters(self):
        """
        Initialize batch normalization parameters for hidden layers only.
        """
        for hidden_size in self.layer_sizes[1:-1]:
            gamma = np.ones((1, hidden_size))
            beta = np.zeros((1, hidden_size))

            running_mean = np.zeros((1, hidden_size))
            running_var = np.ones((1, hidden_size))

            self.bn_gamma.append(gamma)
            self.bn_beta.append(beta)
            self.bn_running_mean.append(running_mean)
            self.bn_running_var.append(running_var)

    def _relu(self, Z):
        """
        Apply ReLU activation.
        """
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """
        Compute ReLU derivative.
        """
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        """
        Apply numerically stable softmax activation.
        """
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_scores = np.exp(Z_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        """
        Run the forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with shape (n_samples, n_features).
        training : bool, default=True
            Whether the model is in training mode.

        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        self.cache = {"A0": X}
        A = X

        num_layers = len(self.weights)

        for i in range(num_layers - 1):
            layer_num = i + 1

            Z = A @ self.weights[i] + self.biases[i]
            self.cache[f"Z{layer_num}"] = Z

            if self.use_batch_norm:
                Z_for_activation = self._batch_norm_forward(
                    Z,
                    layer_idx=i,
                    training=training
                )
                self.cache[f"Z_tilde{layer_num}"] = Z_for_activation
            else:
                Z_for_activation = Z

            A = self._relu(Z_for_activation)
            self.cache[f"A{layer_num}"] = A

        Z_out = A @ self.weights[-1] + self.biases[-1]
        A_out = self._softmax(Z_out)

        self.cache[f"Z{num_layers}"] = Z_out
        self.cache[f"A{num_layers}"] = A_out

        return A_out
    
    def _batch_norm_forward(self, Z, layer_idx, training=True):
        """
        Apply batch normalization to a hidden layer.

        Parameters
        ----------
        Z : np.ndarray
            Linear output before activation.
        layer_idx : int
            Hidden layer index.
        training : bool, default=True
            Whether the model is in training mode.

        Returns
        -------
        np.ndarray
            Batch-normalized output.
        """
        gamma = self.bn_gamma[layer_idx]
        beta = self.bn_beta[layer_idx]

        if training:
            batch_mean = np.mean(Z, axis=0, keepdims=True)
            batch_var = np.var(Z, axis=0, keepdims=True)

            Z_centered = Z - batch_mean
            std_inv = 1.0 / np.sqrt(batch_var + self.bn_epsilon)
            Z_hat = Z_centered * std_inv

            self.bn_running_mean[layer_idx] = (
                self.bn_momentum * self.bn_running_mean[layer_idx]
                + (1 - self.bn_momentum) * batch_mean
            )

            self.bn_running_var[layer_idx] = (
                self.bn_momentum * self.bn_running_var[layer_idx]
                + (1 - self.bn_momentum) * batch_var
            )

            self.cache[f"BN{layer_idx + 1}"] = {
                "Z_hat": Z_hat,
                "Z_centered": Z_centered,
                "std_inv": std_inv,
                "gamma": gamma
            }

        else:
            Z_centered = Z - self.bn_running_mean[layer_idx]
            std_inv = 1.0 / np.sqrt(self.bn_running_var[layer_idx] + self.bn_epsilon)
            Z_hat = Z_centered * std_inv

        return gamma * Z_hat + beta
    
    def _batch_norm_backward(self, dZ_norm, layer_idx):
        """
        Backpropagate through batch normalization.

        Parameters
        ----------
        dZ_norm : np.ndarray
            Gradient with respect to the normalized layer output.
        layer_idx : int
            Hidden layer index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Gradients with respect to Z, gamma and beta.
        """
        cache = self.cache[f"BN{layer_idx + 1}"]

        Z_hat = cache["Z_hat"]
        Z_centered = cache["Z_centered"]
        std_inv = cache["std_inv"]
        gamma = cache["gamma"]

        m = dZ_norm.shape[0]

        dgamma = np.sum(dZ_norm * Z_hat, axis=0, keepdims=True)
        dbeta = np.sum(dZ_norm, axis=0, keepdims=True)

        dZ_hat = dZ_norm * gamma

        dvar = np.sum(
            dZ_hat * Z_centered * (-0.5) * (std_inv ** 3),
            axis=0,
            keepdims=True
        )

        dmean = (
            np.sum(-dZ_hat * std_inv, axis=0, keepdims=True)
            + dvar * np.mean(-2.0 * Z_centered, axis=0, keepdims=True)
        )

        dZ = (
            dZ_hat * std_inv
            + dvar * 2.0 * Z_centered / m
            + dmean / m
        )

        return dZ, dgamma, dbeta

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Compute multiclass cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            One-hot encoded labels.
        y_pred : np.ndarray
            Predicted class probabilities.

        Returns
        -------
        float
            Mean cross-entropy loss.
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _update_parameters(
        self,
        grads_W,
        grads_b,
        grads_gamma=None,
        grads_beta=None,
        learning_rate=None
        ):
        """
        Update parameters using standard gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_W[i]
            self.biases[i] -= learning_rate * grads_b[i]

        if self.use_batch_norm:
            for i in range(len(self.bn_gamma)):
                self.bn_gamma[i] -= learning_rate * grads_gamma[i]
                self.bn_beta[i] -= learning_rate * grads_beta[i]
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        return self.forward(X, training=False)

    def predict(self, X):
        """
        Predict class labels.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    
    def accuracy_score(self, y_true, y_pred):
        """
        Compute classification accuracy.
        """
        return np.mean(y_true == y_pred)

    def confusion_matrix(self, y_true, y_pred, num_classes):
        """
        Compute confusion matrix without external ML libraries.
        """
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            matrix[true_label, pred_label] += 1

        return matrix

    def f1_macro_score(self, y_true, y_pred, num_classes):
        """
        Compute macro F1-score from scratch.
        """
        cm = self.confusion_matrix(y_true, y_pred, num_classes)

        f1_scores = []

        for cls in range(num_classes):
            tp = cm[cls, cls]
            fp = np.sum(cm[:, cls]) - tp
            fn = np.sum(cm[cls, :]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            f1_scores.append(f1)

        return np.mean(f1_scores)
    
    def evaluate(self, X, y_one_hot):
        """
        Evaluate model performance.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y_one_hot : np.ndarray
            One-hot encoded labels.

        Returns
        -------
        dict
            Dictionary with loss, accuracy, confusion matrix and macro F1-score.
        """
        y_true = np.argmax(y_one_hot, axis=1)

        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)

        num_classes = y_one_hot.shape[1]

        return {
            "cross_entropy": self.cross_entropy_loss(y_one_hot, y_proba),
            "accuracy": self.accuracy_score(y_true, y_pred),
            "confusion_matrix": self.confusion_matrix(y_true, y_pred, num_classes),
            "f1_macro": self.f1_macro_score(y_true, y_pred, num_classes)
        }
    
    def _create_batches(self, X, y, batch_size, shuffle=True):
        """
        Create mini-batches from the training data.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield X[batch_indices], y[batch_indices]
    
    def _get_learning_rate(
        self,
        epoch,
        initial_lr,
        scheduler=None,
        final_lr=1e-4,
        decay_rate=0.95,
        max_epochs=100
    ):
        """
        Compute the learning rate for the current epoch.
        """
        if scheduler is None:
            return initial_lr

        if scheduler == "linear":
            progress = epoch / max(max_epochs - 1, 1)
            lr = initial_lr - progress * (initial_lr - final_lr)
            return max(lr, final_lr)

        if scheduler == "exponential":
            lr = initial_lr * (decay_rate ** epoch)
            return max(lr, final_lr)

        raise ValueError("scheduler must be None, 'linear', or 'exponential'")
    
    def backward(self, y_true, l2_lambda=0.0):
        """
        Run backpropagation and compute gradients.
        """
        m = y_true.shape[0]
        num_layers = len(self.weights)

        grads_W = [None] * num_layers
        grads_b = [None] * num_layers

        grads_gamma = [None] * (num_layers - 1)
        grads_beta = [None] * (num_layers - 1)

        A_out = self.cache[f"A{num_layers}"]

        # Softmax + cross-entropy simplified gradient.
        dZ = (A_out - y_true) / m

        for i in reversed(range(num_layers)):
            A_prev = self.cache[f"A{i}"]

            grads_W[i] = A_prev.T @ dZ

            # L2 is applied only to weights, not biases
            if l2_lambda > 0:
                grads_W[i] += (l2_lambda / m) * self.weights[i]

            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            if i > 0:
                dA_prev = dZ @ self.weights[i].T

                if self.use_batch_norm:
                    Z_tilde = self.cache[f"Z_tilde{i}"]
                    dZ_hidden = dA_prev * self._relu_derivative(Z_tilde)

                    dZ, dgamma, dbeta = self._batch_norm_backward(
                        dZ_hidden,
                        layer_idx=i - 1
                    )

                    grads_gamma[i - 1] = dgamma
                    grads_beta[i - 1] = dbeta

                else:
                    Z_prev = self.cache[f"Z{i}"]
                    dZ = dA_prev * self._relu_derivative(Z_prev)

        return grads_W, grads_b, grads_gamma, grads_beta

    def _initialize_adam_state(self):
        """
        Initialize Adam first and second moment estimates.
        """
        self.adam_m_W = [np.zeros_like(W) for W in self.weights]
        self.adam_v_W = [np.zeros_like(W) for W in self.weights]

        self.adam_m_b = [np.zeros_like(b) for b in self.biases]
        self.adam_v_b = [np.zeros_like(b) for b in self.biases]

        if self.use_batch_norm:
            self.adam_m_gamma = [np.zeros_like(g) for g in self.bn_gamma]
            self.adam_v_gamma = [np.zeros_like(g) for g in self.bn_gamma]

            self.adam_m_beta = [np.zeros_like(b) for b in self.bn_beta]
            self.adam_v_beta = [np.zeros_like(b) for b in self.bn_beta]

        self.adam_t = 0

    def _adam_update(
        self,
        grads_W,
        grads_b,
        grads_gamma,
        grads_beta,
        learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
        ):
        """
        Update parameters using the Adam optimizer.
        """
        self.adam_t += 1

        for i in range(len(self.weights)):
            self.adam_m_W[i] = beta1 * self.adam_m_W[i] + (1 - beta1) * grads_W[i]
            self.adam_v_W[i] = beta2 * self.adam_v_W[i] + (1 - beta2) * (grads_W[i] ** 2)

            self.adam_m_b[i] = beta1 * self.adam_m_b[i] + (1 - beta1) * grads_b[i]
            self.adam_v_b[i] = beta2 * self.adam_v_b[i] + (1 - beta2) * (grads_b[i] ** 2)

            m_W_hat = self.adam_m_W[i] / (1 - beta1 ** self.adam_t)
            v_W_hat = self.adam_v_W[i] / (1 - beta2 ** self.adam_t)

            m_b_hat = self.adam_m_b[i] / (1 - beta1 ** self.adam_t)
            v_b_hat = self.adam_v_b[i] / (1 - beta2 ** self.adam_t)

            self.weights[i] -= learning_rate * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        if self.use_batch_norm:
            for i in range(len(self.bn_gamma)):
                self.adam_m_gamma[i] = beta1 * self.adam_m_gamma[i] + (1 - beta1) * grads_gamma[i]
                self.adam_v_gamma[i] = beta2 * self.adam_v_gamma[i] + (1 - beta2) * (grads_gamma[i] ** 2)

                self.adam_m_beta[i] = beta1 * self.adam_m_beta[i] + (1 - beta1) * grads_beta[i]
                self.adam_v_beta[i] = beta2 * self.adam_v_beta[i] + (1 - beta2) * (grads_beta[i] ** 2)

                m_gamma_hat = self.adam_m_gamma[i] / (1 - beta1 ** self.adam_t)
                v_gamma_hat = self.adam_v_gamma[i] / (1 - beta2 ** self.adam_t)

                m_beta_hat = self.adam_m_beta[i] / (1 - beta1 ** self.adam_t)
                v_beta_hat = self.adam_v_beta[i] / (1 - beta2 ** self.adam_t)

                self.bn_gamma[i] -= learning_rate * m_gamma_hat / (np.sqrt(v_gamma_hat) + epsilon)
                self.bn_beta[i] -= learning_rate * m_beta_hat / (np.sqrt(v_beta_hat) + epsilon)

    def _copy_parameters(self):
        """
        Copy current model parameters.
        """
        parameters = {
            "weights": [W.copy() for W in self.weights],
            "biases": [b.copy() for b in self.biases]
        }

        if self.use_batch_norm:
            parameters["bn_gamma"] = [g.copy() for g in self.bn_gamma]
            parameters["bn_beta"] = [b.copy() for b in self.bn_beta]
            parameters["bn_running_mean"] = [m.copy() for m in self.bn_running_mean]
            parameters["bn_running_var"] = [v.copy() for v in self.bn_running_var]

        return parameters

    def _restore_parameters(self, parameters):
        """
        Restore model parameters from a saved copy.
        """
        self.weights = [W.copy() for W in parameters["weights"]]
        self.biases = [b.copy() for b in parameters["biases"]]

        if self.use_batch_norm:
            self.bn_gamma = [g.copy() for g in parameters["bn_gamma"]]
            self.bn_beta = [b.copy() for b in parameters["bn_beta"]]
            self.bn_running_mean = [m.copy() for m in parameters["bn_running_mean"]]
            self.bn_running_var = [v.copy() for v in parameters["bn_running_var"]]

    def _clip_gradients_by_global_norm(
        self,
        grads_W,
        grads_b,
        grads_gamma=None,
        grads_beta=None,
        max_norm=None
        ):
        """
        Clip gradients using global norm.
        """
        if max_norm is None:
            return grads_W, grads_b, grads_gamma, grads_beta

        total_norm_sq = 0.0

        all_grads = grads_W + grads_b

        if grads_gamma is not None:
            all_grads += [g for g in grads_gamma if g is not None]

        if grads_beta is not None:
            all_grads += [g for g in grads_beta if g is not None]

        for grad in all_grads:
            if grad is not None:
                total_norm_sq += np.sum(grad ** 2)

        total_norm = np.sqrt(total_norm_sq)

        if total_norm <= max_norm:
            return grads_W, grads_b, grads_gamma, grads_beta

        scale = max_norm / (total_norm + 1e-12)

        grads_W = [g * scale for g in grads_W]
        grads_b = [g * scale for g in grads_b]

        if grads_gamma is not None:
            grads_gamma = [
                g * scale if g is not None else None
                for g in grads_gamma
            ]

        if grads_beta is not None:
            grads_beta = [
                g * scale if g is not None else None
                for g in grads_beta
            ]

        return grads_W, grads_b, grads_gamma, grads_beta

    def _smooth_labels(self, y, smoothing):
        """
        Apply label smoothing to one-hot encoded labels.
        """
        if smoothing <= 0:
            return y

        num_classes = y.shape[1]

        return y * (1 - smoothing) + smoothing / num_classes
    
    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=128,
        optimizer="gd",
        scheduler=None,
        initial_lr=0.01,
        final_lr=1e-4,
        decay_rate=0.95,
        l2_lambda=0.0,
        early_stopping=False,
        patience=10,
        label_smoothing=0,
        gradient_clip_norm=None,
        verbose=True
        ):
        """
        Train the model using mini-batch optimization and optional improvements.

        Supported improvements:
        - Mini-batch SGD
        - Linear learning rate scheduling
        - Exponential learning rate scheduling
        - Adam optimizer
        - L2 regularization
        - Early stopping
        - Label smoothing
        - Gradient clipping
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

        if optimizer == "adam":
            self._initialize_adam_state()

        best_val_loss = np.inf
        best_parameters = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            current_lr = self._get_learning_rate(
                epoch=epoch,
                initial_lr=initial_lr,
                scheduler=scheduler,
                final_lr=final_lr,
                decay_rate=decay_rate,
                max_epochs=epochs
            )

            for X_batch, y_batch in self._create_batches(
                X_train,
                y_train,
                batch_size=batch_size,
                shuffle=True
            ):
                # Apply label smoothing only during training.
                y_batch_train = self._smooth_labels(
                    y_batch,
                    smoothing=label_smoothing
                )

                # Forward pass in training mode.
                self.forward(X_batch, training=True)

                # Backward pass.
                grads_W, grads_b, grads_gamma, grads_beta = self.backward(
                    y_batch_train,
                    l2_lambda=l2_lambda
                )

                # Clip gradients before updating parameters.
                grads_W, grads_b, grads_gamma, grads_beta = self._clip_gradients_by_global_norm(
                    grads_W,
                    grads_b,
                    grads_gamma,
                    grads_beta,
                    max_norm=gradient_clip_norm
                )

                if optimizer == "gd":
                    self._update_parameters(
                        grads_W,
                        grads_b,
                        grads_gamma,
                        grads_beta,
                        learning_rate=current_lr
                    )

                elif optimizer == "adam":
                    self._adam_update(
                        grads_W,
                        grads_b,
                        grads_gamma,
                        grads_beta,
                        learning_rate=current_lr
                    )

                else:
                    raise ValueError("optimizer must be 'gd' or 'adam'")

            train_pred = self.forward(X_train, training=False)
            val_pred = self.forward(X_val, training=False)

            train_loss = self.cross_entropy_loss(y_train, train_pred)
            val_loss = self.cross_entropy_loss(y_val, val_pred)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_parameters = self._copy_parameters()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch:03d}")

                    self._restore_parameters(best_parameters)
                    break

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"LR: {current_lr:.6f} | "
                    f"Train CE: {train_loss:.4f} | "
                    f"Val CE: {val_loss:.4f}"
                )

        return history