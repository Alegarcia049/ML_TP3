import numpy as np


class MLPClassifier:
    """
    Multi-layer perceptron classifier implemented with NumPy.

    The network supports an arbitrary number of hidden layers, ReLU activations
    in hidden layers, softmax output, cross-entropy loss, and standard gradient
    descent optimization.
    """
    ### MO
    def __init__(self, layer_sizes, learning_rate):
        """
        Initialize the MLP model.

        Parameters
        ----------
        layer_sizes : list[int]
            Network architecture. Example: [784, 128, 64, 47].
        learning_rate : float
            Gradient descent learning rate.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []
        self.cache = {}

        self._initialize_parameters()

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
    
    def forward(self, X):
        """
        Run the forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probabilities with shape (n_samples, n_classes).
        """
        self.cache = {"A0": X}
        A = X

        num_layers = len(self.weights)

        for i in range(num_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self._relu(Z)

            self.cache[f"Z{i + 1}"] = Z
            self.cache[f"A{i + 1}"] = A

        Z_out = A @ self.weights[-1] + self.biases[-1]
        A_out = self._softmax(Z_out)

        self.cache[f"Z{num_layers}"] = Z_out
        self.cache[f"A{num_layers}"] = A_out

        return A_out
    
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
    
    def update_parameters(self, grads_W, grads_b, learning_rate=None):
        """
        Update parameters using standard gradient descent.
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_W[i]
            self.biases[i] -= learning_rate * grads_b[i]
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        return self.forward(X)

    def predict(self, X):
        """
        Predict class labels.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """
        Train the model using full-batch gradient descent.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            One-hot encoded training labels.
        X_val : np.ndarray, optional
            Validation features.
        y_val : np.ndarray, optional
            One-hot encoded validation labels.
        epochs : int
            Number of training epochs.

        Returns
        -------
        dict
            Training history with loss values.
        """
        history = {
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            train_loss = self.cross_entropy_loss(y_train, y_pred)

            grads_W, grads_b = self.backward(y_train)
            self.update_parameters(grads_W, grads_b)

            history["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(y_val, val_pred)
                history["val_loss"].append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f}")

        return history
    
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
    
    ###M1
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
            progress = epoch / max_epochs
            lr = initial_lr - progress * (initial_lr - final_lr)
            return max(lr, final_lr)

        if scheduler == "exponential":
            lr = initial_lr * (decay_rate ** epoch)
            return max(lr, final_lr)

        raise ValueError("scheduler must be None, 'linear', or 'exponential'")
    
    def _l2_penalty(self, l2_lambda, n_samples):
        """
        Compute L2 regularization penalty.
        """
        if l2_lambda == 0:
            return 0.0

        penalty = 0.0

        for W in self.weights:
            penalty += np.sum(W ** 2)

        return (l2_lambda / (2 * n_samples)) * penalty
    
    def backward(self, y_true, l2_lambda=0.0):
        """
        Run backpropagation and compute gradients.
        """
        m = y_true.shape[0]
        num_layers = len(self.weights)

        grads_W = [None] * num_layers
        grads_b = [None] * num_layers

        A_out = self.cache[f"A{num_layers}"]

        # Softmax + cross-entropy simplified gradient
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
                Z_prev = self.cache[f"Z{i}"]
                dZ = dA_prev * self._relu_derivative(Z_prev)

        return grads_W, grads_b

    def _initialize_adam_state(self):
        """
        Initialize Adam first and second moment estimates.
        """
        self.adam_m_W = [np.zeros_like(W) for W in self.weights]
        self.adam_v_W = [np.zeros_like(W) for W in self.weights]

        self.adam_m_b = [np.zeros_like(b) for b in self.biases]
        self.adam_v_b = [np.zeros_like(b) for b in self.biases]

        self.adam_t = 0

    def _adam_update(
        self,
        grads_W,
        grads_b,
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

    def _copy_parameters(self):
        """
        Copy current model parameters.
        """
        return {
            "weights": [W.copy() for W in self.weights],
            "biases": [b.copy() for b in self.biases]
        }


    def _restore_parameters(self, parameters):
        """
        Restore model parameters from a saved copy.
        """
        self.weights = [W.copy() for W in parameters["weights"]]
        self.biases = [b.copy() for b in parameters["biases"]]

    def fit_advanced(
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
        """
        history = {
            "train_ce": [],
            "val_ce": [],
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
                y_pred_batch = self.forward(X_batch)
                grads_W, grads_b = self.backward(y_batch, l2_lambda=l2_lambda)

                if optimizer == "gd":
                    self.update_parameters(grads_W, grads_b, learning_rate=current_lr)

                elif optimizer == "adam":
                    self._adam_update(
                        grads_W,
                        grads_b,
                        learning_rate=current_lr
                    )

                else:
                    raise ValueError("optimizer must be 'gd' or 'adam'")

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_ce = self.cross_entropy_loss(y_train, train_pred)
            val_ce = self.cross_entropy_loss(y_val, val_pred)

            history["train_ce"].append(train_ce)
            history["val_ce"].append(val_ce)
            history["learning_rate"].append(current_lr)

            if early_stopping:
                if val_ce < best_val_loss:
                    best_val_loss = val_ce
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
                    f"Train CE: {train_ce:.4f} | "
                    f"Val CE: {val_ce:.4f}"
                )

        return history