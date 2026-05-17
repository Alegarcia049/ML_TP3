import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import time
from src.training_tuning import *

class TorchMLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier implemented with PyTorch.

    This model is intended to replicate the architecture of the best NumPy MLP
    model, but using PyTorch modules and autograd.
    """

    def __init__(
        self,
        layer_sizes,
        use_batch_norm=False,
        bn_momentum=0.9,
        bn_epsilon=1e-5,
        device=None,
        activation_name="relu",
        dropout_rate=0.0
    ):
        """
        Initialize the PyTorch MLP model.

        Parameters
        ----------
        layer_sizes : list[int]
            Network architecture. Example: [784, 128, 64, 47].
        use_batch_norm : bool
            Whether to apply batch normalization after hidden linear layers.
        bn_momentum : float
            Momentum equivalent to the NumPy implementation.
        bn_epsilon : float
            Numerical stability constant for batch normalization.
        device : str, optional
            Device used for training and inference.
        activation_name : str, optional
            Activation function used in hidden layers
        dropout_rate : float, optional
            Dropout rate used in training
        """
        super().__init__()

        self.layer_sizes = layer_sizes
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.activation_name = activation_name
        self.dropout_rate = dropout_rate
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.network = self._build_network()
        self.to(self.device)

    def _get_activation_layer(self):
        """
        Return the selected hidden-layer activation.
        """
        if self.activation_name == "relu":
            return nn.ReLU()

        if self.activation_name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)

        if self.activation_name in ["silu", "swish"]:
            return nn.SiLU()

        if self.activation_name == "gelu":
            return nn.GELU()

        raise ValueError(
            "activation_name must be 'relu', 'leaky_relu', 'silu', "
            "'swish', or 'gelu'"
        )
    
    def _build_network(self):
        """
        Build the neural network architecture.
        """
        layers = []

        for i in range(len(self.layer_sizes) - 2):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i + 1]

            linear_layer = nn.Linear(in_features, out_features)
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
            nn.init.zeros_(linear_layer.bias)

            layers.append(linear_layer)

            if self.use_batch_norm:
                layers.append(
                    nn.BatchNorm1d(
                        num_features=out_features,
                        eps=self.bn_epsilon,
                        momentum=1 - self.bn_momentum
                    )
                )

            layers.append(self._get_activation_layer())

            if self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))

        output_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1]
        )

        nn.init.kaiming_normal_(output_layer.weight, nonlinearity="relu")
        nn.init.zeros_(output_layer.bias)

        layers.append(output_layer)

        return nn.Sequential(*layers)

    def forward(self, X):
        """
        Run the forward pass.

        Important:
        This method returns logits, not probabilities.
        CrossEntropyLoss applies log-softmax internally.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)

        return self.network(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        self.eval()

        with torch.no_grad():
            logits = self.forward(X)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, X):
        """
        Predict class labels.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def _prepare_targets(self, y):
        """
        Convert labels to class indices.

        Accepts either class labels with shape (n_samples,)
        or one-hot labels with shape (n_samples, n_classes).
        """
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device)
        else:
            y = y.to(self.device)

        if y.ndim == 2:
            y = torch.argmax(y, dim=1)

        return y.long()

    def evaluate(self, X, y):
        """
        Evaluate model using cross-entropy and accuracy.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input features.
        y : np.ndarray or torch.Tensor
            Class labels or one-hot encoded labels.

        Returns
        -------
        dict
            Dictionary with cross-entropy, accuracy, y_true and y_pred.
        """
        self.eval()

        y_true = self._prepare_targets(y)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self.forward(X)
            loss = criterion(logits, y_true)

            y_pred = torch.argmax(logits, dim=1)
            accuracy = torch.mean((y_pred == y_true).float())

        return {
            "cross_entropy": loss.item(),
            "accuracy": accuracy.item(),
            "y_true": y_true.cpu().numpy(),
            "y_pred": y_pred.cpu().numpy()
        }
    
def _prepare_targets(y, device):
    """
    Convert labels to class indices.

    Accepts labels with shape (n_samples,) or one-hot labels with shape
    (n_samples, n_classes).
    """
    y_tensor = torch.as_tensor(y).to(device)

    if y_tensor.ndim == 2:
        y_tensor = torch.argmax(y_tensor, dim=1)

    return y_tensor.long()


def _get_learning_rate(
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


def _set_optimizer_lr(optimizer, learning_rate):
    """
    Update optimizer learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def _l2_penalty(model, l2_lambda, batch_size):
    """
    Compute L2 penalty only over weight matrices, not biases.

    This is closer to the NumPy implementation than using optimizer weight_decay.
    """
    if l2_lambda == 0:
        return 0.0

    penalty = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name and param.ndim > 1:
            penalty = penalty + torch.sum(param ** 2)

    return (l2_lambda / (2 * batch_size)) * penalty


def labels_from_targets(y):
    """
    Convert labels to class indices.
    """
    y = np.asarray(y)

    if y.ndim == 2:
        return np.argmax(y, axis=1)

    return y.reshape(-1).astype(int)


def f1_macro_from_confusion_matrix(cm):
    """
    Compute macro F1-score from a confusion matrix.
    """
    f1_scores = []

    for cls in range(cm.shape[0]):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp
        fn = np.sum(cm[cls, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def evaluate_torch_model(
    model,
    X,
    y,
    batch_size=1024,
    num_classes=None,
    compute_confusion=True
):
    """
    Evaluate a PyTorch classifier with CE, accuracy, confusion matrix and F1.
    """
    model.eval()

    device = model.device

    y_labels = labels_from_targets(y)

    if num_classes is None:
        num_classes = model.layer_sizes[-1]

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    cm = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            y_pred = torch.argmax(logits, dim=1)

            current_batch_size = X_batch.shape[0]

            total_loss += loss.item() * current_batch_size
            total_correct += torch.sum(y_pred == y_batch).item()
            total_samples += current_batch_size

            if compute_confusion:
                y_true_np = y_batch.cpu().numpy()
                y_pred_np = y_pred.cpu().numpy()

                np.add.at(cm, (y_true_np, y_pred_np), 1)

    result = {
        "cross_entropy": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }

    if compute_confusion:
        result["confusion_matrix"] = cm
        result["f1_macro"] = f1_macro_from_confusion_matrix(cm)

    return result


def train_torch_model(
    model,
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
    label_smoothing=0.0,
    gradient_clip_norm=None,
    beta1=0.9,
    beta2=0.999,
    adam_epsilon=1e-8,
    verbose=False
):
    """
    Train a TorchMLPClassifier using mini-batch optimization.

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
    device = model.device

    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_tensor = _prepare_targets(y_train, device="cpu")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    if optimizer == "gd":
        torch_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=initial_lr
        )

    elif optimizer == "adam":
        torch_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=initial_lr,
            betas=(beta1, beta2),
            eps=adam_epsilon
        )

    else:
        raise ValueError("optimizer must be 'gd' or 'adam'")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_time": []
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        start_time = time.time()

        current_lr = _get_learning_rate(
            epoch=epoch,
            initial_lr=initial_lr,
            scheduler=scheduler,
            final_lr=final_lr,
            decay_rate=decay_rate,
            max_epochs=epochs
        )

        _set_optimizer_lr(torch_optimizer, current_lr)

        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            torch_optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            if l2_lambda > 0:
                loss = loss + _l2_penalty(
                    model=model,
                    l2_lambda=l2_lambda,
                    batch_size=X_batch.shape[0]
                )

            loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm
                )

            torch_optimizer.step()

        train_metrics = evaluate_torch_model(
            model=model,
            X=X_train,
            y=y_train,
            batch_size=1024
        )

        val_metrics = evaluate_torch_model(
            model=model,
            X=X_val,
            y=y_val,
            batch_size=1024
        )

        train_loss = train_metrics["cross_entropy"]
        val_loss = val_metrics["cross_entropy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(time.time() - start_time)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch:03d}")

                model.load_state_dict(best_state)
                break

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"LR: {current_lr:.6f} | "
                f"Train CE: {train_loss:.4f} | "
                f"Val CE: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

    return history

def train_and_evaluate_torch_model(
    model_name,
    implementation,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    train_config,
    plot=False
):
    """
    Train a PyTorch model and return ablation-compatible results.
    """
    start_time = time.time()

    history = train_torch_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        **train_config
    )

    elapsed_time = time.time() - start_time

    num_classes = model.layer_sizes[-1]

    train_metrics = evaluate_torch_model(
        model=model,
        X=X_train,
        y=y_train,
        num_classes=num_classes,
        compute_confusion=True
    )

    val_metrics = evaluate_torch_model(
        model=model,
        X=X_val,
        y=y_val,
        num_classes=num_classes,
        compute_confusion=True
    )

    result = {
        "Model": model_name,
        "Implementation": implementation,
        "Time [sec]": elapsed_time,
        "Train CE": train_metrics["cross_entropy"],
        "Val CE": val_metrics["cross_entropy"],
        "Train Accuracy": train_metrics["accuracy"],
        "Val Accuracy": val_metrics["accuracy"],
        "Train F1 Macro": train_metrics["f1_macro"],
        "Val F1 Macro": val_metrics["f1_macro"]
    }

    result = pd.Series(result)

    if plot:
        display(result)
        plot_loss_curves(history)

        plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            title=f"{model_name} - Validation Confusion Matrix"
        )

        cm_val_norm = normalize_confusion_matrix(
            val_metrics["confusion_matrix"]
        )

        plot_confusion_matrix(
            cm_val_norm,
            title=f"{model_name} - Normalized Validation Confusion Matrix"
        )

    return result, history

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix using NumPy.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    return matrix


def f1_macro(y_true, y_pred, num_classes):
    """
    Compute macro F1-score using NumPy.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)

    f1_scores = []

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp
        fn = np.sum(cm[cls, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def complete_torch_metrics(model, X, y, num_classes):
    """
    Complete model.evaluate output with confusion matrix and macro F1.
    """
    y_true = labels_from_targets(y)

    metrics = model.evaluate(X, y_true)

    y_pred = model.predict(X)

    metrics["confusion_matrix"] = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes
    )

    metrics["f1_macro"] = f1_macro(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes
    )

    return metrics

def grid_search_cv_torch_mlp(
    X,
    y,
    model_cls,
    grid_configs,
    k_folds=3,
    clear_cuda_cache=True
):
    """
    Run a PyTorch MLP hyperparameter search.

    If k_folds=1, uses a single stratified holdout split.
    If k_folds>1, uses stratified K-fold cross-validation.
    """
    y_labels = labels_from_targets(y)

    if np.asarray(y).ndim == 2:
        num_classes = np.asarray(y).shape[1]
    else:
        num_classes = int(np.max(y_labels)) + 1

    folds = stratified_k_fold_indices(
        y=y_labels,
        k_folds=k_folds
    )

    summary_rows = []
    all_runs = []

    for config_id, config in enumerate(grid_configs, start=1):
        model_kwargs = config["model_kwargs"]
        fit_kwargs = config["fit_kwargs"]

        print(f"\nConfig {config_id}/{len(grid_configs)}")
        print("Model:", model_kwargs)
        print("Fit:", fit_kwargs)

        fold_rows = []

        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train_fold = X[train_idx]
            y_train_fold = y_labels[train_idx]

            X_val_fold = X[val_idx]
            y_val_fold = y_labels[val_idx]

            model = model_cls(**model_kwargs)

            start_time = time.perf_counter()

            history = train_torch_model(
                model=model,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                **fit_kwargs
            )

            elapsed_time = time.perf_counter() - start_time

            train_metrics = evaluate_torch_model(
                model=model,
                X=X_train_fold,
                y=y_train_fold,
                num_classes=num_classes,
                compute_confusion=True
            )

            val_metrics = evaluate_torch_model(
                model=model,
                X=X_val_fold,
                y=y_val_fold,
                num_classes=num_classes,
                compute_confusion=True
            )

            fold_row = {
                "config_id": config_id,
                "fold": fold_id,
                "model_kwargs": model_kwargs.copy(),
                "fit_kwargs": fit_kwargs.copy(),
                "time": elapsed_time,
                "train_ce": train_metrics["cross_entropy"],
                "val_ce": val_metrics["cross_entropy"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_f1_macro": val_metrics["f1_macro"]
            }

            fold_rows.append(fold_row)

            print(
                f"Fold {fold_id} | "
                f"Val CE: {val_metrics['cross_entropy']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            del model
            del history
            del X_train_fold, X_val_fold
            del y_train_fold, y_val_fold

            gc.collect()

            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_ce_values = [row["train_ce"] for row in fold_rows]
        val_ce_values = [row["val_ce"] for row in fold_rows]
        val_acc_values = [row["val_accuracy"] for row in fold_rows]
        val_f1_values = [row["val_f1_macro"] for row in fold_rows]
        time_values = [row["time"] for row in fold_rows]

        summary_rows.append({
            "config_id": config_id,
            "model_kwargs": model_kwargs.copy(),
            "fit_kwargs": fit_kwargs.copy(),
            "mean_train_ce": float(np.mean(train_ce_values)),
            "mean_val_ce": float(np.mean(val_ce_values)),
            "std_val_ce": float(np.std(val_ce_values)),
            "mean_val_accuracy": float(np.mean(val_acc_values)),
            "std_val_accuracy": float(np.std(val_acc_values)),
            "mean_val_f1_macro": float(np.mean(val_f1_values)),
            "std_val_f1_macro": float(np.std(val_f1_values)),
            "mean_time": float(np.mean(time_values)),
            "total_time": float(np.sum(time_values)),
            "n_folds": k_folds,
        })

        all_runs.append({
            "config_id": config_id,
            "model_kwargs": model_kwargs.copy(),
            "fit_kwargs": fit_kwargs.copy(),
            "fold_runs": fold_rows,
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        by="mean_val_ce",
        ascending=True
    ).reset_index(drop=True)

    best_row = summary.iloc[0]

    best_run = next(
        run for run in all_runs
        if run["config_id"] == best_row["config_id"]
    )

    return pd.Series({
        "summary": summary,
        "best_model_kwargs": best_row["model_kwargs"],
        "best_fit_kwargs": best_row["fit_kwargs"],
        "best_score": float(best_row["mean_val_ce"]),
        "best_run": best_run,
        "all_runs": all_runs,
    })

def add_gaussian_noise(X, noise_std):
    """Add Gaussian noise to normalized images and clip values to [0, 1]."""
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=X.shape
    )

    X_noisy = X + noise
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    return X_noisy.astype(np.float32)


def evaluate_noise_robustness(
    models_info,
    X_test,
    y_test,
    noise_levels
):
    """
    Evaluate robustness of several trained models under Gaussian noise.

    Parameters
    ----------
    models_info : list[dict]
        Each dict must contain: name, implementation, model, model_type.
    X_test : np.ndarray
        Test features normalized in [0, 1].
    y_test : np.ndarray
        Test labels, either one-hot or integer encoded.
    noise_levels : list[float]
        Gaussian noise standard deviations.
    random_state : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        Robustness results for all models and noise levels.
    """
    rows = []

    for noise_std in noise_levels:
        if noise_std == 0.0:
            X_eval = X_test.astype(np.float32)
        else:
            X_eval = add_gaussian_noise(
                X=X_test,
                noise_std=noise_std
            )

        for model_info in models_info:
            if model_info["model_type"] == "numpy":
                metrics = model_info["model"].evaluate(X_eval, y_test)
            else: 
                metrics = evaluate_torch_model(model_info["model"], X_eval, y_test, compute_confusion=True)

            rows.append({
                "Model": model_info["name"],
                "Implementation": model_info["implementation"],
                "Noise Std": noise_std,
                "Test CE": metrics["cross_entropy"],
                "Test Accuracy": metrics["accuracy"],
                "Test F1 Macro": metrics["f1_macro"]
            })

    results = pd.DataFrame(rows)

    clean_results = results[results["Noise Std"] == 0.0][
        ["Model", "Test CE", "Test Accuracy", "Test F1 Macro"]
    ].rename(columns={
        "Test CE": "Clean Test CE",
        "Test Accuracy": "Clean Test Accuracy",
        "Test F1 Macro": "Clean Test F1 Macro"
    })

    results = results.merge(
        clean_results,
        on="Model",
        how="left"
    )

    results["CE Increase"] = (
        results["Test CE"] - results["Clean Test CE"]
    )

    results["Accuracy Drop"] = (
        results["Clean Test Accuracy"] - results["Test Accuracy"]
    )

    results["F1 Drop"] = (
        results["Clean Test F1 Macro"] - results["Test F1 Macro"]
    )

    return results