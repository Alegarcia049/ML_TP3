import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import display

def stratified_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """Create a stratified train/validation/test split using only NumPy."""

    train_indices = []
    val_indices = []
    test_indices = []

    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

def show_label_distribution(y, name):
    """Print the class distribution of a label vector."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)

    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=classes, color='skyblue', edgecolor='black', rwidth=0.8)
    plt.title(f'Distribución de labels en {name}', fontsize=15)
    plt.xlabel('Código de la Clase (Letra/Número)', fontsize=12)
    plt.ylabel('Frecuencia (Cantidad de imágenes)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def flatten_images(X):
    """Flatten image tensors from (N, 28, 28) to (N, 784)."""
    return X.reshape(X.shape[0], -1)

def one_hot_encode(y, num_classes):
    """Convert integer labels into one-hot encoded vectors."""
    y = y.astype(int)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def plot_loss_curves(history):
    """
    Plot train and validation cross-entropy loss curves.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_loss"], label="Train CE")

    if len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"], label="Validation CE")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.title("M0 - Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    plt.figure(figsize=(8, 5))

    sns.heatmap(
        cm,
        cmap="Blues",
        square=True,
        cbar=True
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.show()

def normalize_confusion_matrix(cm):
    """
    Normalize confusion matrix by true class counts.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, where=row_sums != 0)

def train_and_evaluate_model(
    model_name,
    change_description,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    train_config,
    plot=False
    ):
    """
    Train a model and return its ablation results.
    """
    start_time = time.time()

    history = model.fit_advanced(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        **train_config
    )

    elapsed_time = time.time() - start_time

    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)

    result = {
        "Model": model_name,
        "Change": change_description,
        "Time": elapsed_time,
        "Train CE": train_metrics["cross_entropy"],
        "Val CE": val_metrics["cross_entropy"],
        "Train Accuracy": train_metrics["accuracy"],
        "Val Accuracy": val_metrics["accuracy"],
        "Train F1 Macro": train_metrics["f1 macro"],
        "Val F1 Macro": val_metrics["f1_macro"]
    }

    if plot:
        display(result)
        plot_loss_curves(history)
        plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            title=f"{model_name} - Validation Confusion Matrix"
        )
        cm_val_norm = normalize_confusion_matrix(val_metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm_val_norm,
            title=f"{model_name} - Normalized Validation Confusion Matrix"
        )

    return result, history