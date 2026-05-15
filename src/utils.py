import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

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
    return np.divide(cm, row_sums, out=None, where=row_sums != 0)