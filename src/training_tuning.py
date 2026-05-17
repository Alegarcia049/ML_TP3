import time
import numpy as np
import gc
import pandas as pd
import itertools
from IPython.display import display
from src.utils import *

def train_and_evaluate_model(
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
    Train a model and return its ablation results.
    """
    start_time = time.time()

    history = model.fit(
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
        cm_val_norm = normalize_confusion_matrix(val_metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm_val_norm,
            title=f"{model_name} - Normalized Validation Confusion Matrix"
        )

    return result, history

def expand_param_grid(param_grid):
    """
    Expand a hyperparameter grid into a list of parameter combinations.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []

    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations

def stratified_split(X, y, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Create a stratified train/validation/test split using only NumPy."""
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

def stratified_k_fold_indices(y, k_folds=3):
    """
    Create stratified K-fold indices using only NumPy.

    Parameters
    ----------
    y : np.ndarray
        Integer class labels with shape (n_samples,).
    k_folds : int
        Number of folds.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples.
    """
    y = np.asarray(y).reshape(-1)

    folds = [[] for _ in range(k_folds)]

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        cls_folds = np.array_split(cls_indices, k_folds)

        for fold_id in range(k_folds):
            folds[fold_id].extend(cls_folds[fold_id])

    folds = [np.array(fold, dtype=int) for fold in folds]

    result = []

    all_indices = np.arange(len(y))

    for fold_id in range(k_folds):
        val_idx = folds[fold_id]
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        result.append((train_idx, val_idx))

    return result

def expand_kwargs_grid(kwargs_grid):
    """
    Expand a dictionary of hyperparameter lists into a list of dictionaries.
    """
    keys = list(kwargs_grid.keys())
    values = list(kwargs_grid.values())

    configs = []

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)

    return configs

def build_grid_configs(model_grid, fit_grid):
    """
    Build all model and fit configuration combinations.
    """
    model_configs = expand_kwargs_grid(model_grid)
    fit_configs = expand_kwargs_grid(fit_grid)

    grid_configs = []

    for model_kwargs in model_configs:
        for fit_kwargs in fit_configs:
            grid_configs.append({
                "model_kwargs": model_kwargs,
                "fit_kwargs": fit_kwargs
            })

    return grid_configs


def grid_search_cv_mlp(
    X,
    y_one_hot,
    model_cls,
    grid_configs,
    k_folds=3,
):
    """
    Run stratified cross-validation grid search using explicit model and fit configs.

    The best configuration is selected by lowest mean validation cross-entropy.
    """
    y_labels = np.argmax(y_one_hot, axis=1)

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
            y_train_fold = y_one_hot[train_idx]

            X_val_fold = X[val_idx]
            y_val_fold = y_one_hot[val_idx]

            model = model_cls(**model_kwargs)

            start_time = time.perf_counter()

            history = model.fit(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                **fit_kwargs
            )

            elapsed_time = time.perf_counter() - start_time

            train_metrics = model.evaluate(X_train_fold, y_train_fold)
            val_metrics = model.evaluate(X_val_fold, y_val_fold)

            fold_row = {
                "config_id": config_id,
                "fold": fold_id,
                "model_kwargs": model_kwargs.copy(),
                "fit_kwargs": fit_kwargs.copy(),
                "time": elapsed_time,
                "train_ce": train_metrics["cross_entropy"],
                "val_ce": val_metrics["cross_entropy"],
                "val_accuracy": val_metrics["accuracy"],
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