"""Classification-specific plot creation functions for inspectors.

This module contains plotting functions specific to classification models
that are used by :class:`~chemotools.inspector.ClassificationInspector`.

Each function receives pre-computed data and returns a matplotlib Figure.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from chemotools.plotting._styles import DATASET_COLORS

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# ==============================================================================
# Confusion matrix
# ==============================================================================


def create_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    dataset_name: str,
    *,
    figsize: Tuple[float, float] = (8, 6),
    normalize: bool = False,
) -> "Figure":
    """Create a confusion matrix heatmap.

    Parameters
    ----------
    confusion_matrix : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix (rows = true, columns = predicted).
    class_names : list of str
        Display names for each class.
    dataset_name : str
        Name of the dataset (used in the title).
    figsize : tuple of float, default=(8, 6)
        Figure size in inches.
    normalize : bool, default=False
        If True, normalize rows to show proportions.

    Returns
    -------
    Figure
    """
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with no true samples
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = confusion_matrix.astype(float) / row_sums
    else:
        cm = confusion_matrix

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({dataset_name.capitalize()})",
    )
    ax.tick_params(axis="x", rotation=45)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    return fig


# ==============================================================================
# ROC curve (binary classification)
# ==============================================================================


def create_roc_curve_plot(
    datasets_roc: Dict[str, Dict[str, np.ndarray]],
    *,
    figsize: Tuple[float, float] = (8, 6),
) -> "Figure":
    """Create ROC curve plot for one or multiple datasets.

    Parameters
    ----------
    datasets_roc : dict of str to dict
        Mapping of dataset name to dict with ``'fpr'``, ``'tpr'``, ``'auc'``
        keys.
    figsize : tuple of float, default=(8, 6)
        Figure size in inches.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for ds_name, roc_data in datasets_roc.items():
        color = DATASET_COLORS.get(ds_name, "black")
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            color=color,
            linewidth=2,
            label=f"{ds_name.capitalize()} (AUC = {roc_data['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# Precision-recall curve (binary classification)
# ==============================================================================


def create_precision_recall_plot(
    datasets_pr: Dict[str, Dict[str, np.ndarray]],
    *,
    figsize: Tuple[float, float] = (8, 6),
) -> "Figure":
    """Create precision-recall curve plot for one or multiple datasets.

    Parameters
    ----------
    datasets_pr : dict of str to dict
        Mapping of dataset name to dict with ``'precision'``, ``'recall'``,
        ``'ap'`` (average precision) keys.
    figsize : tuple of float, default=(8, 6)
        Figure size in inches.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for ds_name, pr_data in datasets_pr.items():
        color = DATASET_COLORS.get(ds_name, "black")
        ax.plot(
            pr_data["recall"],
            pr_data["precision"],
            color=color,
            linewidth=2,
            label=f"{ds_name.capitalize()} (AP = {pr_data['ap']:.3f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# Probability distribution
# ==============================================================================


def create_probability_distribution_plot(
    probabilities: np.ndarray,
    y_true_encoded: np.ndarray,
    class_names: List[str],
    dataset_name: str,
    *,
    figsize: Tuple[float, float] = (10, 6),
) -> "Figure":
    """Plot predicted probability distributions per class.

    Creates one subplot per class, showing the distribution of
    ``P(class)`` coloured by the true class label.

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities from ``predict_proba``.
    y_true_encoded : np.ndarray of shape (n_samples,)
        Integer-encoded true labels (indices into *class_names*).
    class_names : list of str
        Display names for each class.
    dataset_name : str
        Name of the dataset (used in the title).
    figsize : tuple of float, default=(10, 6)
        Figure size in inches.

    Returns
    -------
    Figure
    """
    n_classes = len(class_names)
    ncols = min(n_classes, 3)
    nrows = math.ceil(n_classes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)

    cmap = plt.colormaps.get_cmap("tab10")

    for cls_idx in range(n_classes):
        row, col = divmod(cls_idx, ncols)
        ax = axes[row][col]
        probs_for_class = probabilities[:, cls_idx]

        for true_idx, true_name in enumerate(class_names):
            mask = y_true_encoded == true_idx
            if not np.any(mask):
                continue
            ax.hist(
                probs_for_class[mask],
                bins=20,
                alpha=0.6,
                color=cmap(true_idx),
                label=f"True: {true_name}",
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlabel(f"P({class_names[cls_idx]})", fontsize=10)
        ax.set_title(f"P({class_names[cls_idx]})", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_classes, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Predicted Probability Distributions ({dataset_name.capitalize()})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ==============================================================================
# Prediction confidence
# ==============================================================================


def create_confidence_plot(
    probabilities: np.ndarray,
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    dataset_name: str,
    *,
    figsize: Tuple[float, float] = (10, 5),
) -> "Figure":
    """Plot prediction confidence (max probability) per sample.

    Points are coloured green for correct predictions and red for incorrect.

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities.
    y_true_encoded : np.ndarray of shape (n_samples,)
        Integer-encoded true labels.
    y_pred_encoded : np.ndarray of shape (n_samples,)
        Integer-encoded predicted labels.
    dataset_name : str
        Name of the dataset (used in the title).
    figsize : tuple of float, default=(10, 5)
        Figure size in inches.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    max_probs = np.max(probabilities, axis=1)
    correct = y_true_encoded == y_pred_encoded

    ax.scatter(
        np.where(correct)[0],
        max_probs[correct],
        c="#2ca02c",
        alpha=0.6,
        s=30,
        label="Correct",
        edgecolors="black",
        linewidths=0.3,
    )
    ax.scatter(
        np.where(~correct)[0],
        max_probs[~correct],
        c="#d62728",
        alpha=0.8,
        s=50,
        marker="x",
        linewidths=1.5,
        label="Incorrect",
    )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Prediction Confidence (max probability)", fontsize=11)
    ax.set_title(
        f"Prediction Confidence ({dataset_name.capitalize()})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# Decision boundary (PCA 2D projection)
# ==============================================================================


def create_decision_boundary_plot(
    X_2d: np.ndarray,
    y_encoded: np.ndarray,
    class_names: List[str],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    pca_model: object,
    dataset_name: str,
    *,
    figsize: Tuple[float, float] = (8, 8),
    resolution: int = 100,
) -> "Figure":
    """Plot decision boundary in PCA-projected 2D space.

    The decision boundary is approximate: a meshgrid in PCA space is
    inverse-transformed back to the preprocessed feature space and fed
    through the classifier. Information lost in the PCA projection is
    reconstructed as the best rank-2 approximation.

    Parameters
    ----------
    X_2d : np.ndarray of shape (n_samples, 2)
        PCA-projected data.
    y_encoded : np.ndarray of shape (n_samples,)
        Integer-encoded labels.
    class_names : list of str
        Display names for each class.
    predict_fn : callable
        Callable that accepts data in the preprocessed feature space and
        returns integer-encoded predictions.
    pca_model : PCA
        Fitted ``PCA(n_components=2)`` used for inverse_transform.
    dataset_name : str
        Name of the dataset (used in the title).
    figsize : tuple of float, default=(8, 8)
        Figure size in inches.
    resolution : int, default=100
        Number of grid points along each axis.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    margin_x = (X_2d[:, 0].max() - X_2d[:, 0].min()) * 0.1
    margin_y = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.1
    x_min = X_2d[:, 0].min() - margin_x
    x_max = X_2d[:, 0].max() + margin_x
    y_min = X_2d[:, 1].min() - margin_y
    y_max = X_2d[:, 1].max() + margin_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca_model.inverse_transform(grid_2d)  # type: ignore[union-attr]
    grid_preds = predict_fn(grid_original)

    Z = grid_preds.reshape(xx.shape)

    cmap_bg = plt.colormaps.get_cmap("Pastel1")
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.3)

    cmap_pts = plt.colormaps.get_cmap("tab10")
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_encoded == cls_idx
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[cmap_pts(cls_idx)],
            label=cls_name,
            edgecolors="black",
            linewidths=0.5,
            s=60,
            alpha=0.8,
        )

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(
        f"Decision Boundary ({dataset_name.capitalize()})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
