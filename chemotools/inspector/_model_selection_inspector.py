"""Model Selection Inspector for candidate comparison and visualization."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .core.summaries import ModelSelectionSummary

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from chemotools.model_selection import CandidateSelector
    from chemotools.model_selection._fitted_model import BaseFittedModel


class ModelSelectionInspector:
    """Inspector for comparing model selection candidates.

    Wraps a fitted :class:`~chemotools.model_selection.CandidateSelector` and
    provides diagnostic plots and summaries following the inspector pattern
    used elsewhere in ``chemotools.inspector``.

    Parameters
    ----------
    selector : CandidateSelector
        A fitted ``CandidateSelector`` instance.

    Attributes
    ----------
    selector : CandidateSelector
        The underlying selector.
    candidates : list of BaseFittedModel
        Ranked candidate list (shortcut for ``selector.candidates_``).

    Examples
    --------
    >>> from chemotools.model_selection import CandidateSelector
    >>> from chemotools.inspector import ModelSelectionInspector
    >>>
    >>> selector = CandidateSelector(estimator, param_grid, scoring=...)
    >>> selector.fit(X_train, y_train)
    >>>
    >>> inspector = ModelSelectionInspector(selector)
    >>> inspector.summary()
    >>> figs = inspector.inspect()
    """

    def __init__(self, selector: CandidateSelector) -> None:
        check_is_fitted(selector, ["candidates_"])
        self._selector = selector
        self._tracked_figures: List[Figure] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def selector(self) -> CandidateSelector:
        """Return the underlying ``CandidateSelector``."""
        return self._selector

    @property
    def candidates(self) -> List[BaseFittedModel]:
        """Return the ranked candidate list."""
        return self._selector.candidates_

    @property
    def _has_rmse_metrics(self) -> bool:
        """Whether candidates have RMSE-specific metrics populated."""
        if not self.candidates:
            return False
        first = self.candidates[0]
        return first.rmsecv is not None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> ModelSelectionSummary:
        """Return a structured summary of the model selection results.

        Returns
        -------
        ModelSelectionSummary
        """
        sel = self._selector
        scoring = sel.scoring
        if callable(scoring):
            scoring = getattr(scoring, "__name__", str(scoring))

        return ModelSelectionSummary(
            estimator_type=type(sel.estimator).__name__,
            scoring=scoring,
            n_candidates=len(sel.candidates_),
            best_score=sel.best_score_,
            best_params=sel.best_params_,
            cv_folds=sel.cv,
        )

    # ------------------------------------------------------------------
    # Figure management
    # ------------------------------------------------------------------
    def close_figures(self) -> None:
        """Close all figures created by this inspector."""
        for fig in self._tracked_figures:
            plt.close(fig)
        self._tracked_figures.clear()

    def _track_figures(self, figures: Dict[str, Figure]) -> Dict[str, Figure]:
        """Track figures for later cleanup and return them."""
        self._tracked_figures.extend(figures.values())
        return figures

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def inspect(
        self,
        color_by: Optional[str] = None,
        *,
        figsize: Tuple[int, int] = (10, 6),
        show_ratio_threshold: Optional[float] = 1.0,
    ) -> Dict[str, Figure]:
        """Create all model selection diagnostic plots.

        Parameters
        ----------
        color_by : str, optional
            Parameter name to colour points by.  If ``None``, auto-detects
            the first parameter in the grid.
        figsize : tuple, default=(10, 6)
            Figure size for each plot.
        show_ratio_threshold : float or None, default=1.0
            If not ``None``, draws a horizontal reference line at this RMSE
            ratio value on the CV metrics plot.

        Returns
        -------
        dict of str to Figure
            Dictionary with keys ``'cv_metrics'`` and ``'score_vs_variance'``.
        """
        self.close_figures()

        figures: Dict[str, Figure] = {}

        # CV metrics plot
        try:
            fig_cv, ax_cv = plt.subplots(figsize=figsize)
            if self._has_rmse_metrics:
                self._create_scatter_plot(
                    x_metric="rmsecv",
                    y_metric="rmse_ratio",
                    color_by=color_by,
                    ax=ax_cv,
                    title="Cross-validation Error vs Overfitting",
                    xlabel="RMSECV",
                    ylabel="RMSECV / RMSEC",
                    hline=show_ratio_threshold,
                )
            else:
                self._create_scatter_plot(
                    x_metric="mean_test_score",
                    y_metric="mean_train_score",
                    color_by=color_by,
                    ax=ax_cv,
                    title="Cross-validation: Test Score vs Train Score",
                    xlabel="Mean Test Score",
                    ylabel="Mean Train Score",
                )
            figures["cv_metrics"] = fig_cv
        except ValueError:
            plt.close(fig_cv)

        # Score vs variance plot
        try:
            fig_sv, ax_sv = plt.subplots(figsize=figsize)
            self._create_scatter_plot(
                x_metric="variance",
                y_metric="mean_test_score",
                color_by=color_by,
                ax=ax_sv,
                title="Model Stability vs Performance",
                xlabel="Variance",
                ylabel="Mean Test Score",
            )
            figures["score_vs_variance"] = fig_sv
        except ValueError:
            plt.close(fig_sv)

        return self._track_figures(figures)

    def plot_cv_metrics(
        self,
        color_by: Optional[str] = None,
        *,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_ratio_threshold: Optional[float] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """Plot cross-validation metrics for model selection.

        When RMSE-based scoring was used, plots RMSECV vs RMSE ratio.
        Otherwise, plots Mean Test Score vs Mean Train Score.

        Parameters
        ----------
        color_by : str, optional
            Parameter name to colour points by.
        ax : Axes, optional
            Axes to plot on. If ``None``, creates a new figure.
        figsize : tuple, default=(10, 6)
            Figure size if creating a new figure.
        show_ratio_threshold : float or None, default=None
            Draws a horizontal line at this value. Only used for
            RMSE-based plots.
        title : str, optional
            Custom title.

        Returns
        -------
        Axes
        """
        if self._has_rmse_metrics:
            return self._create_scatter_plot(
                x_metric="rmsecv",
                y_metric="rmse_ratio",
                color_by=color_by,
                ax=ax,
                figsize=figsize,
                title=title or "Cross-validation Error vs Overfitting",
                xlabel="RMSECV",
                ylabel="RMSECV / RMSEC",
                hline=show_ratio_threshold,
            )
        return self._create_scatter_plot(
            x_metric="mean_test_score",
            y_metric="mean_train_score",
            color_by=color_by,
            ax=ax,
            figsize=figsize,
            title=title or "Cross-validation: Test Score vs Train Score",
            xlabel="Mean Test Score",
            ylabel="Mean Train Score",
        )

    def plot_score_vs_variance(
        self,
        color_by: Optional[str] = None,
        *,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ) -> Axes:
        """Plot test score vs variance for model selection.

        Parameters
        ----------
        color_by : str, optional
            Parameter name to colour points by.
        ax : Axes, optional
            Axes to plot on. If ``None``, creates a new figure.
        figsize : tuple, default=(10, 6)
            Figure size if creating a new figure.
        title : str, optional
            Custom title.

        Returns
        -------
        Axes
        """
        return self._create_scatter_plot(
            x_metric="variance",
            y_metric="mean_test_score",
            color_by=color_by,
            ax=ax,
            figsize=figsize,
            title=title or "Model Stability vs Performance",
            xlabel="Variance",
            ylabel="Mean Test Score",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_scatter_plot(
        self,
        x_metric: str,
        y_metric: str,
        color_by: Optional[str],
        ax: Optional[Axes],
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (10, 6),
        hline: Optional[float] = None,
    ) -> Axes:
        """Internal helper to create scatter plots with consistent styling."""
        candidates = self.candidates

        # Auto-detect color_by parameter
        if color_by is None and candidates:
            color_by = next(iter(candidates[0].params), None)

        # Group data by color_by parameter
        groups: Dict[Any, List[Tuple[float, float]]] = {}
        for c in candidates:
            x_val = getattr(c, x_metric, None) or c.to_dict().get(x_metric)
            y_val = getattr(c, y_metric, None) or c.to_dict().get(y_metric)
            if x_val is None or y_val is None:
                continue
            key = c.params.get(color_by) if color_by else c.rank
            groups.setdefault(key, []).append((x_val, y_val))

        if not groups:
            raise ValueError(
                f"No valid data found for metrics '{x_metric}' and '{y_metric}'."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        markers = ["o", "s", "^", "D", "v", "*", "p", "h"]
        cmap = plt.colormaps.get_cmap("tab10")

        for idx, key in enumerate(sorted(groups.keys())):
            data = groups[key]
            ax.scatter(
                [d[0] for d in data],
                [d[1] for d in data],
                marker=markers[idx % len(markers)],
                c=[cmap(idx % 10)],
                s=80,
                label=str(key),
                edgecolors="black",
                linewidths=0.5,
                alpha=0.8,
            )

        if hline is not None:
            ax.axhline(y=hline, linestyle="-", color="green", linewidth=2, alpha=0.8)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)

        param_label = (
            color_by.split("__")[-1] if color_by and "__" in color_by else color_by
        )
        ax.legend(title=param_label or "Group", loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        return ax
