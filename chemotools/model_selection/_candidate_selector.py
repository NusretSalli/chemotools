"""
The :mod:`chemotools.model_selection._candidate_selector` module implements
model selection with enhanced candidate evaluation and RMSE metrics.
"""

# Authors: Nusret Emirhan Salli <nusret.emirhan.salli@gmail.com>
# License: MIT

import operator
import warnings
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from ._fitted_model import BaseFittedModel

__all__ = ["CandidateSelector"]


class CandidateSelector(BaseEstimator):
    """Model selection wrapper that produces ranked candidates with RMSE metrics.

    Wraps :class:`~sklearn.model_selection.GridSearchCV` and converts every
    evaluated parameter set into a :class:`BaseFittedModel` instance with
    RMSE-based metrics (RMSECV, RMSE ratio, etc.).

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn compatible estimator.
    param_grid : dict or list of dict
        Parameter grid to search.
    scoring : str or callable, default=None
        Scoring strategy (passed to ``GridSearchCV``).
    cv : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=None
        Parallelism for grid search.
    verbose : int, default=0
        Verbosity level.
    return_train_score : bool, default=True
        Whether to compute training scores (required for RMSE ratio).

    Attributes
    ----------
    cv_results_ : dict
        Raw results from the underlying ``GridSearchCV``.
    best_estimator_ : estimator
        Estimator refitted on the full training set with best parameters.
    best_params_ : dict
        Parameters that achieved the best score.
    best_score_ : float
        Best cross-validation score.
    candidates_ : list of BaseFittedModel
        All evaluated candidates sorted by rank.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Union[Dict, List[Dict]],
        *,
        scoring: Optional[Union[str, Callable[..., float]]] = None,
        cv: int = 5,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        return_train_score: bool = True,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.return_train_score = return_train_score

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params
    ) -> "CandidateSelector":
        """Run grid search and build candidate list."""
        grid = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=self.return_train_score,
            refit=True,
        )
        grid.fit(X, y, **fit_params)

        self.cv_results_ = grid.cv_results_
        self.best_estimator_ = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_
        self.best_index_ = grid.best_index_

        self.candidates_ = self._build_candidates()
        return self

    def _build_candidates(self) -> List[BaseFittedModel]:
        """Create sorted list of ``BaseFittedModel`` from cv_results_."""
        candidates = []
        for idx in range(len(self.cv_results_["params"])):
            candidate = BaseFittedModel.from_cv_results(
                estimator=self.estimator,
                cv_results=self.cv_results_,
                index=idx,
                scoring=self.scoring,
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: (c.rank or float("inf"), c.cv_results_index or 0))
        return candidates

    def get_candidates(self, n: Optional[int] = None) -> List[BaseFittedModel]:
        """Return top *n* candidates (all if n is None)."""
        check_is_fitted(self, ["candidates_"])
        if n is None:
            return self.candidates_
        return self.candidates_[:n]

    def get_candidate(self, rank: int = 1) -> BaseFittedModel:
        """Return candidate by rank (1 = best)."""
        check_is_fitted(self, ["candidates_"])
        for c in self.candidates_:
            if c.rank == rank:
                return c
        raise ValueError(f"No candidate with rank {rank}.")

    def filter_candidates(
        self,
        metric: str = "rmse_ratio",
        threshold: float = 1.1,
        mode: str = "<=",
    ) -> List[BaseFittedModel]:
        """Filter candidates based on a metric threshold.

        Parameters
        ----------
        metric : str, default='rmse_ratio'
            The metric to filter by. Options: 'rmsecv', 'rmse_train', 'rmse_ratio',
            'mean_test_score', 'std_test_score', 'variance'.
        threshold : float, default=1.1
            The threshold value for filtering.
        mode : str, default='<='
            Comparison mode. Options: '<=', '>=', '<', '>', '=='.

        Returns
        -------
        list of BaseFittedModel
            Candidates that match the filter criteria.

        Examples
        --------
        >>> # Get candidates with RMSE ratio <= 1.2 (good generalization)
        >>> robust = selector.filter_candidates(metric='rmse_ratio', threshold=1.2)
        >>> # Get candidates with low variance (stable across folds)
        >>> stable = selector.filter_candidates(metric='variance', threshold=0.01)
        """

        check_is_fitted(self, ["candidates_"])

        ops = {
            "<=": operator.le,
            ">=": operator.ge,
            "<": operator.lt,
            ">": operator.gt,
            "==": operator.eq,
        }
        cmp = ops.get(mode)
        if cmp is None:
            raise ValueError(f"mode must be one of {list(ops.keys())}")

        result = []
        for c in self.candidates_:
            val = getattr(c, metric, None)
            if val is None:
                val = c.to_dict().get(metric)
            if val is not None and cmp(val, threshold):
                result.append(c)
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best estimator."""
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score using best estimator."""
        check_is_fitted(self, ["best_estimator_"])
        return self.best_estimator_.score(X, y)

    def __len__(self) -> int:
        """Return the number of candidates."""
        check_is_fitted(self, ["candidates_"])
        return len(self.candidates_)

    def __iter__(self) -> Iterator[BaseFittedModel]:
        """Iterate over candidates."""
        check_is_fitted(self, ["candidates_"])
        return iter(self.candidates_)

    def summary(self, n: int = 10) -> str:
        """Return a formatted summary of the top candidates.

        Parameters
        ----------
        n : int, default=10
            Number of top candidates to include in summary.

        Returns
        -------
        str
            Formatted summary string.
        """
        check_is_fitted(self, ["candidates_"])

        lines = [
            "CandidateSelector Summary",
            f"{'=' * 60}",
            f"Total candidates: {len(self.candidates_)}",
            f"Best score: {self.best_score_:.6f}",
            f"Best params: {self.best_params_}",
            "",
            f"Top {min(n, len(self.candidates_))} Candidates:",
            f"{'-' * 60}",
        ]

        # Header
        lines.append(
            f"{'Rank':>4} {'RMSECV':>10} {'RMSE_train':>10} {'Ratio':>8} {'Variance':>12}"
        )
        lines.append(f"{'-' * 4} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12}")

        for c in self.candidates_[:n]:
            rmsecv = f"{c.rmsecv:.4f}" if c.rmsecv is not None else "N/A"
            rmse_train = f"{c.rmse_train:.4f}" if c.rmse_train is not None else "N/A"
            ratio = f"{c.rmse_ratio:.3f}" if c.rmse_ratio is not None else "N/A"
            var = f"{c.variance:.2e}" if c.variance is not None else "N/A"
            lines.append(
                f"{c.rank:>4} {rmsecv:>10} {rmse_train:>10} {ratio:>8} {var:>12}"
            )

        return "\n".join(lines)

    def to_dataframe(self):
        """Convert all candidates to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with all candidate metrics and parameters.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        check_is_fitted(self, ["candidates_"])

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install pandas"
            )

        records = []
        for c in self.candidates_:
            record = c.to_dict()
            # Flatten params into separate columns
            for key, value in c.params.items():
                record[f"param_{key}"] = value
            records.append(record)

        return pd.DataFrame(records)

    def plot_cv_metrics(self, *args, **kwargs):
        """Plot RMSECV vs RMSE ratio.

        .. deprecated::
            Use ``ModelSelectionInspector(selector).plot_cv_metrics()`` instead.
        """
        warnings.warn(
            "CandidateSelector.plot_cv_metrics() is deprecated. "
            "Use ModelSelectionInspector(selector).plot_cv_metrics() instead.",
            FutureWarning,
            stacklevel=2,
        )
        from chemotools.inspector._model_selection_inspector import (
            ModelSelectionInspector,
        )

        return ModelSelectionInspector(self).plot_cv_metrics(*args, **kwargs)

    def plot_score_vs_variance(self, *args, **kwargs):
        """Plot test score vs variance.

        .. deprecated::
            Use ``ModelSelectionInspector(selector).plot_score_vs_variance()``
            instead.
        """
        warnings.warn(
            "CandidateSelector.plot_score_vs_variance() is deprecated. "
            "Use ModelSelectionInspector(selector).plot_score_vs_variance() instead.",
            FutureWarning,
            stacklevel=2,
        )
        from chemotools.inspector._model_selection_inspector import (
            ModelSelectionInspector,
        )

        return ModelSelectionInspector(self).plot_score_vs_variance(*args, **kwargs)
