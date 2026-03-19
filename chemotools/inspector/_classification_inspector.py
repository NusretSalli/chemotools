"""Classification Inspector for model diagnostics and visualization."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .core.base import InspectorDataset, _DataHoldingBase
from .core.spectra import SpectraMixin
from .core.summaries import ClassificationMetrics, ClassificationSummary
from .core.utils import (
    get_xlabel_for_features,
    normalize_datasets,
    prepare_color_values,
)
from .helpers._classification import (
    create_confidence_plot,
    create_confusion_matrix_plot,
    create_decision_boundary_plot,
    create_precision_recall_plot,
    create_probability_distribution_plot,
    create_roc_curve_plot,
)
from .helpers._preprocessing import create_preprocessing_step_plot

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class ClassificationInspector(SpectraMixin, _DataHoldingBase):
    """Inspector for classification model diagnostics and visualization.

    Accepts a fitted classifier or a scikit-learn
    :class:`~sklearn.pipeline.Pipeline` ending with a classifier.  When a
    Pipeline is provided the inspector can also visualise preprocessing
    effects step-by-step (via :meth:`inspect_preprocessing`) and compare
    raw vs. preprocessed spectra (via :meth:`inspect_spectra`).

    Parameters
    ----------
    model : classifier or Pipeline
        Fitted classifier or Pipeline ending with a classifier.
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training class labels.
    X_test : array-like of shape (n_samples, n_features), optional
        Test feature matrix.
    y_test : array-like of shape (n_samples,), optional
        Test class labels.
    X_val : array-like of shape (n_samples, n_features), optional
        Validation feature matrix.
    y_val : array-like of shape (n_samples,), optional
        Validation class labels.
    x_axis : array-like of shape (n_features,), optional
        Feature names (e.g. wavenumbers for spectroscopy).  If ``None``,
        integer indices are used.

    Attributes
    ----------
    model : classifier or Pipeline
        The original model object.
    classifier : classifier
        The underlying classifier (extracted from Pipeline if applicable).
    transformer : Pipeline or None
        Preprocessing pipeline (``None`` when the model is a bare classifier).
    classes : np.ndarray
        Unique class labels from the fitted classifier.
    n_classes : int
        Number of classes.
    has_predict_proba : bool
        Whether the classifier supports ``predict_proba``.

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVC
    >>> from chemotools.inspector import ClassificationInspector
    >>>
    >>> pipe = make_pipeline(StandardScaler(), SVC(probability=True))
    >>> pipe.fit(X_train, y_train)
    >>>
    >>> inspector = ClassificationInspector(pipe, X_train, y_train,
    ...                                     X_test=X_test, y_test=y_test)
    >>> inspector.summary()
    >>> inspector.inspect()
    >>> inspector.inspect_spectra()
    >>> inspector.inspect_preprocessing()
    >>> inspector.plot_decision_boundary()
    """

    def __init__(
        self,
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_axis: Optional[np.ndarray] = None,
    ) -> None:
        # --- Validate and extract model components ---------------------------
        classifier, transformer = self._validate_and_extract_classifier(model)
        self._model = model
        self.classifier_: object = classifier
        self.transformer_: Optional[Pipeline] = transformer

        # Preprocessing steps for step-by-step visualisation
        self._preprocessing_steps: List[Tuple[str, object]] = []
        if transformer is not None:
            self._preprocessing_steps = list(transformer.steps)

        # --- Validate and build datasets --------------------------------------
        X_train = check_array(
            X_train,
            dtype="numeric",
            ensure_2d=True,
            ensure_all_finite=True,
            input_name="X_train",
        )

        datasets: Dict[str, InspectorDataset] = {
            "train": InspectorDataset(
                X=X_train,
                y=self._validate_y(y_train, X_train.shape[0], "y_train"),
            ),
        }
        self._add_optional_dataset(datasets, "test", X_test, y_test, X_train.shape[1])
        self._add_optional_dataset(datasets, "val", X_val, y_val, X_train.shape[1])

        # --- Initialise data-holding base ------------------------------------
        super().__init__(
            datasets=datasets,
            n_features_in=X_train.shape[1],
            feature_names=x_axis,
        )

        # --- Classification info ---------------------------------------------
        self.classes_: np.ndarray = np.asarray(classifier.classes_)  # type: ignore[union-attr]
        self.n_classes_: int = len(self.classes_)
        self._has_proba: bool = hasattr(classifier, "predict_proba")
        self._is_binary: bool = self.n_classes_ == 2

        # Label encoding helpers
        self._class_to_idx: Dict = {c: i for i, c in enumerate(self.classes_)}
        self._class_names: List[str] = [str(c) for c in self.classes_]

        # Caches
        self._predictions_cache: Dict[str, np.ndarray] = {}
        self._probabilities_cache: Dict[str, np.ndarray] = {}
        self._pca_2d_ = None  # lazy-initialised for decision boundary

    # =========================================================================
    # Validation helpers
    # =========================================================================

    @staticmethod
    def _validate_and_extract_classifier(
        model: object,
    ) -> Tuple[object, Optional[Pipeline]]:
        """Validate the model and separate classifier from preprocessing.

        Returns
        -------
        classifier : object
            The classifier estimator.
        transformer : Pipeline or None
            Preprocessing pipeline (if the model was a Pipeline).

        Raises
        ------
        TypeError
            If the extracted estimator is not a classifier.
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        """
        check_is_fitted(model)

        if isinstance(model, Pipeline):
            classifier = model[-1]
            transformer = Pipeline(model.steps[:-1]) if len(model) > 1 else None
        else:
            classifier = model
            transformer = None

        if not is_classifier(classifier):
            raise TypeError(
                f"Expected a classifier, got {type(classifier).__name__}. "
                f"Pass a fitted classifier or a Pipeline ending with a classifier."
            )
        return classifier, transformer

    @staticmethod
    def _validate_y(
        y: Optional[np.ndarray], expected_n: int, name: str
    ) -> Optional[np.ndarray]:
        """Validate and normalise a target array."""
        if y is None:
            return None
        arr = check_array(
            y,
            dtype=None,
            ensure_2d=False,
            ensure_all_finite=True,
            input_name=name,
        )
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        if arr.shape[0] != expected_n:
            raise ValueError(
                f"{name} must have {expected_n} samples, got {arr.shape[0]}."
            )
        return arr

    @staticmethod
    def _add_optional_dataset(
        datasets: Dict[str, InspectorDataset],
        name: str,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        expected_features: int,
    ) -> None:
        """Validate and store an optional (test / val) dataset."""
        if X is None:
            return
        X = check_array(
            X,
            dtype="numeric",
            ensure_2d=True,
            ensure_all_finite=True,
            input_name=f"X_{name}",
        )
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X_{name} must have the same number of features as X_train. "
                f"Got {X.shape[1]} vs {expected_features}."
            )
        datasets[name] = InspectorDataset(
            X=X,
            y=ClassificationInspector._validate_y(y, X.shape[0], f"y_{name}"),
        )

    # =========================================================================
    # SpectraMixin protocol
    # =========================================================================

    @property
    def transformer(self) -> Optional[Pipeline]:
        """Return the preprocessing pipeline or ``None``."""
        return self.transformer_

    def _get_preprocessed_data(self, dataset: str) -> np.ndarray:
        """Return preprocessed X (through all preprocessing steps)."""
        if dataset in self._preprocessed_cache:
            return self._preprocessed_cache[dataset]

        X = self._get_dataset(dataset).X
        if self.transformer_ is None:
            result = X
        else:
            result = self.transformer_.transform(X)

        self._preprocessed_cache[dataset] = result
        return result

    def _get_preprocessed_x_axis(self) -> np.ndarray:
        """Return x-axis values after full preprocessing."""
        X_prep = self._get_preprocessed_data("train")
        return self._x_axis_for_n_features(X_prep.shape[1])

    def _x_axis_for_n_features(self, n_features_out: int) -> np.ndarray:
        """Return appropriate x-axis for the given output dimensionality."""
        if n_features_out == self.n_features_in_ and self.feature_names is not None:
            return self.feature_names.copy()
        if n_features_out == self.n_features_in_:
            return self._x_axis.copy()
        return np.arange(n_features_out)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def model(self) -> object:
        """Return the original model."""
        return self._model

    @property
    def classifier(self) -> object:
        """Return the underlying classifier."""
        return self.classifier_

    @property
    def classes(self) -> np.ndarray:
        """Return unique class labels."""
        return self.classes_

    @property
    def n_classes(self) -> int:
        """Return the number of classes."""
        return self.n_classes_

    @property
    def has_predict_proba(self) -> bool:
        """Whether the classifier supports ``predict_proba``."""
        return self._has_proba

    # =========================================================================
    # Prediction helpers (cached)
    # =========================================================================

    def _get_predictions(self, dataset: str) -> np.ndarray:
        """Return cached predictions for *dataset*."""
        if dataset not in self._predictions_cache:
            X_prep = self._get_preprocessed_data(dataset)
            self._predictions_cache[dataset] = self.classifier_.predict(X_prep)  # type: ignore[union-attr]
        return self._predictions_cache[dataset]

    def _get_probabilities(self, dataset: str) -> np.ndarray:
        """Return cached ``predict_proba`` output for *dataset*.

        Raises
        ------
        RuntimeError
            If the classifier does not support ``predict_proba``.
        """
        if not self._has_proba:
            raise RuntimeError(
                f"{type(self.classifier_).__name__} does not support "
                f"predict_proba. Use a classifier with probability support "
                f"(e.g. SVC(probability=True))."
            )
        if dataset not in self._probabilities_cache:
            X_prep = self._get_preprocessed_data(dataset)
            self._probabilities_cache[dataset] = self.classifier_.predict_proba(X_prep)  # type: ignore[union-attr]
        return self._probabilities_cache[dataset]

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode original class labels to integer indices."""
        return np.array([self._class_to_idx[v] for v in y])

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_accuracy(self, dataset: str = "train") -> float:
        """Return accuracy for the given dataset."""
        _, y = self._get_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        return float(accuracy_score(y, y_pred))

    def get_metrics(self, dataset: str = "train") -> ClassificationMetrics:
        """Return classification metrics for the given dataset.

        Returns
        -------
        ClassificationMetrics
            Dataclass with accuracy, precision, recall and f1.
        """
        _, y = self._get_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        average = "binary" if self._is_binary else "weighted"
        return ClassificationMetrics(
            accuracy=float(accuracy_score(y, y_pred)),
            precision=float(
                precision_score(y, y_pred, average=average, zero_division=0)
            ),
            recall=float(recall_score(y, y_pred, average=average, zero_division=0)),
            f1=float(f1_score(y, y_pred, average=average, zero_division=0)),
        )

    # =========================================================================
    # Individual plot methods
    # =========================================================================

    def plot_confusion_matrix(
        self,
        dataset: str = "train",
        *,
        normalize: bool = False,
        figsize: Tuple[float, float] = (8, 6),
    ) -> "Figure":
        """Plot the confusion matrix for *dataset*.

        Parameters
        ----------
        dataset : str, default='train'
            Dataset to evaluate.
        normalize : bool, default=False
            Normalise rows so each row sums to 1.
        figsize : tuple of float, default=(8, 6)
            Figure size.

        Returns
        -------
        Figure
        """
        _, y_true = self._get_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)
        return create_confusion_matrix_plot(
            cm,
            self._class_names,
            dataset,
            figsize=figsize,
            normalize=normalize,
        )

    def plot_roc_curve(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        *,
        figsize: Tuple[float, float] = (8, 6),
    ) -> "Figure":
        """Plot ROC curve (binary classifiers only).

        Parameters
        ----------
        dataset : str or sequence of str, default='train'
            Dataset(s) to include.
        figsize : tuple of float, default=(8, 6)
            Figure size.

        Returns
        -------
        Figure

        Raises
        ------
        ValueError
            If the classifier is not binary.
        RuntimeError
            If the classifier does not support ``predict_proba``.
        """
        if not self._is_binary:
            raise ValueError("ROC curve is only supported for binary classifiers.")

        datasets = normalize_datasets(dataset)
        datasets_roc: Dict[str, Dict[str, np.ndarray]] = {}

        for ds in datasets:
            _, y_true = self._get_raw_data(ds)
            proba = self._get_probabilities(ds)
            y_score = proba[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=self.classes_[1])
            auc = roc_auc_score(y_true, y_score)
            datasets_roc[ds] = {"fpr": fpr, "tpr": tpr, "auc": np.float64(auc)}

        return create_roc_curve_plot(datasets_roc, figsize=figsize)

    def plot_precision_recall(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        *,
        figsize: Tuple[float, float] = (8, 6),
    ) -> "Figure":
        """Plot precision-recall curve (binary classifiers only).

        Parameters
        ----------
        dataset : str or sequence of str, default='train'
            Dataset(s) to include.
        figsize : tuple of float, default=(8, 6)
            Figure size.

        Returns
        -------
        Figure

        Raises
        ------
        ValueError
            If the classifier is not binary.
        RuntimeError
            If the classifier does not support ``predict_proba``.
        """
        if not self._is_binary:
            raise ValueError(
                "Precision-recall curve is only supported for binary classifiers."
            )

        datasets = normalize_datasets(dataset)
        datasets_pr: Dict[str, Dict[str, np.ndarray]] = {}

        for ds in datasets:
            _, y_true = self._get_raw_data(ds)
            proba = self._get_probabilities(ds)
            y_score = proba[:, 1]
            precision, recall, _ = precision_recall_curve(
                y_true, y_score, pos_label=self.classes_[1]
            )
            ap = average_precision_score(y_true, y_score, pos_label=self.classes_[1])
            datasets_pr[ds] = {
                "precision": precision,
                "recall": recall,
                "ap": np.float64(ap),
            }

        return create_precision_recall_plot(datasets_pr, figsize=figsize)

    def plot_probability_distribution(
        self,
        dataset: str = "train",
        *,
        figsize: Tuple[float, float] = (10, 6),
    ) -> "Figure":
        """Plot predicted probability distributions per class.

        Parameters
        ----------
        dataset : str, default='train'
            Dataset to evaluate.
        figsize : tuple of float, default=(10, 6)
            Figure size.

        Returns
        -------
        Figure

        Raises
        ------
        RuntimeError
            If the classifier does not support ``predict_proba``.
        """
        _, y_true = self._get_raw_data(dataset)
        proba = self._get_probabilities(dataset)
        y_enc = self._encode_labels(y_true)
        return create_probability_distribution_plot(
            proba, y_enc, self._class_names, dataset, figsize=figsize
        )

    def plot_confidence(
        self,
        dataset: str = "train",
        *,
        figsize: Tuple[float, float] = (10, 5),
    ) -> "Figure":
        """Plot prediction confidence (max probability) per sample.

        Parameters
        ----------
        dataset : str, default='train'
            Dataset to evaluate.
        figsize : tuple of float, default=(10, 5)
            Figure size.

        Returns
        -------
        Figure

        Raises
        ------
        RuntimeError
            If the classifier does not support ``predict_proba``.
        """
        _, y_true = self._get_raw_data(dataset)
        y_pred = self._get_predictions(dataset)
        proba = self._get_probabilities(dataset)
        y_true_enc = self._encode_labels(y_true)
        y_pred_enc = self._encode_labels(y_pred)
        return create_confidence_plot(
            proba, y_true_enc, y_pred_enc, dataset, figsize=figsize
        )

    def plot_decision_boundary(
        self,
        dataset: str = "train",
        *,
        figsize: Tuple[float, float] = (8, 8),
        resolution: int = 100,
    ) -> "Figure":
        """Plot decision boundary projected to 2D via PCA.

        Fits ``PCA(n_components=2)`` on the preprocessed training data and
        projects both the meshgrid and the data points into this space.
        Grid predictions are obtained by inverse-transforming back to the
        preprocessed space and classifying — the boundary is therefore an
        *approximation* limited by the rank-2 reconstruction.

        Parameters
        ----------
        dataset : str, default='train'
            Dataset to plot.
        figsize : tuple of float, default=(8, 8)
            Figure size.
        resolution : int, default=100
            Number of grid points along each axis.

        Returns
        -------
        Figure
        """
        from sklearn.decomposition import PCA

        # Lazy-fit the PCA projection on training data
        if self._pca_2d_ is None:
            X_prep_train = self._get_preprocessed_data("train")
            pca = PCA(n_components=2)
            pca.fit(X_prep_train)
            self._pca_2d_ = pca

        X_prep = self._get_preprocessed_data(dataset)
        X_2d = self._pca_2d_.transform(X_prep)

        _, y_true = self._get_raw_data(dataset)
        y_enc = self._encode_labels(y_true)

        def _predict_encoded(X_grid: np.ndarray) -> np.ndarray:
            preds = self.classifier_.predict(X_grid)  # type: ignore[union-attr]
            return self._encode_labels(preds)

        return create_decision_boundary_plot(
            X_2d,
            y_enc,
            self._class_names,
            _predict_encoded,
            self._pca_2d_,
            dataset,
            figsize=figsize,
            resolution=resolution,
        )

    # =========================================================================
    # Composite inspection methods
    # =========================================================================

    def inspect(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        *,
        figsize: Tuple[float, float] = (8, 6),
    ) -> Dict[str, "Figure"]:
        """Create all standard classification diagnostic plots.

        Produces:

        - Confusion matrix (one per dataset)
        - ROC curve (binary + ``predict_proba`` only, all datasets together)
        - Precision-recall curve (binary + ``predict_proba`` only)
        - Probability distributions (``predict_proba`` only, one per dataset)
        - Prediction confidence (``predict_proba`` only, one per dataset)

        The decision boundary plot is *not* included because it is
        computationally expensive.  Call :meth:`plot_decision_boundary`
        separately.

        Parameters
        ----------
        dataset : str or sequence of str, default='train'
            Dataset(s) to visualise.
        figsize : tuple of float, default=(8, 6)
            Base figure size.

        Returns
        -------
        figures : dict of str to Figure
        """
        self.close_figures()
        datasets = normalize_datasets(dataset)

        figures: Dict[str, "Figure"] = {}

        # Confusion matrices
        for ds in datasets:
            figures[f"confusion_matrix_{ds}"] = self.plot_confusion_matrix(
                ds, figsize=figsize
            )

        # ROC and precision-recall (binary + proba)
        if self._is_binary and self._has_proba:
            figures["roc_curve"] = self.plot_roc_curve(datasets, figsize=figsize)
            figures["precision_recall"] = self.plot_precision_recall(
                datasets, figsize=figsize
            )

        # Probability-based plots
        if self._has_proba:
            for ds in datasets:
                figures[f"probability_distribution_{ds}"] = (
                    self.plot_probability_distribution(ds, figsize=(10, 6))
                )
                figures[f"confidence_{ds}"] = self.plot_confidence(ds, figsize=(10, 5))

        return self._track_figures(figures)

    def inspect_preprocessing(
        self,
        dataset: Union[str, Sequence[str]] = "train",
        color_by: Optional[Union[str, Dict[str, np.ndarray]]] = "y",
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 5),
        color_mode: Optional[Literal["continuous", "categorical"]] = None,
    ) -> Dict[str, "Figure"]:
        """Generate one plot per preprocessing step showing cumulative effects.

        Requires the model to be a Pipeline with preprocessing steps.

        Parameters
        ----------
        dataset : str or sequence of str, default='train'
            Dataset(s) to visualise.
        color_by : str or dict, default='y'
            Colouring specification (single-dataset mode only).
        xlim : tuple of float, optional
            X-axis limits for zooming.
        figsize : tuple of float, default=(12, 5)
            Figure size per subplot.
        color_mode : {'continuous', 'categorical'}, optional
            Override automatic colour-mode detection.

        Returns
        -------
        figures : dict of str to Figure

        Raises
        ------
        ValueError
            If the model has no preprocessing steps.
        """
        if not self._preprocessing_steps:
            raise ValueError(
                "No preprocessing steps to visualise. "
                "Pass a Pipeline with preprocessing steps before the classifier."
            )

        datasets = normalize_datasets(dataset)
        is_multi = len(datasets) > 1
        xlabel = get_xlabel_for_features(self.feature_names is not None)

        figures: Dict[str, "Figure"] = {}

        # --- Raw data plot ---------------------------------------------------
        if is_multi:
            figures["raw"] = self._plot_multi_dataset_step(
                datasets,
                step_data=None,
                title="Raw Spectra",
                xlabel=xlabel,
                xlim=xlim,
                figsize=figsize,
                color_mode=color_mode,
            )
        else:
            ds_name = datasets[0]
            ds = self._get_dataset(ds_name)
            color_values = prepare_color_values(color_by, ds_name, ds.y, ds.X.shape[0])
            figures["raw"] = create_preprocessing_step_plot(
                X=ds.X,
                x_axis=self._x_axis,
                title=f"Raw Spectra ({ds_name.capitalize()})",
                xlabel=xlabel,
                color_values=color_values,
                xlim=xlim,
                figsize=figsize,
                color_mode=color_mode,
            )

        # --- Cumulative preprocessing steps ----------------------------------
        cumulative: Dict[str, np.ndarray] = {
            ds: self._get_dataset(ds).X for ds in datasets
        }

        for step_idx, (step_name, step_transformer) in enumerate(
            self._preprocessing_steps, start=1
        ):
            for ds in datasets:
                cumulative[ds] = step_transformer.transform(cumulative[ds])  # type: ignore[union-attr]

            step_type = type(step_transformer).__name__
            fig_key = f"step_{step_idx}_{step_name}"
            title = f"Step {step_idx}: after {step_type}"

            if is_multi:
                step_x_axis = self._x_axis_for_n_features(
                    cumulative[datasets[0]].shape[1]
                )
                figures[fig_key] = self._plot_multi_dataset_step(
                    datasets,
                    step_data=dict(cumulative),
                    title=title,
                    xlabel=xlabel,
                    xlim=xlim,
                    figsize=figsize,
                    color_mode=color_mode,
                    step_x_axis=step_x_axis,
                )
            else:
                ds_name = datasets[0]
                ds = self._get_dataset(ds_name)
                step_x_axis = self._x_axis_for_n_features(cumulative[ds_name].shape[1])
                color_values = prepare_color_values(
                    color_by, ds_name, ds.y, ds.X.shape[0]
                )
                figures[fig_key] = create_preprocessing_step_plot(
                    X=cumulative[ds_name],
                    x_axis=step_x_axis,
                    title=f"{title} ({ds_name.capitalize()})",
                    xlabel=xlabel,
                    color_values=color_values,
                    xlim=xlim,
                    figsize=figsize,
                    color_mode=color_mode,
                )

        return self._track_figures(figures)

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> ClassificationSummary:
        """Return a summary of the classification model.

        Returns
        -------
        ClassificationSummary
        """
        preprocessing_steps = [
            {"step": i, "name": name, "type": type(t).__name__}
            for i, (name, t) in enumerate(self._preprocessing_steps, start=1)
        ]

        train_metrics = self.get_metrics("train")
        test_metrics = self.get_metrics("test") if "test" in self.datasets_ else None
        val_metrics = self.get_metrics("val") if "val" in self.datasets_ else None

        return ClassificationSummary(
            model_type=type(self.classifier_).__name__,
            has_preprocessing=self.transformer_ is not None,
            n_features=self.n_features_in_,
            n_samples=self.n_samples,
            n_classes=self.n_classes_,
            class_names=self._class_names,
            has_predict_proba=self._has_proba,
            preprocessing_steps=preprocessing_steps,
            train=train_metrics,
            test=test_metrics,
            val=val_metrics,
        )

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:  # noqa: D105
        datasets = ", ".join(
            f"{name}({ds.n_samples})" for name, ds in self.datasets_.items()
        )
        return (
            f"ClassificationInspector("
            f"classifier={type(self.classifier_).__name__}, "
            f"classes={self.n_classes_}, "
            f"features={self.n_features_in_}, "
            f"datasets=[{datasets}])"
        )

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _plot_multi_dataset_step(
        self,
        datasets: List[str],
        step_data: Optional[Dict[str, np.ndarray]],
        title: str,
        xlabel: str,
        xlim: Optional[Tuple[float, float]],
        figsize: Tuple[float, float],
        color_mode: Optional[Literal["continuous", "categorical"]],
        step_x_axis: Optional[np.ndarray] = None,
    ) -> "Figure":
        """Create a figure with multiple datasets overlaid."""
        import matplotlib.pyplot as plt

        from chemotools.plotting import SpectraPlot
        from chemotools.plotting._styles import DATASET_COLORS

        fig, ax = plt.subplots(figsize=figsize)

        for ds_name in datasets:
            if step_data is not None:
                X = step_data[ds_name]
                x_ax = step_x_axis if step_x_axis is not None else np.arange(X.shape[1])
            else:
                X = self._get_dataset(ds_name).X
                x_ax = self._x_axis

            color = DATASET_COLORS.get(ds_name, "black")
            labels: List[Optional[str]] = [ds_name.capitalize()] + [None] * (
                X.shape[0] - 1
            )

            plot = SpectraPlot(x=x_ax, y=X, labels=labels, color_mode=color_mode)
            plot.render(ax=ax, color=color, alpha=0.6, linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if xlim is not None:
            ax.set_xlim(xlim)

        fig.tight_layout()
        return fig
