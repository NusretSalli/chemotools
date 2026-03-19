from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RegressionMetrics:
    rmse: float
    r2: float
    bias: float


@dataclass(kw_only=True)
class InspectorSummary:
    """Base class for all inspector summaries."""

    model_type: str
    has_preprocessing: bool
    n_features: int
    n_samples: Dict[str, int]
    preprocessing_steps: List[Dict[str, Any]]

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(kw_only=True)
class LatentSummary:
    n_components: int
    hotelling_t2_limit: float
    q_residuals_limit: float


@dataclass(kw_only=True)
class RegressionSummary:
    train: RegressionMetrics
    test: Optional[RegressionMetrics] = None
    val: Optional[RegressionMetrics] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """Get regression metrics as a dictionary suitable for pandas DataFrame.

        Returns a dictionary where keys are metric names (e.g. 'rmse', 'r2')
        and values are dictionaries mapping dataset names to metric values.
        This structure results in a DataFrame where:
        - Columns are metrics (RMSE, R2, Bias)
        - Rows are datasets (train, test, val)
        """
        # 1. Collect data by dataset
        by_dataset = {}
        for dataset in ["train", "test", "val"]:
            obj = getattr(self, dataset)
            if obj is not None:
                by_dataset[dataset] = asdict(obj)

        if not by_dataset:
            return {}

        # 2. Invert to be by metric (for DataFrame columns)
        # Assuming all datasets have the same metrics (defined by RegressionMetrics)
        metric_names = next(iter(by_dataset.values())).keys()

        return {
            metric: {ds: values[metric] for ds, values in by_dataset.items()}
            for metric in metric_names
        }


@dataclass(kw_only=True)
class PCASummary(InspectorSummary, LatentSummary):
    """Summary for PCA models."""

    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    pc_variances: Dict[str, float]
    total_variance: float
    variance_thresholds: Dict[str, Dict[str, Any]]


@dataclass(kw_only=True)
class PLSRegressionSummary(InspectorSummary, LatentSummary, RegressionSummary):
    """Summary for PLS Regression models."""

    explained_x_variance_ratio: Optional[List[float]] = None
    total_x_variance: Optional[float] = None
    explained_y_variance_ratio: Optional[List[float]] = None
    total_y_variance: Optional[float] = None


@dataclass(kw_only=True)
class PreprocessingSummary:
    """Summary for preprocessing pipelines.

    Attributes
    ----------
    pipeline_type : str
        Class name of the pipeline (e.g. ``'Pipeline'``).
    total_steps : int
        Total number of steps in the pipeline (including model steps).
    n_preprocessing_steps : int
        Number of preprocessing steps that are visualised.
    n_excluded_steps : int
        Number of model/estimator steps that were excluded.
    n_features : int
        Number of input features.
    n_samples : dict of str to int
        Number of samples per dataset split.
    steps : list of dict
        Details of each preprocessing step (step number, name, type).
    excluded : list of dict
        Details of each excluded model step (name, type).
    """

    pipeline_type: str
    total_steps: int
    n_preprocessing_steps: int
    n_excluded_steps: int
    n_features: int
    n_samples: Dict[str, int]
    steps: List[Dict[str, Any]]
    excluded: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Return summary as a plain dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self) -> str:  # noqa: D105
        lines = [
            "=" * 60,
            "Preprocessing Inspector Summary",
            "=" * 60,
            f"  Pipeline:             {self.pipeline_type}",
            f"  Total steps:          {self.total_steps}",
            f"  Preprocessing steps:  {self.n_preprocessing_steps}",
            f"  Excluded steps:       {self.n_excluded_steps}",
            f"  Input features:       {self.n_features}",
            f"  Datasets:             {self.n_samples}",
            "-" * 60,
            "  Preprocessing steps (visualised):",
        ]
        for s in self.steps:
            lines.append(f"    {s['step']}. {s['name']} ({s['type']})")
        if self.excluded:
            lines.append("  Excluded (model) steps:")
            for s in self.excluded:
                lines.append(f"    - {s['name']} ({s['type']})")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ClassificationMetrics:
    """Metrics for a single classification dataset split."""

    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(kw_only=True)
class ClassificationSummary:
    """Summary for classification models.

    Attributes
    ----------
    model_type : str
        Class name of the classifier.
    has_preprocessing : bool
        Whether the model includes preprocessing steps.
    n_features : int
        Number of input features.
    n_samples : dict of str to int
        Number of samples per dataset split.
    n_classes : int
        Number of classes.
    class_names : list of str
        Display names for each class.
    has_predict_proba : bool
        Whether the classifier supports ``predict_proba``.
    preprocessing_steps : list of dict
        Details of each preprocessing step.
    train : ClassificationMetrics
        Metrics on the training set.
    test : ClassificationMetrics, optional
        Metrics on the test set.
    val : ClassificationMetrics, optional
        Metrics on the validation set.
    """

    model_type: str
    has_preprocessing: bool
    n_features: int
    n_samples: Dict[str, int]
    n_classes: int
    class_names: List[str]
    has_predict_proba: bool
    preprocessing_steps: List[Dict[str, Any]]
    train: ClassificationMetrics
    test: Optional[ClassificationMetrics] = None
    val: Optional[ClassificationMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return summary as a plain dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self) -> str:  # noqa: D105
        lines = [
            "=" * 60,
            "Classification Inspector Summary",
            "=" * 60,
            f"  Classifier:           {self.model_type}",
            f"  Preprocessing:        {self.has_preprocessing}",
            f"  Input features:       {self.n_features}",
            f"  Classes ({self.n_classes}):          {self.class_names}",
            f"  Predict proba:        {self.has_predict_proba}",
            f"  Datasets:             {self.n_samples}",
        ]

        if self.preprocessing_steps:
            lines.append("-" * 60)
            lines.append("  Preprocessing steps:")
            for s in self.preprocessing_steps:
                lines.append(f"    {s['step']}. {s['name']} ({s['type']})")

        lines.append("-" * 60)
        lines.append("  Metrics:")
        lines.append(
            f"  {'Dataset':<12} {'Accuracy':>10} {'Precision':>10} "
            f"{'Recall':>10} {'F1':>10}"
        )
        lines.append(f"  {'-' * 52}")

        for ds_name in ("train", "test", "val"):
            m = getattr(self, ds_name)
            if m is not None:
                lines.append(
                    f"  {ds_name:<12} {m.accuracy:>10.4f} {m.precision:>10.4f} "
                    f"{m.recall:>10.4f} {m.f1:>10.4f}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)
