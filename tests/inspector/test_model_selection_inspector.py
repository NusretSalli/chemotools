"""Tests for ModelSelectionInspector and from_candidate_selector class methods."""

import matplotlib
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chemotools.inspector import (  # noqa: E402
    ClassificationInspector,
    ModelSelectionInspector,
    PCAInspector,
    PLSRegressionInspector,
)
from chemotools.inspector.core.summaries import ModelSelectionSummary  # noqa: E402
from chemotools.model_selection import CandidateSelector  # noqa: E402


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def regression_data():
    """Generate simple regression data with train/test split."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.7]) + rng.normal(scale=0.1, size=80)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.fixture
def pls_selector(regression_data):
    """Return a fitted CandidateSelector wrapping PLSRegression."""
    X_train, y_train, _, _ = regression_data
    selector = CandidateSelector(
        estimator=PLSRegression(),
        param_grid={"n_components": [1, 2, 3]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=1,
    )
    selector.fit(X_train, y_train)
    return selector


@pytest.fixture
def pca_selector(regression_data):
    """Return a fitted CandidateSelector wrapping PCA."""
    X_train, _, _, _ = regression_data
    selector = CandidateSelector(
        estimator=PCA(),
        param_grid={"n_components": [1, 2, 3]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=1,
    )
    # PCA doesn't use y, but GridSearchCV needs a target for
    # neg_root_mean_squared_error scoring.  Use a Ridge pipeline instead.
    # We'll use a simpler approach: fit PCA with explained_variance scoring.
    selector_pca = CandidateSelector(
        estimator=PCA(),
        param_grid={"n_components": [1, 2, 3]},
        cv=3,
        n_jobs=1,
    )
    selector_pca.fit(X_train)
    return selector_pca


@pytest.fixture
def ridge_selector(regression_data):
    """Return a fitted CandidateSelector wrapping Ridge for RMSE-based tests."""
    X_train, y_train, _, _ = regression_data
    selector = CandidateSelector(
        estimator=Ridge(random_state=0),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=1,
    )
    selector.fit(X_train, y_train)
    return selector


# ==============================================================================
# ModelSelectionInspector tests
# ==============================================================================


class TestModelSelectionInspectorInit:
    """Test ModelSelectionInspector initialization."""

    def test_init_with_fitted_selector(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        assert inspector.selector is ridge_selector
        assert len(inspector.candidates) == 3

    def test_init_with_unfitted_selector_raises(self):
        unfitted = CandidateSelector(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
        )
        with pytest.raises(Exception):
            ModelSelectionInspector(unfitted)


class TestModelSelectionInspectorSummary:
    """Test summary method."""

    def test_summary_returns_dataclass(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        summary = inspector.summary()

        assert isinstance(summary, ModelSelectionSummary)
        assert summary.estimator_type == "Ridge"
        assert summary.scoring == "neg_root_mean_squared_error"
        assert summary.n_candidates == 3
        assert summary.cv_folds == 3
        assert isinstance(summary.best_score, float)
        assert isinstance(summary.best_params, dict)

    def test_summary_repr(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        summary = inspector.summary()
        text = repr(summary)

        assert "Model Selection Inspector Summary" in text
        assert "Ridge" in text

    def test_summary_to_dict(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        summary = inspector.summary()
        d = summary.to_dict()

        assert isinstance(d, dict)
        assert "estimator_type" in d
        assert "best_score" in d


class TestModelSelectionInspectorPlots:
    """Test plotting methods."""

    def test_inspect_returns_figures(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        figs = inspector.inspect()

        assert isinstance(figs, dict)
        assert "cv_metrics" in figs
        assert "score_vs_variance" in figs
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)

    def test_plot_cv_metrics(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        ax = inspector.plot_cv_metrics()

        assert ax is not None
        assert ax.get_xlabel() == "RMSECV"
        assert ax.get_ylabel() == "RMSECV / RMSEC"

    def test_plot_score_vs_variance(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        ax = inspector.plot_score_vs_variance()

        assert ax is not None
        assert ax.get_xlabel() == "Variance"
        assert ax.get_ylabel() == "Mean Test Score"

    def test_plot_cv_metrics_with_custom_ax(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        fig, ax = plt.subplots()
        result = inspector.plot_cv_metrics(ax=ax)
        assert result is ax

    def test_close_figures(self, ridge_selector):
        inspector = ModelSelectionInspector(ridge_selector)
        inspector.inspect()
        assert len(inspector._tracked_figures) > 0
        inspector.close_figures()
        assert len(inspector._tracked_figures) == 0


# ==============================================================================
# CandidateSelector deprecated plot methods
# ==============================================================================


class TestCandidateSelectorDeprecatedPlots:
    """Test that old plotting methods still work but emit FutureWarning."""

    def test_plot_cv_metrics_deprecation(self, ridge_selector):
        with pytest.warns(FutureWarning, match="deprecated"):
            ax = ridge_selector.plot_cv_metrics()
        assert ax is not None

    def test_plot_score_vs_variance_deprecation(self, ridge_selector):
        with pytest.warns(FutureWarning, match="deprecated"):
            ax = ridge_selector.plot_score_vs_variance()
        assert ax is not None


# ==============================================================================
# PLSRegressionInspector.from_candidate_selector
# ==============================================================================


class TestPLSRegressionInspectorFromCandidateSelector:
    """Test the from_candidate_selector class method on PLSRegressionInspector."""

    def test_basic_creation(self, pls_selector, regression_data):
        X_train, y_train, X_test, y_test = regression_data

        inspector = PLSRegressionInspector.from_candidate_selector(
            pls_selector,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        assert isinstance(inspector, PLSRegressionInspector)
        assert isinstance(inspector.estimator, PLSRegression)
        assert inspector.n_samples["train"] == 60
        assert inspector.n_samples["test"] == 20

    def test_rank_selection(self, pls_selector, regression_data):
        X_train, y_train, _, _ = regression_data

        inspector_1 = PLSRegressionInspector.from_candidate_selector(
            pls_selector, X_train, y_train, rank=1
        )
        inspector_2 = PLSRegressionInspector.from_candidate_selector(
            pls_selector, X_train, y_train, rank=2
        )

        # Different ranks should generally produce different n_components
        # (though not guaranteed depending on grid search results)
        assert isinstance(inspector_1, PLSRegressionInspector)
        assert isinstance(inspector_2, PLSRegressionInspector)

    def test_invalid_rank_raises(self, pls_selector, regression_data):
        X_train, y_train, _, _ = regression_data
        with pytest.raises(ValueError, match="No candidate with rank"):
            PLSRegressionInspector.from_candidate_selector(
                pls_selector, X_train, y_train, rank=999
            )

    def test_with_x_axis(self, pls_selector, regression_data):
        X_train, y_train, _, _ = regression_data
        x_axis = np.arange(X_train.shape[1])

        inspector = PLSRegressionInspector.from_candidate_selector(
            pls_selector, X_train, y_train, x_axis=x_axis
        )

        np.testing.assert_array_equal(inspector.x_axis, x_axis)

    def test_summary_works(self, pls_selector, regression_data):
        X_train, y_train, _, _ = regression_data
        inspector = PLSRegressionInspector.from_candidate_selector(
            pls_selector, X_train, y_train
        )

        summary = inspector.summary()
        assert summary is not None

    def test_predictions_work(self, pls_selector, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        inspector = PLSRegressionInspector.from_candidate_selector(
            pls_selector, X_train, y_train, X_test=X_test, y_test=y_test
        )

        assert inspector.RMSE_train > 0
        assert inspector.RMSE_test > 0


# ==============================================================================
# PCAInspector.from_candidate_selector
# ==============================================================================


class TestPCAInspectorFromCandidateSelector:
    """Test the from_candidate_selector class method on PCAInspector."""

    def test_basic_creation(self, pca_selector, regression_data):
        X_train, y_train, X_test, _ = regression_data

        inspector = PCAInspector.from_candidate_selector(
            pca_selector,
            X_train,
            y_train=y_train,
            X_test=X_test,
        )

        assert isinstance(inspector, PCAInspector)
        assert isinstance(inspector.estimator, PCA)
        assert inspector.n_samples["train"] == 60
        assert inspector.n_samples["test"] == 20

    def test_rank_selection(self, pca_selector, regression_data):
        X_train, _, _, _ = regression_data

        inspector = PCAInspector.from_candidate_selector(
            pca_selector, X_train, rank=1
        )
        assert isinstance(inspector, PCAInspector)

    def test_summary_works(self, pca_selector, regression_data):
        X_train, _, _, _ = regression_data
        inspector = PCAInspector.from_candidate_selector(
            pca_selector, X_train
        )

        summary = inspector.summary()
        assert summary is not None
        assert inspector.n_components > 0

    def test_scores_work(self, pca_selector, regression_data):
        X_train, _, _, _ = regression_data
        inspector = PCAInspector.from_candidate_selector(
            pca_selector, X_train
        )

        scores = inspector.get_scores("train")
        assert scores.shape[0] == 60


# ==============================================================================
# Classification fixtures
# ==============================================================================


@pytest.fixture
def classification_data():
    """Generate simple binary classification data with train/test split."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.fixture
def svc_selector(classification_data):
    """Return a fitted CandidateSelector wrapping SVC with accuracy scoring."""
    X_train, y_train, _, _ = classification_data
    selector = CandidateSelector(
        estimator=SVC(probability=True),
        param_grid={"C": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=1,
    )
    selector.fit(X_train, y_train)
    return selector


# ==============================================================================
# ClassificationInspector.from_candidate_selector
# ==============================================================================


class TestClassificationInspectorFromCandidateSelector:
    """Test the from_candidate_selector class method on ClassificationInspector."""

    def test_basic_creation(self, svc_selector, classification_data):
        X_train, y_train, X_test, y_test = classification_data

        inspector = ClassificationInspector.from_candidate_selector(
            svc_selector,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
        )

        assert isinstance(inspector, ClassificationInspector)
        assert inspector.n_samples["train"] == 60
        assert inspector.n_samples["test"] == 20

    def test_rank_selection(self, svc_selector, classification_data):
        X_train, y_train, _, _ = classification_data

        inspector_1 = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train, rank=1
        )
        inspector_2 = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train, rank=2
        )

        assert isinstance(inspector_1, ClassificationInspector)
        assert isinstance(inspector_2, ClassificationInspector)

    def test_invalid_rank_raises(self, svc_selector, classification_data):
        X_train, y_train, _, _ = classification_data
        with pytest.raises(ValueError, match="No candidate with rank"):
            ClassificationInspector.from_candidate_selector(
                svc_selector, X_train, y_train, rank=999
            )

    def test_with_x_axis(self, svc_selector, classification_data):
        X_train, y_train, _, _ = classification_data
        x_axis = np.arange(X_train.shape[1])

        inspector = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train, x_axis=x_axis
        )

        np.testing.assert_array_equal(inspector.feature_names, x_axis)

    def test_summary_works(self, svc_selector, classification_data):
        X_train, y_train, _, _ = classification_data
        inspector = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train
        )

        summary = inspector.summary()
        assert summary is not None
        assert summary.n_classes == 2

    def test_metrics_work(self, svc_selector, classification_data):
        X_train, y_train, X_test, y_test = classification_data
        inspector = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train, X_test=X_test, y_test=y_test
        )

        train_acc = inspector.get_accuracy("train")
        test_acc = inspector.get_accuracy("test")
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1

    def test_has_predict_proba(self, svc_selector, classification_data):
        X_train, y_train, _, _ = classification_data
        inspector = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train
        )

        assert inspector.has_predict_proba is True

    def test_classes(self, svc_selector, classification_data):
        X_train, y_train, _, _ = classification_data
        inspector = ClassificationInspector.from_candidate_selector(
            svc_selector, X_train, y_train
        )

        assert set(inspector.classes) == {0, 1}


# ==============================================================================
# ModelSelectionInspector with classification scoring
# ==============================================================================


class TestModelSelectionInspectorClassification:
    """Test ModelSelectionInspector with non-RMSE (classification) scoring."""

    def test_inspect_returns_figures(self, svc_selector):
        inspector = ModelSelectionInspector(svc_selector)
        figs = inspector.inspect()

        assert isinstance(figs, dict)
        # cv_metrics should use generic test/train scores
        assert "cv_metrics" in figs
        assert "score_vs_variance" in figs
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)

    def test_plot_cv_metrics_uses_generic_labels(self, svc_selector):
        inspector = ModelSelectionInspector(svc_selector)
        ax = inspector.plot_cv_metrics()

        assert ax is not None
        assert ax.get_xlabel() == "Mean Test Score"
        assert ax.get_ylabel() == "Mean Train Score"

    def test_plot_score_vs_variance(self, svc_selector):
        inspector = ModelSelectionInspector(svc_selector)
        ax = inspector.plot_score_vs_variance()

        assert ax is not None
        assert ax.get_xlabel() == "Variance"
        assert ax.get_ylabel() == "Mean Test Score"

    def test_summary(self, svc_selector):
        inspector = ModelSelectionInspector(svc_selector)
        summary = inspector.summary()

        assert summary.estimator_type == "SVC"
        assert summary.scoring == "accuracy"
        assert summary.n_candidates == 3
