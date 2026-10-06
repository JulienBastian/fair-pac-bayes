import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LinearFERMClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Fair Empirical Risk Minimization classifier.

    The classifier learns a representation whose conditional feature means
    coincide across the two sensitive groups for the favorable class. A
    linear SVM is then fitted on this representation.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter of the linear SVM.
    sensitive_feature : int, default=0
        Column of X containing the binary sensitive attribute.
    """

    def __init__(
        self,
        C=1.0,
        sensitive_feature=0,
    ):
        self.C = C
        self.sensitive_feature = sensitive_feature

    def _validate_parameters(self, n_features):
        if self.C <= 0:
            raise ValueError(f"C must be strictly positive, got {self.C}.")
        if not isinstance(self.sensitive_feature, (int, np.integer)):
            raise TypeError("sensitive_feature must be an integer column index.")
        if not 0 <= self.sensitive_feature < n_features:
            raise ValueError(
                f"sensitive_feature={self.sensitive_feature} is invalid for "
                f"X with {n_features} features."
            )

    def _fair_representation(self, X):
        pivot_values = X[:, self.pivot_] / self.u_[self.pivot_]
        transformed = X - np.outer(pivot_values, self.u_)
        return np.delete(transformed, self.pivot_, axis=1)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self._validate_parameters(X.shape[1])

        sensitive_values = X[:, self.sensitive_feature]
        groups = np.unique(sensitive_values)
        if len(groups) != 2:
            raise ValueError(
                "FERM requires exactly two sensitive groups; "
                f"got {groups.tolist()}."
            )

        self.groups_ = np.sort(groups)
        positive_mask = y == 1
        group_0_positive = positive_mask & (
            sensitive_values == self.groups_[0]
        )
        group_1_positive = positive_mask & (
            sensitive_values == self.groups_[1]
        )

        if not np.any(group_0_positive):
            raise ValueError(
                "The first sensitive group contains no favorable example."
            )
        if not np.any(group_1_positive):
            raise ValueError(
                "The second sensitive group contains no favorable example."
            )

        mean_0 = X[group_0_positive].mean(axis=0)
        mean_1 = X[group_1_positive].mean(axis=0)
        self.u_ = -(mean_1 - mean_0)
        self.u_norm_ = float(np.linalg.norm(self.u_))

        self.pivot_ = int(np.argmax(self.u_))
        if self.u_[self.pivot_] == 0:
            raise ValueError(
                "FERM selected a zero-valued pivot and cannot construct "
                "the fair representation."
            )

        transformed = self._fair_representation(X)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = transformed.shape[1]

        self.classifier_ = SVC(
            C=self.C,
            kernel="linear",
        )
        self.classifier_.fit(transformed, y)

        self.classes_ = self.classifier_.classes_
        self.n_iter_ = int(np.max(np.atleast_1d(self.classifier_.n_iter_)))
        return self

    def transform(self, X):
        check_is_fitted(self, ["classifier_", "u_", "pivot_"])
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected "
                f"{self.n_features_in_}."
            )
        return self._fair_representation(X)

    def decision_function(self, X):
        check_is_fitted(self, "classifier_")
        return self.classifier_.decision_function(self.transform(X))

    def predict(self, X):
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(self.transform(X))

    def score(self, X, y):
        check_is_fitted(self, "classifier_")
        return self.classifier_.score(self.transform(X), y)
