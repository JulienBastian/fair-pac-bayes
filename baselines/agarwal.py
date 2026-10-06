import numpy as np
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    ErrorRate,
    ExponentiatedGradient,
    TruePositiveRateParity,
)
from sklearn.tree import DecisionTreeClassifier


CONSTRAINTS = {
    "DP": DemographicParity,
    "EP": TruePositiveRateParity,
    "EO": EqualizedOdds,
}


def encode_fairlearn_labels(y):
    y = np.asarray(y).reshape(-1)
    if not np.array_equal(np.unique(y), np.array([-1, 1])):
        raise ValueError("Agarwal expects original labels {-1, +1}.")
    return (y == 1).astype(int)


def normalize_max_depth(max_depth):
    if max_depth in {None, "None", "none", "null"}:
        return None
    max_depth = int(max_depth)
    if max_depth <= 0:
        raise ValueError("max_depth must be positive or None.")
    return max_depth


def make_cart_estimator(
    seed,
    max_depth=None,
    criterion="gini",
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
):
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=normalize_max_depth(max_depth),
        max_features=max_features,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        class_weight=None,
        random_state=int(seed),
    )


def make_constraint(metric, difference_bound=0.0):
    if metric not in CONSTRAINTS:
        raise ValueError(
            f"metric must be one of {sorted(CONSTRAINTS)}, got {metric}."
        )
    return CONSTRAINTS[metric](
        difference_bound=float(difference_bound)
    )


def fit_agarwal_reduction(
    estimator,
    X,
    y,
    sensitive,
    metric,
    difference_bound,
    eps,
    max_iter,
    eta0,
    run_linprog_step,
):
    reduction = ExponentiatedGradient(
        estimator=estimator,
        constraints=make_constraint(metric, difference_bound),
        objective=ErrorRate(),
        eps=float(eps),
        max_iter=int(max_iter),
        eta0=float(eta0),
        run_linprog_step=bool(run_linprog_step),
    )
    reduction.fit(
        X,
        encode_fairlearn_labels(y),
        sensitive_features=np.asarray(sensitive).reshape(-1),
    )
    return reduction


def mixture_positive_probabilities(reduction, X):
    weights = np.asarray(reduction.weights_, dtype=float).reshape(-1)
    predictors = list(reduction.predictors_)
    if len(weights) != len(predictors) or len(weights) == 0:
        raise ValueError("Invalid Agarwal mixture.")
    if not np.all(np.isfinite(weights)) or np.any(weights < -1e-12):
        raise ValueError("Agarwal mixture weights must be non-negative.")

    weights = np.clip(weights, 0.0, None)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Agarwal mixture has zero total weight.")
    weights /= total_weight

    predictions = []
    for predictor in predictors:
        prediction = np.asarray(predictor.predict(X)).reshape(-1)
        if np.all(np.isin(prediction, [0, 1])):
            prediction = prediction.astype(float)
        elif np.all(np.isin(prediction, [-1, 1])):
            prediction = (prediction == 1).astype(float)
        else:
            raise ValueError(
                "Agarwal predictors must return binary labels."
            )
        predictions.append(prediction)

    prediction_matrix = np.column_stack(predictions)
    return prediction_matrix @ weights, weights
