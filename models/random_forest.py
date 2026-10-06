import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier


def normalize_max_depth(max_depth):
    if max_depth in {None, "None"}:
        return None
    if not isinstance(max_depth, int) or max_depth <= 0:
        raise ValueError(
            "max_depth must be a positive integer or None, "
            f"got {max_depth}."
        )
    return max_depth


def normalize_max_features(max_features):
    if max_features in {None, "None"}:
        return None
    return max_features


def resolve_class_weight(class_weight, auto_value):
    if class_weight is None:
        return None

    normalized = str(class_weight).lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        return auto_value
    if normalized in {"balanced", "balanced_subsample"}:
        return normalized

    raise ValueError(
        "class_weight must be auto, none, balanced or "
        f"balanced_subsample, got {class_weight}."
    )


def validate_tree_parameters(
        n,
        criterion,
        min_samples_split,
        min_samples_leaf):
    if not isinstance(n, int) or n <= 0:
        raise ValueError(
            f"n must be a positive integer, got {n}."
        )
    if criterion not in {"gini", "entropy", "log_loss"}:
        raise ValueError(
            "criterion must be gini, entropy or log_loss, "
            f"got {criterion}."
        )
    if (
        not isinstance(min_samples_split, int)
        or min_samples_split < 2
    ):
        raise ValueError(
            "min_samples_split must be an integer >= 2, "
            f"got {min_samples_split}."
        )
    if (
        not isinstance(min_samples_leaf, int)
        or min_samples_leaf <= 0
    ):
        raise ValueError(
            "min_samples_leaf must be a positive integer, "
            f"got {min_samples_leaf}."
        )


def random_forest(
        n,
        X,
        y,
        samples_prop,
        max_depth,
        criterion,
        class_weight,
        max_features,
        min_samples_split,
        min_samples_leaf,
        binary,
        seed):
    """
    Create a collection of two random forests classifiers.

    n (int): number of tree per forest classifier.
    X, y: features and labels to learn on.
    max_samples, max_depth (ints): sklearn.ensemble.RandomForestClassifier argument.
    binary (bool): whether of not the dataset is binary.

    Returns both the forests in a tuple, and the total number of base classifiers (trees, 2 * n).
    """
    validate_tree_parameters(
        n,
        criterion,
        min_samples_split,
        min_samples_leaf,
    )
    if not 0 < samples_prop <= 1:
        raise ValueError(
            "samples_prop must lie in (0, 1], "
            f"got {samples_prop}."
        )

    trees = trained_random_forest(
        n,
        (X, y),
        samples_prop=samples_prop,
        max_depth=normalize_max_depth(max_depth),
        criterion=criterion,
        class_weight=resolve_class_weight(
            class_weight,
            auto_value="balanced_subsample",
        ),
        max_features=normalize_max_features(max_features),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )

    predictors = lambda x: trees_predict(x, trees, binary_dataset=binary)

    return predictors, len(trees)


def trained_random_forest(
        n,
        data,
        samples_prop=.75,
        max_depth=None,
        criterion="gini",
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None):
    """
    Create a trained sklearn.ensemble.RandomForestClassifier instance.
    max_samples (float): proportion of data used to train each tree.
    """
    bootstrap = True
    if samples_prop == 1.:
        bootstrap = False
        samples_prop = None

    forest = RandomForestClassifier(
        n_estimators=n,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        max_samples=samples_prop,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=random_state,
    )
    forest.fit(*data)

    return forest.estimators_


def trees_predict(x, trees, binary_dataset=True):
    """
    Returns the prediction of each tree in the forest.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    pred = np.column_stack([
        tree.predict(x) for tree in trees
    ])

    if binary_dataset:
        pred = np.where(pred > 0, 1.0, -1.0)

    return torch.as_tensor(pred, dtype=torch.float32)
