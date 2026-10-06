import numpy as np
from sklearn.metrics import f1_score


def _as_vector(values, name):
    vector = np.asarray(values).reshape(-1)
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty.")
    return vector


def _rate(event, condition, name):
    if not np.any(condition):
        raise ValueError(f"Cannot compute {name}: empty subgroup.")
    return float(np.mean(event[condition]))


def infer_fscore_positive_label(y_train):
    """Return the least represented label in the train split."""
    y_train = _as_vector(y_train, "y_train")
    if not np.array_equal(np.unique(y_train), np.array([-1, 1])):
        raise ValueError("y_train must contain labels {-1, +1}.")

    count_lz = int(np.sum(y_train == -1))
    count_lo = int(np.sum(y_train == 1))
    return -1 if count_lz < count_lo else 1


def compute_ferm_metrics(
    y_true,
    y_pred,
    sensitive,
    split,
    fscore_positive_label,
):
    """
    Compute deterministic empirical metrics for one FERM split.

    F-score uses the minority train label supplied by the caller.
    Fairness metrics always use label +1 as the favorable outcome.
    """
    y_true = _as_vector(y_true, "y_true")
    y_pred = _as_vector(y_pred, "y_pred")
    sensitive = _as_vector(sensitive, "sensitive")

    if not len(y_true) == len(y_pred) == len(sensitive):
        raise ValueError(
            "y_true, y_pred and sensitive must have identical lengths."
        )
    if split not in {"tr", "va", "te"}:
        raise ValueError(f"split must be tr, va or te; got {split}.")
    if fscore_positive_label not in {-1, 1}:
        raise ValueError("fscore_positive_label must be -1 or +1.")
    if not np.array_equal(np.unique(y_true), np.array([-1, 1])):
        raise ValueError("y_true must contain labels {-1, +1}.")
    if not np.all(np.isin(np.unique(y_pred), [-1, 1])):
        raise ValueError("y_pred must only contain labels -1 and +1.")

    groups = np.unique(sensitive)
    if len(groups) != 2:
        raise ValueError(
            "Fairness metrics require exactly two sensitive groups."
        )

    group_A = sensitive == groups[0]
    group_B = sensitive == groups[1]
    label_lz = y_true == -1
    label_lo = y_true == 1
    predicted_positive = y_pred == 1
    errors = y_pred != y_true

    overall_error = float(np.mean(errors))
    error_lz = _rate(errors, label_lz, "label -1 error")
    error_lo = _rate(errors, label_lo, "label +1 error")

    dp_error = abs(
        _rate(predicted_positive, group_A, "group A positive rate")
        - _rate(predicted_positive, group_B, "group B positive rate")
    )

    tpr_A = _rate(
        predicted_positive, group_A & label_lo, "group A TPR"
    )
    tpr_B = _rate(
        predicted_positive, group_B & label_lo, "group B TPR"
    )
    ep_error = abs(tpr_A - tpr_B)

    fpr_A = _rate(
        predicted_positive, group_A & label_lz, "group A FPR"
    )
    fpr_B = _rate(
        predicted_positive, group_B & label_lz, "group B FPR"
    )
    positive_proportion = float(np.mean(label_lo))
    eo_error = (
        (1.0 - positive_proportion) * abs(fpr_A - fpr_B)
        + positive_proportion * ep_error
    )

    suffix = f"{split}_af"
    return {
        f"det_OA_err_{suffix}": overall_error,
        f"det_OA_err_{suffix}_lz": error_lz,
        f"det_OA_err_{suffix}_lo": error_lo,
        f"det_OA_fscore_{suffix}": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=fscore_positive_label,
                zero_division=0,
            )
        ),
        f"det_DP_err_{suffix}": dp_error,
        f"det_EO_err_{suffix}": eo_error,
        f"det_EP_err_{suffix}": ep_error,
        "fscore_positive_label": int(fscore_positive_label),
    }


def compute_agarwal_metrics(
    y_true,
    positive_probabilities,
    sensitive,
    split,
    fscore_positive_label,
):
    """Compute empirical metrics for an Agarwal randomized classifier."""
    y_true = _as_vector(y_true, "y_true")
    positive_probabilities = _as_vector(
        positive_probabilities,
        "positive_probabilities",
    ).astype(float)
    sensitive = _as_vector(sensitive, "sensitive")

    if not len(y_true) == len(positive_probabilities) == len(sensitive):
        raise ValueError(
            "y_true, positive_probabilities and sensitive must have "
            "identical lengths."
        )
    if split not in {"tr", "va", "te"}:
        raise ValueError(f"split must be tr, va or te; got {split}.")
    if fscore_positive_label not in {-1, 1}:
        raise ValueError("fscore_positive_label must be -1 or +1.")
    if not np.array_equal(np.unique(y_true), np.array([-1, 1])):
        raise ValueError("y_true must contain labels {-1, +1}.")
    if (
        not np.all(np.isfinite(positive_probabilities))
        or np.any(positive_probabilities < -1e-12)
        or np.any(positive_probabilities > 1.0 + 1e-12)
    ):
        raise ValueError(
            "positive_probabilities must contain finite values in [0, 1]."
        )

    positive_probabilities = np.clip(
        positive_probabilities,
        0.0,
        1.0,
    )
    groups = np.unique(sensitive)
    if len(groups) != 2:
        raise ValueError(
            "Fairness metrics require exactly two sensitive groups."
        )

    group_A = sensitive == groups[0]
    group_B = sensitive == groups[1]
    label_lz = y_true == -1
    label_lo = y_true == 1

    expected_errors = np.where(
        label_lo,
        1.0 - positive_probabilities,
        positive_probabilities,
    )
    overall_error = float(np.mean(expected_errors))
    error_lz = _rate(expected_errors, label_lz, "label -1 error")
    error_lo = _rate(expected_errors, label_lo, "label +1 error")

    dp_error = abs(
        _rate(
            positive_probabilities,
            group_A,
            "group A positive rate",
        )
        - _rate(
            positive_probabilities,
            group_B,
            "group B positive rate",
        )
    )

    tpr_A = _rate(
        positive_probabilities,
        group_A & label_lo,
        "group A TPR",
    )
    tpr_B = _rate(
        positive_probabilities,
        group_B & label_lo,
        "group B TPR",
    )
    ep_error = abs(tpr_A - tpr_B)

    fpr_A = _rate(
        positive_probabilities,
        group_A & label_lz,
        "group A FPR",
    )
    fpr_B = _rate(
        positive_probabilities,
        group_B & label_lz,
        "group B FPR",
    )
    fpr_error = abs(fpr_A - fpr_B)
    positive_proportion = float(np.mean(label_lo))
    eo_error = (
        (1.0 - positive_proportion) * fpr_error
        + positive_proportion * ep_error
    )

    probability_fscore_positive = (
        positive_probabilities
        if fscore_positive_label == 1
        else 1.0 - positive_probabilities
    )
    fscore_label = y_true == fscore_positive_label
    true_positive = float(
        np.sum(probability_fscore_positive[fscore_label])
    )
    false_positive = float(
        np.sum(probability_fscore_positive[~fscore_label])
    )
    false_negative = float(
        np.sum(1.0 - probability_fscore_positive[fscore_label])
    )
    denominator = (
        2.0 * true_positive + false_positive + false_negative
    )
    fscore = (
        0.0
        if denominator == 0.0
        else 2.0 * true_positive / denominator
    )

    suffix = f"{split}_af"
    return {
        f"gibbs_OA_err_{suffix}": overall_error,
        f"gibbs_OA_err_{suffix}_lz": error_lz,
        f"gibbs_OA_err_{suffix}_lo": error_lo,
        f"gibbs_OA_fscore_{suffix}": float(fscore),
        f"gibbs_DP_err_{suffix}": dp_error,
        f"gibbs_EO_err_{suffix}": eo_error,
        f"gibbs_EP_err_{suffix}": ep_error,
        f"gibbs_EO_FPR_err_{suffix}": fpr_error,
        f"gibbs_EO_TPR_err_{suffix}": ep_error,
        "fscore_positive_label": int(fscore_positive_label),
    }
