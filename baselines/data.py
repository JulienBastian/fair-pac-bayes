from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data.init import Dataset


def prepare_baseline_splits(
    dataset,
    sensible_variable,
    data_path,
    normalize,
    seed,
    validation_fraction=0.25,
    offline=None,
):
    """
    Build the train/validation/test splits used by the empirical baselines.

    The repository Dataset first creates the common 80/20 train/test split.
    This function preserves that test set and divides the existing 80% train
    set into 75% train and 25% validation, giving final 60/20/20 proportions.

    ``pred=None`` prevents the Random Forest-specific prior split in Dataset.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1, "
            f"got {validation_fraction}."
        )

    data = Dataset(
        dataset=dataset,
        sensible_variable=sensible_variable,
        pred=None,
        data_path=Path(data_path),
        normalize=normalize,
        seed=seed,
        offline=offline,
    )

    # Copy the original preprocessed features before any PAC-Bayes model can
    # replace data.X with weak-learner predictions.
    X_train_val = np.array(data.X["train"], copy=True)
    y_train_val = np.array(data.y["train"], copy=True)
    s_train_val = np.array(data.s["train"], copy=True)

    X_test = np.array(data.X["test"], copy=True)
    y_test = np.array(data.y["test"], copy=True)
    s_test = np.array(data.s["test"], copy=True)

    joint_labels = np.column_stack((y_train_val, s_train_val))
    joint_values, strata = np.unique(
        joint_labels,
        axis=0,
        return_inverse=True,
    )
    stratum_counts = np.bincount(strata)

    if len(joint_values) != 4:
        raise ValueError(
            "The train split must contain all four (label, sensitive group) "
            f"combinations; got {joint_values.tolist()}."
        )
    if np.any(stratum_counts < 2):
        raise ValueError(
            "Every (label, sensitive group) stratum must contain at least "
            f"two examples; got counts {stratum_counts.tolist()}."
        )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        s_train,
        s_val,
    ) = train_test_split(
        X_train_val,
        y_train_val,
        s_train_val,
        test_size=validation_fraction,
        random_state=seed,
        stratify=strata,
    )

    return {
        "train": {
            "X": X_train,
            "y": y_train,
            "s": s_train,
        },
        "val": {
            "X": X_val,
            "y": y_val,
            "s": s_val,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "s": s_test,
        },
    }
