#!/usr/bin/env python3
"""
Verify that local raw datasets are sufficient for offline loading.

Run after transferring data/raw to the cluster.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.init import BINARY_DATASETS, Dataset


def check_one(dataset, sensible_variable, data_root, normalize):
    data = Dataset(
        dataset=dataset,
        sensible_variable=sensible_variable,
        pred="RandomForests",
        data_path=data_root,
        normalize=normalize,
        seed=541944,
        offline=True,
    )

    s_train = np.asarray(data.s["train"])
    y_train = np.asarray(data.y["train"])
    s_unique = np.unique(s_train)
    y_unique = np.unique(y_train)

    print(f"{dataset:7s} | {sensible_variable:6s} | "
          f"n={len(data.X['train'])} | s={s_unique} | y={y_unique}")

    if not np.all(np.isin(s_unique, [0, 1])):
        raise ValueError(f"{dataset}/{sensible_variable}: sensitive attribute is not in {{0,1}}")
    if not np.all(np.isin(y_unique, [-1, 1])):
        raise ValueError(f"{dataset}/{sensible_variable}: labels are not in {{-1,1}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--normalize", action="store_true", help="Enable normalize=True checks.")
    args = parser.parse_args()

    datasets = BINARY_DATASETS
    sens = ["GENDER", "OTHER"]

    print(f"Checking offline loading from: {args.data_root.resolve()}")
    for ds in datasets:
        for sv in sens:
            check_one(ds, sv, args.data_root, args.normalize)

    print("\nOffline verification OK.")


if __name__ == "__main__":
    main()
