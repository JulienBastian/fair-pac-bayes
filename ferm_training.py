from pathlib import Path
from time import perf_counter

import hydra
import numpy as np

from baselines import (
    LinearFERMClassifier,
    compute_ferm_metrics,
    infer_fscore_positive_label,
)
from baselines.data import prepare_baseline_splits
from data.init import BINARY_DATASETS


def validate_config(cfg):
    if cfg.dataset not in BINARY_DATASETS:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}.")
    if cfg.training.sensible_variable not in {"GENDER", "OTHER"}:
        raise ValueError(
            "training.sensible_variable must be GENDER or OTHER."
        )
    if int(cfg.num_trials) <= 0:
        raise ValueError("num_trials must be strictly positive.")
    if float(cfg.model.C) <= 0:
        raise ValueError("model.C must be strictly positive.")
    if not 0.0 < float(cfg.training.validation_fraction) < 1.0:
        raise ValueError(
            "training.validation_fraction must be between 0 and 1."
        )
    if cfg.fairness.optimized_metric != "EP":
        raise ValueError("Linear FERM only optimizes EP.")
    if int(cfg.fairness.favorable_label) != 1:
        raise ValueError("Linear FERM requires favorable_label=1.")


def create_result_dir(cfg, repository_root):
    results_root = Path(str(cfg.results_root))
    if not results_root.is_absolute():
        results_root = repository_root / results_root

    return (
        results_root
        / str(cfg.dataset)
        / "FERM"
        / "fairness=ferm"
        / "LinearFERM"
        / "fairness_metric=EP"
        / f"sensible_variable={cfg.training.sensible_variable}"
        / f"C={cfg.model.C}"
    )


def aggregate_seed_results(all_seed_results):
    numeric_values = {}

    for seed_results in all_seed_results:
        for key, value in seed_results.items():
            if isinstance(value, (bool, np.bool_)):
                continue
            try:
                scalar = float(np.asarray(value).item())
            except (TypeError, ValueError):
                continue
            numeric_values.setdefault(key, []).append(scalar)

    aggregated = {}
    for key, values in numeric_values.items():
        array = np.asarray(values, dtype=float)
        aggregated[key] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "n": int(len(array)),
        }

    return aggregated


def train_one_seed(cfg, repository_root, seed):
    splits = prepare_baseline_splits(
        dataset=cfg.dataset,
        sensible_variable=cfg.training.sensible_variable,
        data_path=repository_root / "data",
        normalize=cfg.training.normalize_data,
        seed=seed,
        validation_fraction=cfg.training.validation_fraction,
        offline=cfg.offline,
    )

    fscore_positive_label = infer_fscore_positive_label(
        splits["train"]["y"]
    )

    model = LinearFERMClassifier(
        C=float(cfg.model.C),
        sensitive_feature=0,
    )

    start = perf_counter()
    model.fit(
        splits["train"]["X"],
        splits["train"]["y"],
    )
    elapsed = perf_counter() - start

    seed_results = {
        "seed": int(seed),
        "C": float(cfg.model.C),
        "time": float(elapsed),
        "pivot": int(model.pivot_),
        "u_norm": float(model.u_norm_),
        "n_iter": int(model.n_iter_),
        "fscore_positive_label": int(fscore_positive_label),
    }

    split_codes = {
        "train": "tr",
        "val": "va",
        "test": "te",
    }
    for split_name, split_code in split_codes.items():
        split = splits[split_name]
        predictions = model.predict(split["X"])
        seed_results.update(
            compute_ferm_metrics(
                y_true=split["y"],
                y_pred=predictions,
                sensitive=split["s"],
                split=split_code,
                fscore_positive_label=fscore_positive_label,
            )
        )

    return seed_results


@hydra.main(config_path="ferm_config.yaml")
def main(cfg):
    validate_config(cfg)

    repository_root = Path(hydra.utils.get_original_cwd())
    result_dir = create_result_dir(cfg, repository_root)
    result_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = []
    for trial in range(int(cfg.num_trials)):
        seed = int(cfg.training.seed) + trial
        seed_dir = result_dir / f"seed={seed}"
        seed_path = seed_dir / "err-b.npy"

        if seed_path.is_file():
            print(f"Seed {seed} already completed, loading results.")
            seed_results = np.load(
                seed_path,
                allow_pickle=True,
            ).item()
        else:
            print(f"Training FERM with seed {seed}.")
            seed_results = train_one_seed(
                cfg,
                repository_root,
                seed,
            )
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.save(seed_path, seed_results)

        all_seed_results.append(seed_results)

    metrics = aggregate_seed_results(all_seed_results)
    results = {
        "train-error": (
            metrics["det_OA_err_tr_af"]["mean"],
            metrics["det_OA_err_tr_af"]["std"],
        ),
        "validation-error": (
            metrics["det_OA_err_va_af"]["mean"],
            metrics["det_OA_err_va_af"]["std"],
        ),
        "test-error": (
            metrics["det_OA_err_te_af"]["mean"],
            metrics["det_OA_err_te_af"]["std"],
        ),
        "metrics": metrics,
    }
    np.save(result_dir / "err-b.npy", results)

    print(f"Results saved in {result_dir}.")


if __name__ == "__main__":
    main()
