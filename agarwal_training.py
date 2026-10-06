from pathlib import Path
from time import perf_counter

import hydra
import numpy as np

from baselines.agarwal import (
    fit_agarwal_reduction,
    make_cart_estimator,
    mixture_positive_probabilities,
    normalize_max_depth,
)
from baselines.data import prepare_baseline_splits
from baselines.metrics import (
    compute_agarwal_metrics,
    compute_ferm_metrics,
    infer_fscore_positive_label,
)
from data.init import BINARY_DATASETS


FAIRNESS_METRICS = {"DP", "EP", "EO"}


def validate_config(cfg):
    if cfg.dataset not in BINARY_DATASETS:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}.")
    if cfg.fairness.optimized_metric not in FAIRNESS_METRICS:
        raise ValueError(
            "fairness.optimized_metric must be DP, EP or EO."
        )
    if float(cfg.fairness.difference_bound) != 0.0:
        raise ValueError(
            "The experimental protocol fixes difference_bound to zero."
        )
    if int(cfg.fairness.favorable_label) != 1:
        raise ValueError("Agarwal requires favorable_label=1.")
    if cfg.training.sensible_variable not in {"GENDER", "OTHER"}:
        raise ValueError(
            "training.sensible_variable must be GENDER or OTHER."
        )
    if int(cfg.num_trials) <= 0:
        raise ValueError("num_trials must be strictly positive.")
    if not 0.0 < float(cfg.training.validation_fraction) < 1.0:
        raise ValueError(
            "training.validation_fraction must be between 0 and 1."
        )
    if float(cfg.reduction.eps) <= 0.0:
        raise ValueError("reduction.eps must be strictly positive.")
    if int(cfg.reduction.max_iter) <= 0:
        raise ValueError("reduction.max_iter must be positive.")
    if float(cfg.reduction.eta0) <= 0.0:
        raise ValueError("reduction.eta0 must be strictly positive.")
    if normalize_max_depth(cfg.model.max_depth) is not None:
        raise ValueError("The protocol fixes CART max_depth to None.")
    if cfg.model.criterion != "gini":
        raise ValueError("The protocol fixes CART criterion to gini.")
    if cfg.model.max_features != "sqrt":
        raise ValueError("The protocol fixes CART max_features to sqrt.")
    if int(cfg.model.min_samples_split) != 2:
        raise ValueError("The protocol fixes CART min_samples_split to 2.")
    if int(cfg.model.min_samples_leaf) != 1:
        raise ValueError("The protocol fixes CART min_samples_leaf to 1.")


def create_result_dir(cfg, repository_root):
    results_root = Path(str(cfg.results_root))
    if not results_root.is_absolute():
        results_root = repository_root / results_root

    return (
        results_root
        / str(cfg.dataset)
        / "Agarwal"
        / "fairness=agarwal"
        / "CART"
        / f"fairness_metric={cfg.fairness.optimized_metric}"
        / f"sensible_variable={cfg.training.sensible_variable}"
        / f"max_depth={cfg.model.max_depth}"
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
        values = np.asarray(values, dtype=float)
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(len(values)),
        }
    return aggregated


def empirical_constraint_violation(metrics, metric, split):
    suffix = f"{split}_af"
    if metric == "DP":
        return metrics[f"gibbs_DP_err_{suffix}"]
    if metric == "EP":
        return metrics[f"gibbs_EP_err_{suffix}"]
    return max(
        metrics[f"gibbs_EO_FPR_err_{suffix}"],
        metrics[f"gibbs_EO_TPR_err_{suffix}"],
    )


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
    estimator = make_cart_estimator(
        seed=seed,
        max_depth=cfg.model.max_depth,
        criterion=cfg.model.criterion,
        max_features=cfg.model.max_features,
        min_samples_split=cfg.model.min_samples_split,
        min_samples_leaf=cfg.model.min_samples_leaf,
    )

    start = perf_counter()
    reduction = fit_agarwal_reduction(
        estimator=estimator,
        X=splits["train"]["X"],
        y=splits["train"]["y"],
        sensitive=splits["train"]["s"],
        metric=cfg.fairness.optimized_metric,
        difference_bound=cfg.fairness.difference_bound,
        eps=cfg.reduction.eps,
        max_iter=cfg.reduction.max_iter,
        eta0=cfg.reduction.eta0,
        run_linprog_step=cfg.reduction.run_linprog_step,
    )
    elapsed = perf_counter() - start

    mixture_weights = np.asarray(
        reduction.weights_,
        dtype=float,
    ).reshape(-1)
    seed_results = {
        "seed": int(seed),
        "oracle": "cart",
        "optimized_metric": str(cfg.fairness.optimized_metric),
        "difference_bound": float(cfg.fairness.difference_bound),
        "reduction_eps": float(cfg.reduction.eps),
        "time": float(elapsed),
        "best_gap": float(reduction.best_gap_),
        "best_iter": int(reduction.best_iter_),
        "last_iter": int(reduction.last_iter_),
        "n_predictors": int(len(reduction.predictors_)),
        "n_active_predictors": int(
            np.sum(mixture_weights > 1e-12)
        ),
        "mixture_weights": mixture_weights,
        "fscore_positive_label": int(fscore_positive_label),
    }

    split_codes = {"train": "tr", "val": "va", "test": "te"}
    for split_name, split_code in split_codes.items():
        split = splits[split_name]
        probabilities, normalized_weights = (
            mixture_positive_probabilities(reduction, split["X"])
        )
        seed_results["mixture_weights"] = normalized_weights

        expected_metrics = compute_agarwal_metrics(
            y_true=split["y"],
            positive_probabilities=probabilities,
            sensitive=split["s"],
            split=split_code,
            fscore_positive_label=fscore_positive_label,
        )
        seed_results.update(expected_metrics)
        seed_results[f"constraint_violation_{split_code}"] = (
            empirical_constraint_violation(
                expected_metrics,
                cfg.fairness.optimized_metric,
                split_code,
            )
        )

        deterministic_predictions = np.where(
            probabilities >= 0.5,
            1,
            -1,
        )
        seed_results.update(
            compute_ferm_metrics(
                y_true=split["y"],
                y_pred=deterministic_predictions,
                sensitive=split["s"],
                split=split_code,
                fscore_positive_label=fscore_positive_label,
            )
        )

    return seed_results


@hydra.main(config_path="agarwal_config.yaml")
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
            print(
                "Training Agarwal CART, "
                f"{cfg.fairness.optimized_metric}, seed {seed}."
            )
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
            metrics["gibbs_OA_err_tr_af"]["mean"],
            metrics["gibbs_OA_err_tr_af"]["std"],
        ),
        "validation-error": (
            metrics["gibbs_OA_err_va_af"]["mean"],
            metrics["gibbs_OA_err_va_af"]["std"],
        ),
        "test-error": (
            metrics["gibbs_OA_err_te_af"]["mean"],
            metrics["gibbs_OA_err_te_af"]["std"],
        ),
        "metrics": metrics,
        "seed_results": all_seed_results,
    }
    np.save(result_dir / "err-b.npy", results)
    print(f"Results saved in {result_dir}.")


if __name__ == "__main__":
    main()
