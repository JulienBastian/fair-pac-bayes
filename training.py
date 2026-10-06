import hydra
from pathlib import Path
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.init import Dataset
from models.majority_vote import MajorityVote

from core.losses import initialize_risk
from core.deterministic_bounding import clip_weak_learners, compute_triple_bound, manual_coordinate_descent, \
    weights_rescaling, compute_fairness_metrics, compute_oneto_bound, evaluate
from core.bounds import BOUNDS
from core.monitors import MonitorMV
from core.utils import *
from core.optimization import triple_routine, oneto_routine


@hydra.main(config_path='base_config.yaml')
def main(cfg):
    # This function tests whether the hyperparameters are coherent with each others.
    whether_to_run_run(cfg)

    # For each run, there exists a unique repository.
    ROOT_DIR = create_root_dir(cfg)
    ROOT_DIR = Path(ROOT_DIR)
    print("Results will be saved in: ", ROOT_DIR.resolve(), ".")

    all_seed_results = []
    # We iterate for as many trials as required, incrementing the random seed by 1 for each one.
    for i in range(cfg.num_trials):
        current_seed = cfg.training.seed + i
        print(f"Current seed: {current_seed}.")

        # Each package has its random seed set according to the current seed.
        deterministic(current_seed)

        SAVE_DIR = ROOT_DIR / f"seed={current_seed}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if (SAVE_DIR / "err-b.npy").is_file():
            # If results are already saved in that folder, the run has already been done.
            print("Run already done, skipping...")

            # The aggregated results are later computed for the corresponding number of trials.
            seed_results = np.load(SAVE_DIR / "err-b.npy", allow_pickle=True).item()

        else:
            seed_results = {}

            # Initialize the dataset, loss utilized, etc.
            data = Dataset(cfg.dataset, cfg.training.sensible_variable, cfg.model.pred,
                           Path(hydra.utils.get_original_cwd()) / "data", cfg.training.normalize_data, current_seed)
            ms = {'train': torch.tensor(len(data.X['train'])),
                  'train_A': torch.tensor(len(data.X['train_A'])),
                  'train_B': torch.tensor(len(data.X['train_B'])),
                  'train_A0': torch.tensor(len(data.X['train_A0'])),
                  'train_B0': torch.tensor(len(data.X['train_B0'])),
                  'train_A1': torch.tensor(len(data.X['train_A1'])),
                  'train_B1': torch.tensor(len(data.X['train_B1'])),
                  'train_0': torch.tensor(sum(data.y['train'] == -1)),
                  'train_1': torch.tensor(sum(data.y['train'] ==  1)),
                  'test': torch.tensor(len(data.X['test'])),
                  'test_0': torch.tensor(sum(data.y['test'] == -1)),
                  'test_1': torch.tensor(sum(data.y['test'] ==  1))}

            # Define params for each method.
            n_classes = get_n_classes(cfg.dataset)
            loss, bound_coeff, kl_factor = initialize_risk(cfg, n_classes)
            distribution_name = cfg.training.distribution

            # The main bound to optimize, and the surrogate bound to test (if risk == Triple).
            bound = lambda _, modl, risk: BOUNDS[cfg.bound.type](
                ms['train'], modl, risk, cfg.bound.delta / cfg.bound.union_bound_coeff, bound_coeff
            )
            # Fairness bound
            fairness_union = cfg.bound.union_bound_coeff if cfg.fairness_bound.union_bound_coeff is None else cfg.fairness_bound.union_bound_coeff
            fairness_bound = lambda _, risk_A, risk_B, m_1, m_2 : BOUNDS[cfg.fairness_bound.type](model, risk_A, risk_B, m_1, m_2, cfg.bound.delta / fairness_union)

            # Initializing the predictors, the number of base classifiers.
            predictors, n = initialize_predictors(
                cfg,
                data,
                seed=current_seed,
            )

            # This corresponds to the prior value to consider for each parameter.
            prior_coefficient = 1 / n if cfg.model.prior == "adjusted" else int(cfg.model.prior)

            # This corresponds to the replacement value to use in the "crop_weak_learner" method.
            prior_value = -5 if cfg.training.distribution == "categorical" else prior_coefficient

            betas = torch.ones(n) * prior_coefficient  # uniform prior
            if cfg.device == "gpu":
                betas = betas.cuda()

            model = MajorityVote(predictors, betas, n_classes, distribution_name, kl_factor)

            # We replace the features by the fixed voters' predictions.
            loaders = creating_data_loaders(data, model, cfg)

            monitor = MonitorMV(SAVE_DIR)
            routine = evaluate

            # First training phase
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
            if cfg.training.risk == 'Triple':
                model, final_bound, errors, time = \
                    triple_routine(
                        loaders, model, optimizer, bound, fairness_bound, ms, loss,
                        monitor, lr_scheduler, cfg
                    )

                metrics = compute_triple_bound(
                    model, bound, ms['train'], n, loaders['train'], loss, distribution_name,
                    0, final_bound['bound']
                )
                assert metrics['det_upper'].item() > errors['test'] - 1e-6, f"Bound {round(metrics['det_upper'].item(), 4)} cannot be smaller than error {round(errors['test'].item(), 4)}."  # Sanity check

                # We now compute the fairness metrics
                fairness_metrics = compute_fairness_metrics(model, routine, loaders, ms, n, loss, cfg)

                # Results are compiled in the 'seed_results' dictionary
                seed_results = updating_first_seed_results(seed_results, time, errors, metrics, fairness_metrics, ms)

                if cfg.fairness_bound.type == "ours_stochastic_fairness":
                    model = clip_weak_learners(
                        model, ms['train'], bound, loaders['train'], loss,
                        prior_value, distribution_name
                    )
                    model = manual_coordinate_descent(
                        model, ms['train'], bound, loaders['train'], loss,
                        distribution_name
                    )
                    model = weights_rescaling(
                        model, ms['train'], bound, loaders['train'], loss,
                        distribution_name
                    )

                    errors = compute_errors(routine, loaders, model, cfg.device, loss, ms)
                    metrics_tnd = compute_triple_bound(
                        model, bound, ms['train'], n, loaders['train'], loss,
                        distribution_name
                    )
                    assert metrics_tnd['det_upper'].item() > errors['test'] - 1e-6, \
                        f"Bound {round(metrics_tnd['det_upper'].item(), 4)} cannot be smaller than error {round(errors['test'].item(), 4)}."
                    fairness_metrics_tnd = compute_fairness_metrics(
                        model, routine, loaders, ms, n, loss, cfg, False
                    )
                elif cfg.fairness_bound.type == "oneto_fairness":
                    # The Oneto fairness pipeline does not use deterministic-conversion heuristics.
                    errors = compute_errors(routine, loaders, model, cfg.device, loss, ms)
                    metrics_tnd = compute_triple_bound(
                        model, bound, ms['train'], n, loaders['train'], loss,
                        distribution_name, 0, final_bound['bound']
                    )
                    assert metrics_tnd['det_upper'].item() > errors['test'] - 1e-6, \
                        f"Bound {round(metrics_tnd['det_upper'].item(), 4)} cannot be smaller than error {round(errors['test'].item(), 4)}."
                    fairness_metrics_tnd = compute_fairness_metrics(
                        model, routine, loaders, ms, n, loss, cfg, False
                    )
                else:
                    raise ValueError(f"Unsupported fairness bound in Triple routine: {cfg.fairness_bound.type}")

                # Results are compiled in the 'seed_results' dictionary
                seed_results = updating_last_seed_results(seed_results, cfg, errors, metrics_tnd, fairness_metrics_tnd, ms, i)

            else:
                model = oneto_routine(loaders, model, ms, cfg)

                # We need to recompute the train error, test error, and the bounds
                errors = compute_errors(routine, loaders, model, cfg.device, loss, ms)

                Gibbs_risk_upper = compute_oneto_bound(errors['train_gibbs'], ms, cfg)
                metrics_tnd = compute_triple_bound(model, bound, ms['train'], n, loaders['train'], loss,
                                                   distribution_name, 0, Gibbs_risk_upper)
                assert metrics_tnd['det_upper'].item() > errors['test'] - 1e-6, "Bound cannot be smaller than error."  # Sanity check

                # We now compute the fairness metrics
                fairness_metrics_tnd = compute_fairness_metrics(model, routine, loaders, ms, n, loss, cfg, True)
                seed_results = updating_last_seed_results(seed_results, cfg, errors, metrics_tnd, fairness_metrics_tnd, ms, i)
                seed_results = ensure_before_after_schema(seed_results)

            # save seed results
            np.save(SAVE_DIR / "err-b.npy", seed_results)
            monitor.close()

        all_seed_results.append(seed_results)

    # Aggregate all numeric metrics across seeds (mean/std/min/max/n) so hyperparameter
    # selection can use any metric, not only train/test error and one bound.
    numeric_values = {}
    for sr in all_seed_results:
        for k, v in sr.items():
            try:
                scalar = float(torch.as_tensor(v).item())
            except Exception:
                continue
            numeric_values.setdefault(k, []).append(scalar)

    metric_agg = {}
    for k, vals in numeric_values.items():
        arr = np.asarray(vals, dtype=float)
        metric_agg[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    # Keep backward-compatible top-level fields used by existing scripts.
    results = {
        "train-error": (
            metric_agg["det_OA_err_tr_af"]["mean"],
            metric_agg["det_OA_err_tr_af"]["std"],
        ),
        "test-error": (
            metric_agg["det_OA_err_te_af"]["mean"],
            metric_agg["det_OA_err_te_af"]["std"],
        ),
        cfg.bound.type: (
            metric_agg["det_OA_bnd_up_af"]["mean"],
            metric_agg["det_OA_bnd_up_af"]["std"],
        ),
        "metrics": metric_agg,
    }

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()
