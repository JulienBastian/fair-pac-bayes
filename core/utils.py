import math
import torch
import numpy as np
import random
import hydra

from torch.special import erf
from torch.utils.data import DataLoader

from core.expected_risk import BetaInc
from data.init import BINARY_DATASETS, TorchDataset
from models.random_forest import random_forest

def whether_to_run_run(cfg):
    """
    Many tests ensuring that the current run has consistent hyperparameters.
    """
    assert cfg.training.distribution in ["categorical", "dirichlet", "gaussian"]
    if cfg.training.distribution == "categorical":
        assert cfg.model.prior == "adjusted"
    elif cfg.training.distribution == "dirichlet":
        assert cfg.model.prior in ["adjusted", 1]
    elif cfg.training.distribution == "gaussian":
        assert cfg.model.prior == 0

    assert cfg.model.pred == "RandomForests", "Not a valid choice of model."
    assert cfg.dataset in BINARY_DATASETS
    assert cfg.training.sensible_variable in ['OTHER', 'GENDER']

    assert cfg.training.fairness_metric in ['DP', 'EO', 'EP']
    assert cfg.training.risk in ['Triple', 'Oneto']
    assert not (cfg.training.risk == 'Oneto' and cfg.training.distribution != "categorical")
    assert not (cfg.training.risk == 'Oneto' and cfg.training.fairness_metric == "EO")


def create_root_dir(cfg):
    ROOT_DIR = (f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/fairness={cfg.fairness_bound.type}/{cfg.model.pred}/"
                f"fairness_metric={cfg.training.fairness_metric}/sensible_variable={cfg.training.sensible_variable}/"
                f"alpha={cfg.training.alpha}/gamma={cfg.training.gamma}/distribution={cfg.training.distribution}/")

    ROOT_DIR += (
        f"n={cfg.model.n}/"
        f"criterion={cfg.model.tree_criterion}/"
        f"class-weight={cfg.model.tree_class_weight}/"
        f"max-depth={cfg.model.max_tree_depth}/"
        f"max-features={cfg.model.tree_max_features}/"
        f"min-split={cfg.model.tree_min_samples_split}/"
        f"min-leaf={cfg.model.tree_min_samples_leaf}/"
        f"samples-prop={cfg.model.samples_prop}/"
    )
    if cfg.training.distribution == 'dirichlet':
        ROOT_DIR += f"prior={cfg.model.prior}/"

    return ROOT_DIR


def initialize_predictors(cfg, data, seed):
    return random_forest(
        cfg.model.n,
        data.X['train_prior'],
        data.y['train_prior'],
        samples_prop=cfg.model.samples_prop,
        max_depth=cfg.model.max_tree_depth,
        criterion=cfg.model.tree_criterion,
        class_weight=cfg.model.tree_class_weight,
        max_features=cfg.model.tree_max_features,
        min_samples_split=cfg.model.tree_min_samples_split,
        min_samples_leaf=cfg.model.tree_min_samples_leaf,
        binary=data.binary,
        seed=seed,
    )

def updating_first_seed_results(seed_results, time, errors, metrics, fairness_metrics, ms):
    """
    A metric is composed of at most 6 codes:
    1. Choices between 'det' (deterministic) and 'gbs' (Gibbs);
    2. Choices between 'OA' (overall, the standard loss), 'DP' (demographic parity),
            'EO' (equalized odds), and 'EP' (equalized opportunity).
    3. Choices between 'err' (error) and 'bnd' (bound);
    4. Choices between 'tr' (train) and 'te' (test) when 3. is 'err', and choices between 'lw' (lower), 'up' (upper),
            and 'ou' (Oneto, upper), when 3. is 'bnd';
    5. Choices between 'bf' (before finetuning) and 'af' (after finetuning).
    6. Choices between 'lz' (label = zero) and 'lo' (label = one).
    """
    m_train = ms['train']
    m_train_lz = ms['train_0']
    m_train_lo = ms['train_1']
    m_test = ms['test']
    m_test_lz = ms['test_0']
    m_test_lo = ms['test_1']

    seed_results["b_lower_bf"] = metrics['b_lower']
    seed_results["b_upper_bf"] = metrics['b_upper']
    seed_results["c_lower_bf"] = metrics['c_lower']
    seed_results["c_upper_bf"] = metrics['c_upper']

    seed_results["det_OA_err_tr_bf_lz"] = errors['train_lz']
    seed_results["det_OA_err_tr_bf_lo"] = errors['train_lo']
    seed_results["det_OA_err_tr_bf"] = (errors['train_lz'] * m_train_lz + errors['train_lo'] * m_train_lo) / m_train
    seed_results["det_OA_err_te_bf_lz"] = errors['test_lz']
    seed_results["det_OA_err_te_bf_lo"] = errors['test_lo']
    seed_results["det_OA_err_te_bf"] = (errors['test_lz'] * m_test_lz + errors['test_lo'] * m_test_lo) / m_test
    seed_results["det_OA_bnd_up_bf"] = metrics['det_upper'].item()

    seed_results["gbs_OA_err_tr_bf_lz"] = errors['train_gibbs_lz']
    seed_results["gbs_OA_err_tr_bf_lo"] = errors['train_gibbs_lo']
    seed_results["gbs_OA_err_tr_bf"] =  (errors['train_gibbs_lz'] * m_train_lz + errors['train_gibbs_lo'] * m_train_lo) / m_train
    seed_results["gbs_OA_err_te_bf_lz"] = errors['test_gibbs_lz']
    seed_results["gbs_OA_err_te_bf_lo"] = errors['test_gibbs_lo']
    seed_results["gbs_OA_err_te_bf"] = (errors['test_gibbs_lz'] * m_test_lz + errors['test_gibbs_lo'] * m_test_lo) / m_test
    seed_results["gbs_OA_bnd_up_bf"] = metrics['gibbs_risk_upper']

    for type in ['DP', 'EO', 'EP']:
        seed_results["det_" + type + "_err_tr_bf"] = fairness_metrics['train_error_' + type]
        seed_results["det_" + type + "_err_te_bf"] = fairness_metrics['test_error_' + type]
        seed_results["det_" + type + "_bnd_lw_bf"] = fairness_metrics['triple_bounds_' + type][0]
        seed_results["det_" + type + "_bnd_up_bf"] = fairness_metrics['triple_bounds_' + type][1]

        seed_results["gbs_" + type + "_err_tr_bf"] = fairness_metrics['train_error_gibbs_' + type]
        seed_results["gbs_" + type + "_err_te_bf"] = fairness_metrics['test_error_gibbs_' + type]
        seed_results["gbs_" + type + "_bnd_lw_bf"] = fairness_metrics['gibbs_bounds_' + type][0]
        seed_results["gbs_" + type + "_bnd_up_bf"] = fairness_metrics['gibbs_bounds_' + type][1]
        seed_results["gbs_" + type + "_bnd_ou_bf"] = fairness_metrics['gibbs_bounds_' + type][2]

    seed_results["time"] = time
    return seed_results

def updating_last_seed_results(seed_results, cfg, errors, metrics_tnd, fairness_metrics_tnd, ms, i):
    # See description of the above function.
    m_train = ms['train']
    m_train_lz = ms['train_0']
    m_train_lo = ms['train_1']
    m_test = ms['test']
    m_test_lz = ms['test_0']
    m_test_lo = ms['test_1']

    seed_results["b_lower_af"] = metrics_tnd['b_lower']
    seed_results["b_upper_af"] = metrics_tnd['b_upper']
    seed_results["c_lower_af"] = metrics_tnd['c_lower']
    seed_results["c_upper_af"] = metrics_tnd['c_upper']

    seed_results["det_OA_err_tr_af_lz"] = errors['train_lz']
    seed_results["det_OA_err_tr_af_lo"] = errors['train_lo']
    seed_results["det_OA_err_tr_af"] = (errors['train_lz'] * m_train_lz + errors['train_lo'] * m_train_lo) / m_train
    seed_results["det_OA_err_te_af_lz"] = errors['test_lz']
    seed_results["det_OA_err_te_af_lo"] = errors['test_lo']
    seed_results["det_OA_err_te_af"] = (errors['test_lz'] * m_test_lz + errors['test_lo'] * m_test_lo) / m_test
    seed_results["det_OA_bnd_lw_af"] = metrics_tnd['det_lower'].item()
    seed_results["det_OA_bnd_up_af"] = metrics_tnd['det_upper'].item()

    seed_results["gbs_OA_err_tr_af_lz"] = errors['train_gibbs_lz']
    seed_results["gbs_OA_err_tr_af_lo"] = errors['train_gibbs_lo']
    seed_results["gbs_OA_err_tr_af"] = (errors['train_gibbs_lz'] * m_train_lz + errors['train_gibbs_lo'] * m_train_lo) / m_train
    seed_results["gbs_OA_err_te_af_lz"] = errors['test_gibbs_lz']
    seed_results["gbs_OA_err_te_af_lo"] = errors['test_gibbs_lo']
    seed_results["gbs_OA_err_te_af"] = (errors['test_gibbs_lz'] * m_test_lz + errors['test_gibbs_lo'] * m_test_lo) / m_test
    seed_results["gbs_OA_bnd_up_af"] = metrics_tnd['gibbs_risk_upper']
    seed_results["gbs_OA_bnd_lw_af"] = metrics_tnd['gibbs_risk_lower']
    seed_results["gbs_OA_bnd_ou_af"] = metrics_tnd['gibbs_risk_upper_oneto']

    for type in ['DP', 'EO', 'EP']:
        seed_results["det_" + type + "_err_tr_af"] = fairness_metrics_tnd['train_error_' + type]
        seed_results["det_" + type + "_err_te_af"] = fairness_metrics_tnd['test_error_' + type]
        seed_results["det_" + type + "_bnd_lw_af"] = fairness_metrics_tnd['triple_bounds_' + type][0]
        seed_results["det_" + type + "_bnd_up_af"] = fairness_metrics_tnd['triple_bounds_' + type][1]

        seed_results["gbs_" + type + "_err_tr_af"] = fairness_metrics_tnd['train_error_gibbs_' + type]
        seed_results["gbs_" + type + "_err_te_af"] = fairness_metrics_tnd['test_error_gibbs_' + type]
        seed_results["gbs_" + type + "_bnd_lw_af"] = fairness_metrics_tnd['gibbs_bounds_' + type][0]
        seed_results["gbs_" + type + "_bnd_up_af"] = fairness_metrics_tnd['gibbs_bounds_' + type][1]
        seed_results["gbs_" + type + "_bnd_ou_af"] = fairness_metrics_tnd['gibbs_bounds_' + type][2]

    seed_results["seed"] = cfg.training.seed + i
    return seed_results


def ensure_before_after_schema(seed_results):
    """
    Ensure compatibility with analysis scripts that expect both *_af and *_bf keys.
    In Oneto runs there is only one phase, so we mirror missing *_bf values from *_af.
    """
    for key, value in list(seed_results.items()):
        if key.endswith("_af"):
            bf_key = key[:-3] + "_bf"
            if bf_key not in seed_results:
                seed_results[bf_key] = value
    if "time" not in seed_results:
        seed_results["time"] = float("nan")
    return seed_results

def bin_cum(k, m, r):
    """
    P(x <= k), if X ~ Bin(m, r)
    """
    prob_cum = 0
    for i in range(k + 1):
        prob_cum += math.exp(log_prob_bin(torch.tensor(i), m, r))
    return prob_cum

def log_stirling_approximation(m):
    """
    Stirling's approximation for the logarithm of the factorial
    """
    if m == 0:
        return 0
    if m < 25:
        return math.log(math.factorial(m))
    return m * torch.log(m) - m + 0.5 * torch.log(2 * math.pi * m)


def log_binomial_coefficient(m, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation
    """
    return (log_stirling_approximation(m) -
            log_stirling_approximation(k) -
            log_stirling_approximation(m - k))

def log_prob_bin(k, m, r):
    """
    Logarithm of P(x = k), if X ~ Bin(m, r)
    """
    epsilon = torch.tensor(1e-10)
    if not torch.is_tensor(r):
        r = torch.tensor(r)
    return log_binomial_coefficient(m, k) + k * torch.log(torch.max(r, epsilon)) + \
                                      (m - k) * torch.log(torch.max(1 - r, epsilon))

def get_n_classes(dataset):
    """
    Given a dataset name, returns the number of classes it contains.
    """
    if dataset in BINARY_DATASETS:
        return 2
    assert False, "Incorrect dataset"

def deterministic(random_state):
    """
    Set the random seed for all the packages that are used.
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def I(l, u):
    """
    Computes the incomplete beta function at x = 0.5.
    """
    return BetaInc.apply(l, u, torch.tensor(0.5))

def Phi(z):
    """
    Computes the cumulative distribution function (CDF) of a standard unit gaussian function at point z.
    """
    return 1 / 2 * (1 - erf(z / 2 ** 0.5))

def fairness_error(r1, r2):
    """
    Computes the cumulative distribution function (CDF) of a standard unit gaussian function at point z.
    """
    return np.abs(r1 - r2)

def compute_errors(routine, loaders, model, device, loss, ms):
    train_error_lz = routine(loaders['train_0'], model, device)
    train_error_lo = routine(loaders['train_1'], model, device)
    test_error_lz = routine(loaders['test_0'], model, device)
    test_error_lo = routine(loaders['test_1'], model, device)
    test_error = (test_error_lo['error'] * ms['test_1'] + test_error_lz['error'] * ms['test_0']) / ms['test']
    train_error_gibbs_lz = routine(loaders['train_0'], model, device, loss=loss)
    train_error_gibbs_lo = routine(loaders['train_1'], model, device, loss=loss)
    train_error_gibbs = (train_error_gibbs_lo['error'] * ms['train_1'] + train_error_gibbs_lz['error'] * ms['train_0']) / ms['train']
    test_error_gibbs_lz = routine(loaders['test_0'], model, device, loss=loss)
    test_error_gibbs_lo = routine(loaders['test_1'], model, device, loss=loss)

    errors = {'train_lz': train_error_lz['error'],
              'train_lo': train_error_lo['error'],
              'test_lz': test_error_lz['error'],
              'test_lo': test_error_lo['error'],
              'test': test_error,
              'train_gibbs_lz': train_error_gibbs_lz['error'],
              'train_gibbs_lo': train_error_gibbs_lo['error'],
              'train_gibbs': train_error_gibbs,
              'test_gibbs_lz': test_error_gibbs_lz['error'],
              'test_gibbs_lo': test_error_gibbs_lo['error']}
    return errors

def creating_data_loaders(data, model, cfg):
    # Preserve the sensitive attribute (column 0) before applying model.forward.
    keys = ['train', 'train_prior', 'train_0', 'train_1', 'train_A', 'train_B',
            'train_A0', 'train_B0', 'train_A1', 'train_B1',
            'test', 'test_0', 'test_1', 'test_A', 'test_B',
            'test_A0', 'test_B0', 'test_A1', 'test_B1']

    data.X['train'] = model.forward(torch.tensor(data.X['train']))
    data.X['train_A'] = model.forward(torch.tensor(data.X['train_A']))
    data.X['train_B'] = model.forward(torch.tensor(data.X['train_B']))
    data.X['train_A0'] = model.forward(torch.tensor(data.X['train_A0']))
    data.X['train_B0'] = model.forward(torch.tensor(data.X['train_B0']))
    data.X['train_A1'] = model.forward(torch.tensor(data.X['train_A1']))
    data.X['train_B1'] = model.forward(torch.tensor(data.X['train_B1']))
    data.X['train_0'] = model.forward(torch.tensor(data.X['train_0']))
    data.X['train_1'] = model.forward(torch.tensor(data.X['train_1']))

    data.X['test'] = model.forward(torch.tensor(data.X['test']))
    data.X['test_A'] = model.forward(torch.tensor(data.X['test_A']))
    data.X['test_B'] = model.forward(torch.tensor(data.X['test_B']))
    data.X['test_A0'] = model.forward(torch.tensor(data.X['test_A0']))
    data.X['test_B0'] = model.forward(torch.tensor(data.X['test_B0']))
    data.X['test_A1'] = model.forward(torch.tensor(data.X['test_A1']))
    data.X['test_B1'] = model.forward(torch.tensor(data.X['test_B1']))
    data.X['test_0'] = model.forward(torch.tensor(data.X['test_0']))
    data.X['test_1'] = model.forward(torch.tensor(data.X['test_1']))
    if cfg.device == "gpu":
        data.X['train'] = data.X['train'].cuda()
        data.X['train_A'] = data.X['train_A'].cuda()
        data.X['train_B'] = data.X['train_B'].cuda()
        data.X['train_A0'] = data.X['train_A0'].cuda()
        data.X['train_B0'] = data.X['train_B0'].cuda()
        data.X['train_A1'] = data.X['train_A1'].cuda()
        data.X['train_B1'] = data.X['train_B1'].cuda()
        data.X['train_0'] = data.X['train_0'].cuda()
        data.X['train_1'] = data.X['train_1'].cuda()
        data.X['test'] = data.X['test'].cuda()
        data.X['test_A'] = data.X['test_A'].cuda()
        data.X['test_A0'] = data.X['test_A0'].cuda()
        data.X['test_A1'] = data.X['test_A1'].cuda()
        data.X['test_B'] = data.X['test_B'].cuda()
        data.X['test_B0'] = data.X['test_B0'].cuda()
        data.X['test_B1'] = data.X['test_B1'].cuda()
        data.X['test_0'] = data.X['test_0'].cuda()
        data.X['test_1'] = data.X['test_1'].cuda()

        data.y['train'] = torch.from_numpy(data.y['train']).cuda()
        data.y['train_A'] = torch.from_numpy(data.y['train_A']).cuda()
        data.y['train_A0'] = torch.from_numpy(data.y['train_A0']).cuda()
        data.y['train_A1'] = torch.from_numpy(data.y['train_A1']).cuda()
        data.y['train_B'] = torch.from_numpy(data.y['train_B']).cuda()
        data.y['train_B0'] = torch.from_numpy(data.y['train_B0']).cuda()
        data.y['train_B1'] = torch.from_numpy(data.y['train_B1']).cuda()
        data.y['train_0'] = torch.from_numpy(data.y['train_0']).cuda()
        data.y['train_1'] = torch.from_numpy(data.y['train_1']).cuda()
        data.y['test'] = torch.from_numpy(data.y['test']).cuda()
        data.y['test_A'] = torch.from_numpy(data.y['test_A']).cuda()
        data.y['test_A0'] = torch.from_numpy(data.y['test_A0']).cuda()
        data.y['test_A1'] = torch.from_numpy(data.y['test_A1']).cuda()
        data.y['test_B'] = torch.from_numpy(data.y['test_B']).cuda()
        data.y['test_B0'] = torch.from_numpy(data.y['test_B0']).cuda()
        data.y['test_B1'] = torch.from_numpy(data.y['test_B1']).cuda()
        data.y['test_0'] = torch.from_numpy(data.y['test_0']).cuda()
        data.y['test_1'] = torch.from_numpy(data.y['test_1']).cuda()

        for k in keys:
            if data.s[k] is not None:
                data.s[k] = torch.as_tensor(data.s[k], dtype=torch.long).cuda()

    # Pass s to each dataset so data loaders return (x, y, s) batches.
    train_ds = TorchDataset(data.X['train'], data.y['train'], s=data.s['train'])
    train_0_ds = TorchDataset(data.X['train_0'], data.y['train_0'], s=data.s['train_0'])
    train_1_ds = TorchDataset(data.X['train_1'], data.y['train_1'], s=data.s['train_1'])
    train_A_ds = TorchDataset(data.X['train_A'], data.y['train_A'], s=data.s['train_A'])
    train_B_ds = TorchDataset(data.X['train_B'], data.y['train_B'], s=data.s['train_B'])
    train_A0_ds = TorchDataset(data.X['train_A0'], data.y['train_A0'], s=data.s['train_A0'])
    train_B0_ds = TorchDataset(data.X['train_B0'], data.y['train_B0'], s=data.s['train_B0'])
    train_A1_ds = TorchDataset(data.X['train_A1'], data.y['train_A1'], s=data.s['train_A1'])
    train_B1_ds = TorchDataset(data.X['train_B1'], data.y['train_B1'], s=data.s['train_B1'])
    test_ds = TorchDataset(data.X['test'], data.y['test'], s=data.s['test'])
    test_0_ds = TorchDataset(data.X['test_0'], data.y['test_0'], s=data.s['test_0'])
    test_1_ds = TorchDataset(data.X['test_1'], data.y['test_1'], s=data.s['test_1'])
    test_A_ds = TorchDataset(data.X['test_A'], data.y['test_A'], s=data.s['test_A'])
    test_B_ds = TorchDataset(data.X['test_B'], data.y['test_B'], s=data.s['test_B'])
    test_A0_ds = TorchDataset(data.X['test_A0'], data.y['test_A0'], s=data.s['test_A0'])
    test_B0_ds = TorchDataset(data.X['test_B0'], data.y['test_B0'], s=data.s['test_B0'])
    test_A1_ds = TorchDataset(data.X['test_A1'], data.y['test_A1'], s=data.s['test_A1'])
    test_B1_ds = TorchDataset(data.X['test_B1'], data.y['test_B1'], s=data.s['test_B1'])

    loaders = {'train': DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                   num_workers=cfg.num_workers, shuffle=True),
               'train_0': DataLoader(train_0_ds, batch_size=cfg.training.batch_size,
                                     num_workers=cfg.num_workers, shuffle=True),
               'train_1': DataLoader(train_1_ds, batch_size=cfg.training.batch_size,
                                     num_workers=cfg.num_workers, shuffle=True),
               'train_A': DataLoader(train_A_ds, batch_size=cfg.training.batch_size,
                                     num_workers=cfg.num_workers, shuffle=True),
               'train_B': DataLoader(train_B_ds, batch_size=cfg.training.batch_size,
                                     num_workers=cfg.num_workers, shuffle=True),
               'train_A0': DataLoader(train_A0_ds, batch_size=cfg.training.batch_size,
                                      num_workers=cfg.num_workers, shuffle=True),
               'train_B0': DataLoader(train_B0_ds, batch_size=cfg.training.batch_size,
                                      num_workers=cfg.num_workers, shuffle=True),
               'train_A1': DataLoader(train_A1_ds, batch_size=cfg.training.batch_size,
                                      num_workers=cfg.num_workers, shuffle=True),
               'train_B1': DataLoader(train_B1_ds, batch_size=cfg.training.batch_size,
                                      num_workers=cfg.num_workers, shuffle=True),
               'test': DataLoader(test_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_0': DataLoader(test_0_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_1': DataLoader(test_1_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_A': DataLoader(test_A_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_B': DataLoader(test_B_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_A0': DataLoader(test_A0_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_B0': DataLoader(test_B0_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_A1': DataLoader(test_A1_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
               'test_B1': DataLoader(test_B1_ds, batch_size=4096, num_workers=cfg.num_workers, shuffle=False)}
    return loaders
