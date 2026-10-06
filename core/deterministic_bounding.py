import math

import prtpy
import torch
import numpy as np

from core.bounds import test_set_bound, oneto_bound
from core.kl_inv import kl_inv
from core.utils import Phi, I, fairness_error
from tqdm import tqdm


def evaluate(dataloader, model, device, epoch=-1, bounds=None, loss=None, monitor=None, tag="val", replace_labels=False, per_base_classifier=False, fairness_eval=None):
    """
    Evaluate a posterior or its deterministic majority vote.
    """
    model.eval()

    if per_base_classifier:
        risk = torch.zeros(len(model.prior))
    else:
        risk = torch.tensor(0.)
    m = torch.tensor(0)
    if device == "gpu":
        risk = risk.cuda()
        m = m.cuda()
    objective_sum = torch.tensor(0.0, device=risk.device)
    risk_bound_sum = torch.tensor(0.0, device=risk.device)
    fairness_bound_sum = torch.tensor(0.0, device=risk.device)
    has_fairness_eval = fairness_eval is not None

    for batch in dataloader:
        if replace_labels:
            batch[1] = torch.abs(batch[1])
        data = batch[1], batch[0]
        if per_base_classifier:
            risk += model.loss_per_base_classifier(data, mean=False)
        else:
            risk += model.risk(data, loss=loss, mean=False)
        batch_size = len(data[0])
        m += batch_size

        if has_fairness_eval and not per_base_classifier:
            obj, rb, fb = fairness_eval["objective_components_fn"](
                model,
                batch,
                fairness_eval["loss"],
                fairness_eval["bound_fn"],
                fairness_eval["fairness_bound_fn"],
                fairness_eval["cfg"],
            )
            objective_sum += obj.detach() * batch_size
            risk_bound_sum += rb.detach() * batch_size
            fairness_bound_sum += fb.detach() * batch_size
    risk /= m

    if per_base_classifier:
        return risk

    total_metrics = {"error": risk.item()}
    if has_fairness_eval:
        total_metrics["cost"] = (objective_sum / m).item()
        total_metrics["risk_bound"] = (risk_bound_sum / m).item()
        total_metrics["fairness_bound"] = (fairness_bound_sum / m).item()

    if bounds is not None:
        for k in bounds.keys():
            if bounds[k] is not None:
                total_metrics[k] = bounds[k](m, model, risk).item()
    if monitor:
        monitor.write(epoch, **{tag: total_metrics})
    return total_metrics

def get_b_c(possible_values, n, distribution):
    """
    Returns the lower and upper bounds of the b and c quantities.
    """
    # The triple bound is not valid for the multiclass gaussian approach.
    # First compute the values for the upper bound on b and the lower bound on c.
    if possible_values.is_cuda:
        possible_values = possible_values.cpu()
    possible_values_np = possible_values.detach().numpy()
    sum_absolute = torch.sum(torch.abs(possible_values))
    changed_values = possible_values_np
    try:
        # This function computes the partitioning problem (2 bins)
        sums = prtpy.partition(algorithm=prtpy.partitioning.ilp,
                               items=np.sort(np.abs(changed_values)),
                               objective=prtpy.obj.MaximizeSmallestSum,
                               numbins=2,
                               time_limit=5) # time limit, in seconds (should be resolved in ~0.01 sec.)
    except ValueError:
        # In cases where the function do not converge to a solution fast enough, we avoid a crash by doing as follows
        sums = [[sum_absolute / 2], [sum_absolute / 2]]
    sum_1 = torch.sum(torch.tensor(sums[0]))
    sum_2 = torch.sum(torch.tensor(sums[1]))

    biggest_sum = torch.max(sum_1, sum_2)
    smallest_sum = torch.min(sum_1, sum_2)

    # Random-forest voters have no complementary-voter structure.
    max_err = sum_1 + sum_2
    if distribution in ['categorical', 'dirichlet']:
        min_err = 0
    else:
        min_err = -max_err

    ## We are now ready to compute the lower and upper bounds for both b anc c.
    if distribution == "categorical":
        return min_err, biggest_sum, smallest_sum, max_err
    elif distribution == "dirichlet":
        return I(max_err, min_err), I(smallest_sum, biggest_sum), I(biggest_sum, smallest_sum), I(min_err, max_err)
    elif distribution == "gaussian":
        return Phi(max_err / n ** 0.5), Phi((smallest_sum - biggest_sum) / n ** 0.5),\
               Phi((biggest_sum - smallest_sum) / n ** 0.5), Phi(min_err / n ** 0.5)

def compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name, Gibbs_risk_lower=None, Gibbs_risk_upper=None):
    """
    Pipeline for computing the deterministic bound.
    """
    post = model.get_post()
    b_lower, c_lower, b_upper, c_upper = get_b_c(post, n, distribution_name)

    assert b_lower <= b_upper + 1e-6, "Computation of the bounds on the b quantity incoherent."  # Sanity check
    assert c_lower <= c_upper + 1e-6, "Computation of the bounds on the c quantity incoherent."  # Sanity check
    assert b_upper <= c_lower + 1e-6, "Computation of the bounds on the b and c quantities incoherent."  # Sanity check

    # If the bound on the Gibbs risk is not given, we have to compute it
    if Gibbs_risk_upper is None or Gibbs_risk_lower is None:
        Gibbs_risk_lower = 0
        try:
            risk = torch.tensor(evaluate(trainloader, model, 'cpu', loss=loss)["error"])
        except RuntimeError:
            risk = torch.tensor(evaluate(trainloader, model, 'gpu', loss=loss)["error"])
        Gibbs_risk_upper = bound(m, model, risk)
        epsilon = ((model.KL() + math.log(2 * m ** 0.5 / 0.05)) / (2 * m)) ** 0.5
        Gibbs_risk_lower_oneto = risk - epsilon
        Gibbs_risk_upper_oneto = risk + epsilon
    else:
        Gibbs_risk_lower_oneto, Gibbs_risk_upper_oneto = 0, 1

    det_lower = (Gibbs_risk_lower - b_upper) / (c_upper - b_upper)
    det_upper = (Gibbs_risk_upper - b_lower) / (c_lower - b_lower)
    stats = {'det_lower': det_lower,
             'det_upper': det_upper,
             'gibbs_risk_lower': Gibbs_risk_lower,
             'gibbs_risk_upper': Gibbs_risk_upper,
             'gibbs_risk_lower_oneto': Gibbs_risk_lower_oneto,
             'gibbs_risk_upper_oneto': Gibbs_risk_upper_oneto,
             'b_lower': b_lower,
             'b_upper': b_upper,
             'c_lower': c_lower,
             'c_upper': c_upper}
    return stats

def compute_fairness_metrics(model, routine, loaders, ms, n, loss, cfg, oneto_style=False):
    """
    Returns the train error, test error, and both generalization bounds on the quantity |R_{D^a}(h)-R_{D^b}(h)|.
    """
    KL = model.KL()
    delta = cfg.bound.delta
    fairness_metrics = {}
    for type in ['DP', 'EP', 'EO']:
        if type == 'DP':
            trA = 'train_A'
            trB = 'train_B'
            teA = 'test_A'
            teB = 'test_B'
        elif type == 'EP':
            trA = 'train_A1'
            trB = 'train_B1'
            teA = 'test_A1'
            teB = 'test_B1'
        elif type == 'EO':
            trA = 'train_A0'
            trB = 'train_B0'
            teA = 'test_A0'
            teB = 'test_B0'
        train_A_error = torch.tensor(routine(loaders[trA], model, cfg.device, replace_labels=True)['error'])
        train_B_error = torch.tensor(routine(loaders[trB], model, cfg.device, replace_labels=True)['error'])
        fairness_train_error = fairness_error(train_A_error, train_B_error)
        train_A_error_gibbs = torch.tensor(routine(loaders[trA], model, cfg.device, loss=loss, replace_labels=True)['error'])
        train_B_error_gibbs = torch.tensor(routine(loaders[trB], model, cfg.device, loss=loss, replace_labels=True)['error'])
        fairness_train_error_gibbs = fairness_error(train_A_error_gibbs, train_B_error_gibbs)

        test_A_error = torch.tensor(routine(loaders[teA], model, cfg.device, replace_labels=True)['error'])
        test_B_error = torch.tensor(routine(loaders[teB], model, cfg.device, replace_labels=True)['error'])
        fairness_test_error = fairness_error(test_A_error, test_B_error)
        test_A_error_gibbs = torch.tensor(routine(loaders[teA], model, cfg.device, loss=loss, replace_labels=True)['error'])
        test_B_error_gibbs = torch.tensor(routine(loaders[teB], model, cfg.device, loss=loss, replace_labels=True)['error'])
        fairness_test_error_gibbs = fairness_error(test_A_error_gibbs, test_B_error_gibbs)

        if oneto_style:
            _, fairness_gibbs_bounds = oneto_bound(1, fairness_train_error, ms['train'], ms[trA], ms[trB], cfg)
            fairness_gibbs_bounds.append(1)
            fairness_triple_bounds = [0, 1]
        else:
            epsilon_A = (KL + np.log(4 * (ms[trA] ** 0.5) / delta)) / ms[trA]
            lower_A = kl_inv(train_A_error_gibbs, epsilon_A, 'MIN', nb_iter_max=1000)
            upper_A = kl_inv(train_A_error_gibbs, epsilon_A, 'MAX', nb_iter_max=1000)

            epsilon_B = (KL + np.log(4 * (ms[trB] ** 0.5) / delta)) / ms[trB]
            lower_B = kl_inv(train_B_error_gibbs, epsilon_B, 'MIN', nb_iter_max=1000)
            upper_B = kl_inv(train_B_error_gibbs, epsilon_B, 'MAX', nb_iter_max=1000)

            b_lower, c_lower, b_upper, c_upper = get_b_c(model.get_post(), n, cfg.training.distribution)

            fairness_gibbs_bounds, fairness_triple_bounds = compute_fairness_triple_bound(lower_A, lower_B,upper_A, upper_B,
                                                                                        b_lower, c_lower, b_upper, c_upper)

            gibbs_oneto = torch.abs(train_A_error_gibbs - train_B_error_gibbs) + \
                          torch.sqrt((KL + torch.log(4 * torch.sqrt(ms[trA]) / delta)) / (2 * ms[trA])) + \
                          torch.sqrt((KL + torch.log(4 * torch.sqrt(ms[trB]) / delta)) / (2 * ms[trB]))
            fairness_gibbs_bounds.append(gibbs_oneto)

        assert fairness_triple_bounds[0] < fairness_test_error + 1e-6, f"Lower bound cannot be bigger than actual error."  # Sanity check
        assert fairness_triple_bounds[1] > fairness_test_error - 1e-6, "Upper bound cannot be smaller than actual error." # Sanity check
        if type != 'EO':
            fairness_metrics['train_error_'+type] = fairness_train_error
            fairness_metrics['test_error_'+type] = fairness_test_error
            fairness_metrics['train_error_gibbs_'+type] = fairness_train_error_gibbs
            fairness_metrics['test_error_gibbs_'+type] = fairness_test_error_gibbs
            fairness_metrics['gibbs_bounds_'+type] = fairness_gibbs_bounds
            fairness_metrics['triple_bounds_'+type] = fairness_triple_bounds
        else:
            train_prop_y_1 = ms['train_1'] / ms['train']
            test_prop_y_1 = ms['test_1'] / ms['test']
            lw_bnd_train_prop_y_1 = test_set_bound(ms['train_1'], ms['train'], delta, type='lower')
            up_bnd_train_prop_y_1 = test_set_bound(ms['train_1'], ms['train'], delta, type='upper')
            fairness_metrics['train_error_' + type] = (1-train_prop_y_1) * fairness_train_error + train_prop_y_1 * fairness_metrics['train_error_EP']
            fairness_metrics['test_error_' + type] = (1-test_prop_y_1) * fairness_test_error + test_prop_y_1 * fairness_metrics['test_error_EP']
            fairness_metrics['train_error_gibbs_' + type] = (1-train_prop_y_1) * fairness_train_error_gibbs + train_prop_y_1 * fairness_metrics['train_error_gibbs_EP']
            fairness_metrics['test_error_gibbs_' + type] = (1-test_prop_y_1) * fairness_test_error_gibbs + test_prop_y_1 * fairness_metrics['test_error_gibbs_EP']
            fairness_metrics['gibbs_bounds_' + type] = (1-lw_bnd_train_prop_y_1) * torch.tensor(fairness_gibbs_bounds) + up_bnd_train_prop_y_1 * torch.tensor(fairness_metrics['gibbs_bounds_EP'])
            fairness_metrics['triple_bounds_' + type] = (1-lw_bnd_train_prop_y_1) * torch.tensor(fairness_triple_bounds) + up_bnd_train_prop_y_1 * torch.tensor(fairness_metrics['triple_bounds_EP'])
    return fairness_metrics

def compute_fairness_triple_bound(lower_A, lower_B, upper_A, upper_B, b_lower, c_lower, b_upper, c_upper):
    """
    Computation of both fairness upper bounds (type_1, and type_2), for both (Ra - Rb) and (Rb - Ra)
    (respectively: type_x_1 and type_x_2).
    """
    gibbs_1 = upper_A - lower_B
    gibbs_2 = upper_B - lower_A
    det_1 = (upper_A - b_lower) / (c_lower - b_lower) - (lower_B - b_upper) / (c_upper - b_upper)
    det_2 = (upper_B - b_lower) / (c_lower - b_lower) - (lower_A - b_upper) / (c_upper - b_upper)

    return [max(-gibbs_1, -gibbs_2), max(gibbs_1, gibbs_2)], [max(-det_1, -det_2), max(det_1, det_2)]

def compute_oneto_bound(error, ms, cfg):
    if cfg.training.fairness_metric == 'EP':
        ms_A = ms['train_A1']
        ms_B = ms['train_B1']
    else:
        ms_A = ms['train_A']
        ms_B = ms['train_B']
    return oneto_bound(error, 1, ms['train'], ms_A, ms_B, cfg)[0][1]

def clip_weak_learners(model, m, bound, trainloader, loss, prior_coefficient, distribution_name):
    """
    Assigns small weights to predictors with medium weights, so b and c are the smallest possible (triple bound).
    """
    best_post = model.get_unchanged_post().clone()
    sorted_post = torch.sort(torch.abs(best_post.clone()))[0]
    post = model.get_unchanged_post().clone()
    n = len(best_post)
    best_bound = compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name)['det_upper']
    print(f"\nCurrent triple bound: {round(best_bound.item(), 4)}.")
    print("Clipping weak learners...")
    pbar = tqdm(list(range(10, 100, 10)) + list(range(91, 100, 1)))
    for i in pbar:
        max_idx = int(n * i / 100) - 1
        # We try several simplifications of the posterior by clipping the smallest values to the prior values. Having
        #   lesser small values help to obtain better triple bound.
        post[torch.abs(best_post) <= sorted_post[max_idx]] = prior_coefficient
        model.set_post(post)
        cur_bound = compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name)['det_upper']
        if cur_bound < best_bound:
            best_bound = cur_bound.clone()
            best_post = post.clone()
    model.set_post(best_post)
    print(f"\nCurrent triple bound: {round(best_bound.item(), 4)}.")
    return model

def manual_coordinate_descent(model, m, bound, trainloader, loss, distribution_name):
    """
    Manual coordinate descent.
    """
    print("Manual coordinate descent...")
    best_post = model.get_unchanged_post().clone()
    n = len(best_post)
    best_bound = compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name)['det_upper']
    while True:
        previous_best_bound = best_bound.clone()
        rang, factor = [], torch.max(model.get_unchanged_post()) / 100
        for i in range(len(best_post)):
            if best_post[i] >= factor:
                rang.append(i)
        np.random.shuffle(rang)
        pbar = tqdm(rang[:min(100, len(rang))])
        for i in pbar:
            for change in ['min', 'max']:
                post = model.get_unchanged_post().detach()
                new_post_i = post[i] + factor * ((change == 'max') - (change == 'min'))
                if torch.sign(post[i]) == torch.sign(new_post_i):
                    # We slightly modify the current weighting
                    post[i] = new_post_i
                    model.set_post(post)
                    # And compute the resulting bound.
                    cur_bound = compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name)['det_upper']
                    if cur_bound < best_bound:
                        best_bound = cur_bound.clone()
                        best_post = post.clone()
                    else:
                        post[i] = best_post[i]
                        model.set_post(post)
        if previous_best_bound < best_bound + 1e-2:
            break
    model.set_post(best_post.requires_grad_(True))
    print(f"\nCurrent triple bound: {round(best_bound.item(), 4)}.")
    return model


def weights_rescaling(model, m, bound, trainloader, loss, distribution_name):
    """
    We try several rescaling values for computing the optimal triple bound (the higher the rescaling factor, the
    values for b and c are, but the worse is the KL penalty).
    """
    print("Weights rescaling...")
    initial_post = model.get_unchanged_post().clone()
    best_post = model.get_unchanged_post().clone()
    n = len(best_post)
    best_bound = compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name)['det_upper']
    low = 0
    if distribution_name == 'categorical':
        high = 10
    elif distribution_name == 'dirichlet':
        high = 5
    else:
        high = 100
    pbar = tqdm(range(15))
    for _ in pbar:
        try:
            mean = (low + high) / 2
            post = initial_post * mean
            model.set_post(post)
            cur_bound = compute_triple_bound(model, bound, m, n, trainloader, loss, distribution_name)['det_upper']
            if cur_bound < best_bound:
                best_bound = cur_bound.clone()
                best_post = post.clone()
        except AssertionError:
            model.set_post(best_post)
            high -= 1
    model.set_post(best_post)
    print(f"\nCurrent triple bound: {round(best_bound.item(), 4)}.")
    return model
