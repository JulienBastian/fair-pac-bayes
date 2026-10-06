from copy import deepcopy
from time import time
from tqdm import tqdm
import torch
from core.bounds import oneto_bound
from core.deterministic_bounding import compute_triple_bound, evaluate
from core.deterministic_bounding import compute_oneto_bound
from core.utils import fairness_error


def train_stochastic(dataloader, model, optimizer, epoch, cfg, bound=None, fairness_bound=None, loss=None,
                     monitor=None):
    """
    Train the posterior distribution over fixed voters.
    """
    model.train()

    last_iter = epoch * len(dataloader)

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        cost = get_fairness_objective(model, batch, loss, bound, fairness_bound, cfg)

        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write(last_iter + i, train={"Train-obj": cost.item()})

def get_fairness_objective_components(model, batch, loss, bound, fairness_bound, cfg):
    alpha = cfg.training.alpha
    m_batch = len(batch[0])

    x, y, s = batch
    y_flat = y.view(-1)
    s_flat = s.view(-1)
    data = y, x
    risk = model.risk(data, loss)

    if cfg.training.fairness_metric == 'DP':
        data = (torch.abs(y), x)
        idx_A = s_flat == 1
        idx_B = s_flat == 0

        m_A = torch.sum(idx_A)
        m_B = torch.sum(idx_B)

        data_A = (data[0][idx_A], data[1][idx_A])
        data_B = (data[0][idx_B], data[1][idx_B])
        if m_A == 0 or m_B == 0:
            fair_bnd = torch.tensor(0.0, device=x.device)
        else:
            risk_A = model.risk(data_A, loss)
            risk_B = model.risk(data_B, loss)
            fair_bnd = fairness_bound(model, risk_A, risk_B, m_A, m_B)

        risk_bnd = bound(m_batch, model, risk)
        cost = (1 - alpha) * risk_bnd + alpha * fair_bnd


    if cfg.training.fairness_metric in ['EO', 'EP']:
        data = torch.abs(y), x
        idx_A1 = (s_flat == 1) & (y_flat == 1)
        idx_B1 = (s_flat == 0) & (y_flat == 1)
        m_A1 = torch.sum(idx_A1)
        m_B1 = torch.sum(idx_B1)
        m_y1 = m_A1 + m_B1

        data_A1 = (data[0][idx_A1], data[1][idx_A1])
        data_B1 = (data[0][idx_B1], data[1][idx_B1])
        if m_A1 == 0 or m_B1 == 0:
            fair_bnd_y1 = torch.tensor(0.0, device=x.device)
        else:
            risk_A1 = model.risk(data_A1, loss)
            risk_B1 = model.risk(data_B1, loss)
            fair_bnd_y1 = fairness_bound(model, risk_A1, risk_B1, m_A1, m_B1)

        if cfg.training.fairness_metric == 'EO':
            idx_A0 = (s_flat == 1) & (y_flat == -1)
            idx_B0 = (s_flat == 0) & (y_flat == -1)
            m_A0 = torch.sum(idx_A0)
            m_B0 = torch.sum(idx_B0)
            m_y0 = m_A0 + m_B0
            denom = m_y0 + m_y1
            prop_y1 = m_y1 / denom if denom > 0 else torch.tensor(0.0, device=x.device)
            prop_y0 = 1 - prop_y1

            if m_A0 == 0 or m_B0 == 0:
                fair_bnd_y0 = torch.tensor(0.0, device=x.device)
            else:
                data_A0 = (data[0][idx_A0], data[1][idx_A0])
                data_B0 = (data[0][idx_B0], data[1][idx_B0])
                risk_A0 = model.risk(data_A0, loss)
                risk_B0 = model.risk(data_B0, loss)
                fair_bnd_y0 = fairness_bound(model, risk_A0, risk_B0, m_A0, m_B0)

            risk_bnd = bound(m_batch, model, risk)
            fair_bnd = prop_y0 * fair_bnd_y0 + prop_y1 * fair_bnd_y1
            cost = (1 - alpha) * risk_bnd + alpha * fair_bnd
        else :
            risk_bnd = bound(m_batch, model, risk)
            fair_bnd = fair_bnd_y1
            cost = (1 - alpha) * risk_bnd + alpha * fair_bnd
    return cost, risk_bnd, fair_bnd


def get_fairness_objective(model, batch, loss, bound, fairness_bound, cfg):
    return get_fairness_objective_components(
        model, batch, loss, bound, fairness_bound, cfg
    )[0]


def triple_routine(loaders, model, optimizer, bound, fairness_bound, ms, loss, monitor,
                   lr_scheduler, cfg):
    """
    Main training pipeline.
    """
    bound_type = cfg.bound.type
    num_epochs = cfg.training.num_epochs
    distribution_name = cfg.training.distribution

    best_obj = float("inf")
    best_model = deepcopy(model)
    no_improv = 0
    if cfg.device == "gpu":
        model = model.cuda()

    train_routine, val_routine, test_routine = train_stochastic, evaluate, evaluate
    metric_to_optimize = "cost"

    # To keep track of the total training time
    t1 = time()

    pbar = tqdm(range(num_epochs))
    for e in pbar:
        no_improv += 1
        # A training epoch is completed
        train_routine(
            loaders['train'], model, optimizer, e, cfg, bound=bound, fairness_bound=fairness_bound,
            loss=loss, monitor=monitor
        )
        # Just for monitoring purposes
        train_stats = val_routine(
            loaders['train'],
            model,
            cfg.device,
            epoch=e,
            bounds={bound_type: bound},
            loss=loss,
            monitor=monitor,
            tag="train",
            fairness_eval={
                "objective_components_fn": get_fairness_objective_components,
                "bound_fn": bound,
                "fairness_bound_fn": fairness_bound,
                "cfg": cfg,
                "loss": loss,
            },
        )

        pbar.set_description("train obj %s" % train_stats[metric_to_optimize])
        # If there are improvements, we update the best model
        if train_stats[metric_to_optimize] < best_obj:
            best_obj = train_stats[metric_to_optimize]
            best_model = deepcopy(model)
            no_improv = 0
        if lr_scheduler:
            lr_scheduler.step(train_stats[metric_to_optimize])
        # This is the criteria for early stopping
        if no_improv == num_epochs // 4:
            break
    t2 = time()

    train_error_lz = test_routine(loaders['train_0'], best_model, cfg.device)
    train_error_lo = test_routine(loaders['train_1'], best_model, cfg.device)
    test_error_lz = test_routine(loaders['test_0'], best_model, cfg.device)
    test_error_lo = test_routine(loaders['test_1'], best_model, cfg.device)
    train_error_gibbs_lz = test_routine(loaders['train_0'], best_model, cfg.device, loss=loss)
    train_error_gibbs_lo = test_routine(loaders['train_1'], best_model, cfg.device, loss=loss)
    test_error_gibbs_lz = test_routine(loaders['test_0'], best_model, cfg.device, loss=loss)
    test_error_gibbs_lo = test_routine(loaders['test_1'], best_model, cfg.device, loss=loss)

    test_error = (test_error_lo['error'] * ms['test_1'] + test_error_lz['error'] * ms['test_0']) / ms['test']
    if cfg.training.fairness_metric == "DP":
        test_A = test_routine(loaders['test_A'], best_model, cfg.device, replace_labels=True)['error']
        test_B = test_routine(loaders['test_B'], best_model, cfg.device, replace_labels=True)['error']
        test_fairness = abs(test_A - test_B)
    elif cfg.training.fairness_metric == "EP":
        test_A1 = test_routine(loaders['test_A1'], best_model, cfg.device, replace_labels=True)['error']
        test_B1 = test_routine(loaders['test_B1'], best_model, cfg.device, replace_labels=True)['error']
        test_fairness = abs(test_A1 - test_B1)
    elif cfg.training.fairness_metric == "EO":
        test_A0 = test_routine(loaders['test_A0'], best_model, cfg.device, replace_labels=True)['error']
        test_B0 = test_routine(loaders['test_B0'], best_model, cfg.device, replace_labels=True)['error']
        test_A1 = test_routine(loaders['test_A1'], best_model, cfg.device, replace_labels=True)['error']
        test_B1 = test_routine(loaders['test_B1'], best_model, cfg.device, replace_labels=True)['error']
        fair_0 = abs(test_A0 - test_B0)
        fair_1 = abs(test_A1 - test_B1)
        prop_y1 = ms['test_1'] / ms['test']
        prop_y0 = 1 - prop_y1
        test_fairness = prop_y0 * fair_0 + prop_y1 * fair_1
    else:
        raise ValueError(f"Unsupported fairness metric: {cfg.training.fairness_metric}")

    fairness_value = test_fairness.item() if torch.is_tensor(test_fairness) else float(test_fairness)
    string = (
        f"Test error: {round(test_error.item(), 4)}; "
        f"test {cfg.training.fairness_metric}: {round(fairness_value, 4)}"
    )
    best_train_stats = val_routine(
        loaders['train'],
        best_model,
        cfg.device,
        bounds={bound_type: bound},
        loss=loss,
        fairness_eval={
            "objective_components_fn": get_fairness_objective_components,
            "bound_fn": bound,
            "fairness_bound_fn": fairness_bound,
            "cfg": cfg,
            "loss": loss,
        },
    )

    risk_bound_value = best_train_stats["risk_bound"]
    fairness_bound_value = best_train_stats["fairness_bound"]
    cost_value = best_train_stats["cost"]
    final_bound = {'bound': round(risk_bound_value, 4)}
    string += (
        f"; cost: {round(cost_value, 4)}"
        f"; {bound_type} risk bound: {round(risk_bound_value, 4)}"
        f"; fairness bound: {round(fairness_bound_value, 4)}"
    )

    n = torch.prod(torch.tensor(best_model.get_unchanged_post().shape))
    triple_bound = compute_triple_bound(best_model, bound, ms['train'], n, loaders['train'], loss,
                                        distribution_name, 0, final_bound['bound'])['det_upper']
    string += f"; triple bound: {round(triple_bound.item(), 4)}\n"
    print(string)

    errors = {'train_lz': train_error_lz['error'],
              'train_lo': train_error_lo['error'],
              'test_lz': test_error_lz['error'],
              'test_lo': test_error_lo['error'],
              'test': test_error,
              'train_gibbs_lz': train_error_gibbs_lz['error'],
              'train_gibbs_lo': train_error_gibbs_lo['error'],
              'test_gibbs_lz': test_error_gibbs_lz['error'],
              'test_gibbs_lo': test_error_gibbs_lo['error']}

    return best_model, final_bound, errors, t2 - t1

def oneto_routine(loaders, model, ms, cfg):
    routine = evaluate

    post_err = routine(loaders['train'], model, cfg.device, per_base_classifier=True)
    if cfg.training.fairness_metric == 'DP':
        train_error_A = routine(loaders['train_A'], model, cfg.device, replace_labels=True, per_base_classifier=True)
        train_error_B = routine(loaders['train_B'], model, cfg.device, replace_labels=True, per_base_classifier=True)
        post_fai = fairness_error(train_error_A, train_error_B)
    elif cfg.training.fairness_metric == 'EP':
        train_error_A1 = routine(loaders['train_A1'], model, cfg.device, replace_labels=True, per_base_classifier=True)
        train_error_B1 = routine(loaders['train_B1'], model, cfg.device, replace_labels=True, per_base_classifier=True)
        post_fai = fairness_error(train_error_A1, train_error_B1)
    else:
        raise ValueError(f"Unsupported fairness metric for Oneto: {cfg.training.fairness_metric}")

    # The exponential + normalization is applied when the prediction is computed.
    true_post = -cfg.training.gamma * (post_err + cfg.training.alpha * post_fai)
    model.set_post(true_post)

    # summary print
    post = model.get_post()
    train_gibbs = torch.sum(post * post_err)
    train_fairness = torch.sum(post * post_fai)
    risk_upper = compute_oneto_bound(train_gibbs, ms, cfg)
    if cfg.training.fairness_metric == "EP":
        ms_A, ms_B = ms["train_A1"], ms["train_B1"]
    else:
        ms_A, ms_B = ms["train_A"], ms["train_B"]
    fairness_upper = oneto_bound(1, train_fairness, ms["train"], ms_A, ms_B, cfg)[1][1]

    test_error = routine(loaders["test"], model, cfg.device)["error"]
    if cfg.training.fairness_metric == "DP":
        test_A = routine(loaders["test_A"], model, cfg.device, replace_labels=True)["error"]
        test_B = routine(loaders["test_B"], model, cfg.device, replace_labels=True)["error"]
        test_fairness = abs(test_A - test_B)
    elif cfg.training.fairness_metric == "EP":
        test_A1 = routine(loaders["test_A1"], model, cfg.device, replace_labels=True)["error"]
        test_B1 = routine(loaders["test_B1"], model, cfg.device, replace_labels=True)["error"]
        test_fairness = abs(test_A1 - test_B1)
    else:
        # Oneto + EO is already blocked by config checks.
        raise ValueError(f"Unsupported fairness metric for Oneto: {cfg.training.fairness_metric}")

    print(
        f"Oneto summary: Test error: {round(float(test_error), 4)}; "
        f"test {cfg.training.fairness_metric}: {round(float(test_fairness), 4)}; "
        f"oneto risk bound: {round(float(risk_upper), 4)}; "
        f"oneto fairness bound: {round(float(fairness_upper), 4)}"
    )

    return model
