import math
import numpy as np
import torch

from core.kl_inv import klInvFunction
from core.kl_inv import kl_inv
from core.utils import bin_cum


def test_set_bound(m_err, m, delta, type='upper'):
    """
    Implementation of Langford's test set upper and lower bound (respectively: Theorem 3.3 and 3.9,
    Tutorial on Practical Prediction Theory for Classification).
    """

    if type == 'lower':
        gamma_sup, gamma_inf, gamma = 1, 0, 0.5
        for j in range(10):
            pro = bin_cum(m_err, m, gamma)
            if 1 - pro >= delta:
                gamma_sup = gamma
            else:
                gamma_inf = gamma
            gamma = (gamma_sup + gamma_inf) / 2
    elif type == 'upper':
        gamma_sup, gamma_inf, gamma = 1, 0, 0.5
        for j in range(10):
            pro = bin_cum(m_err, m, gamma)
            if pro >= delta:
                gamma_inf = gamma
            else:
                gamma_sup = gamma
            gamma = (gamma_sup + gamma_inf) / 2
    return gamma

def seeger_bound(m, model, risk, delta, coeff=1, verbose=False, monitor=None, lower=False):
    """
    Implementation of Seeger's PAC-Bayes bound.
    """
    # We gather the divergence penalty depending on the situation
    kl = model.KL()

    # We compute the constant depending on the predictor type and the bound type
    const = np.log(2 * (m ** 0.5) / delta)

    if coeff == 0:
        bound = 1
    else:
        # Possible cases where kl + const < 0 for disintegrated bounds, thus max().
        if lower:
            bound = coeff * klInvFunction.apply(risk, torch.max(kl + const, torch.tensor(0)) / m, 'MIN')
        else:
            bound = coeff * klInvFunction.apply(risk, torch.max(kl + const, torch.tensor(0)) / m, 'MAX')

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, m={m}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound

def oneto_bound(err, fairness_err, m, m_1, m_2, cfg): #for evaluation purposes
    alpha = cfg.training.alpha
    gamma = cfg.training.gamma
    delta = cfg.bound.delta / 8

    a = gamma * ((1 - alpha) * (1 / (2 * m)) ** 0.5 + alpha * (1 / (2 * m_1) ** 0.5 + 1 / (2 * m_2) ** 0.5))
    b = 2 * gamma * ((1 - alpha) * (math.log(2 * m ** 0.5 / delta) / (2 * m)) ** 0.5 +
                     alpha * ((math.log(2 * m_1 ** 0.5 / delta) / (2 * m_1)) ** 0.5 +
                              (math.log(2 * m_2 ** 0.5 / delta) / (2 * m_2)) ** 0.5))
    KL = a ** 2 + 2 * a * b ** 0.5 + b

    bnd_epsilon = ((KL + math.log(2 * m ** 0.5 / delta)) / (2 * m)) ** 0.5
    fairness_epsilon = ((KL + math.log(2 * m_1 ** 0.5 / delta)) / (2 * m_1)) ** 0.5 + \
                       ((KL + math.log(2 * m_2 ** 0.5 / delta)) / (2 * m_2)) ** 0.5
    return [err - bnd_epsilon, err + bnd_epsilon], [fairness_err - fairness_epsilon, fairness_err + fairness_epsilon]



#### Fairness bounds


def ours_stochastic_fairness_bound(model, risk_A, risk_B, m_1, m_2, delta):
    upper_A = seeger_bound(m_1, model, risk_A, delta, coeff=1, verbose=False, monitor=None, lower=False)
    upper_B = seeger_bound(m_2, model, risk_B, delta, coeff=1, verbose=False, monitor=None, lower=False)
    lower_A = seeger_bound(m_1, model, risk_A, delta, coeff=1, verbose=False, monitor=None, lower=True)
    lower_B = seeger_bound(m_2, model, risk_B, delta, coeff=1, verbose=False, monitor=None, lower=True)
    return torch.max(upper_A-lower_B, upper_B-lower_A)

def oneto_fairness_bound(model, risk_A, risk_B, m_1, m_2, delta):
    KL = model.KL()
    const_A = np.log(2 * (m_1 ** 0.5) / delta)
    const_A = ((KL + const_A) / (2 * m_1)) ** 0.5
    const_B = np.log(2 * (m_2 ** 0.5) / delta)
    const_B = ((KL + const_B) / (2 * m_2)) ** 0.5
    return torch.abs((risk_A-risk_B))+const_A+const_B

BOUNDS = {
    "test": test_set_bound,
    "seeger": seeger_bound,
    "ours_stochastic_fairness": ours_stochastic_fairness_bound,
    "oneto_fairness": oneto_fairness_bound
}
