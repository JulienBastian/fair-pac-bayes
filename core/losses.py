import torch

from core.expected_risk import gaussian_cdf_precomputations, BetaInc
from core.utils import Phi


def initialize_risk(cfg, n_classes):
    """
    Initialize the loss function, the bound coefficient, kl factor and divergence type given the considered risk.
    """
    dst = cfg.training.distribution
    base_loss = lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, dst, n_classes)
    return base_loss, 1., 1.


def deterministic_loss(y_target, y_pred, theta, distribution, n_classes):
    """
    Compute the loss of the corresponding deterministic classifier.
    """
    if len(theta.shape) == 1:
        # Deterministic prediction for a majority vote.
        if n_classes == 2:
            y_pred = (y_pred + 1) / 2
            y_target = (y_target + 1) / 2
        y_pred_oh = torch.nn.functional.one_hot(y_pred.to(torch.long), n_classes)
        y_target = y_target.to(torch.long)
        weighted_preds = y_pred_oh.transpose(1, 2) * theta
        summed_preds = torch.sum(weighted_preds, dim=2)
        if n_classes == 2:
            if distribution == "gaussian":
                return torch.nn.BCELoss(reduction='none')(torch.sigmoid(summed_preds[:, 1]), y_target.squeeze().to(torch.float))
            return torch.nn.BCELoss(reduction='none')(summed_preds[:, 1], y_target.squeeze().to(torch.float))
        if distribution == "gaussian":
            return torch.nn.CrossEntropyLoss(reduction='none')(torch.sigmoid(summed_preds), y_target.squeeze())
        return torch.nn.CrossEntropyLoss(reduction='none')(summed_preds, y_target.squeeze())

def moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes):
    """
    Moment loss (see the article). Corresponds to the "standard" stochastic loss.
    """
    base_value = torch.zeros(1)
    if y_target.is_cuda:
        base_value = base_value.cuda()
    if distribution == 'categorical':
        return torch.where(y_target != y_pred, theta, base_value).sum(1)
    elif distribution == 'dirichlet':
        correct = torch.where(y_target == y_pred, theta, base_value).sum(1)
        wrong = torch.where(y_target != y_pred, theta, base_value).sum(1)
        return [BetaInc.apply(c, w, torch.tensor(0.5)) for c, w in zip(correct, wrong)]
    elif distribution == 'gaussian':
        if n_classes == 2:
            inner_Phi = (torch.squeeze(y_target) * torch.sum(torch.reshape(theta,(1, -1)) * y_pred, dim=1)) /\
                        torch.sum(y_pred ** 2, dim=1) ** 0.5
            return Phi(inner_Phi)
        else:
            return gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, predictor)
