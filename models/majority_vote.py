import torch

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):
    """
    The basic class for the majority vote.
    prior (torch.tensor of size <number of base classifiers>): prior weights for every base classifier.
    n_classes (int): number of classes for the given task.
    kl_factor (float): KL penalty factor.
    """
    def __init__(self, voters, prior, n_classes, distr_name, kl_factor):
        super(MajorityVote, self).__init__()
        
        if distr_name not in ["dirichlet", "gaussian", "categorical"]:
            raise NotImplementedError

        self.num_voters = len(prior)
        self.distribution_name = distr_name
        self.n_classes = n_classes

        # We inialize the current distribution.
        self.post = None
        self.distribution = distr_dict[distr_name](self.post, self.n_classes)

        # We initialize the distribution initial posterior, depending on the distribution name.
        if distr_name == "dirichlet":
            post = torch.rand(self.num_voters) * 2 + 1e-9  # uniform draws in (0, 2]
        else:
            post = torch.rand(self.num_voters) * 4 - 2  # uniform draws in [-2, 2]
        self.set_post(post)

        self.prior = prior
        self.voters = voters
        self.kl_factor = kl_factor

    def forward(self, x):
        return self.voters(x)

    def loss_per_base_classifier(self, batch, mean=True):
        return self.distribution.loss_per_base_classifier(batch, mean)

    def risk(self, batch, loss=None, mean=True):
        # If the loss function is given, then the deterministic risk is computed using this loss.
        #   Otherwise, the deterministic risk is computed.
        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.deterministic_risk(batch, mean)

    def KL(self):
        return self.kl_factor * self.distribution.KL(self.prior)

    def get_post(self):
        return self.distribution.get_post()

    def get_unchanged_post(self):
        return self.distribution.get_unchanged_post()

    def get_post_grad(self):
        return self.post.grad

    def set_post(self, value):
        assert len(value) == self.num_voters
        self.post = torch.nn.Parameter(value, requires_grad=True)
         
        if self.distribution_name == "categorical":
            self.distribution.theta = self.post
        elif self.distribution_name == "dirichlet":
            self.distribution.alpha = self.post
        elif self.distribution_name == "gaussian":
            self.distribution.w = self.post

    def random_draw_new_post(self):
        value = self.distribution.random_sample()
        self.set_post(value)
