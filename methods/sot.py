import ot
import torch
import torch.nn as nn
from torch import Tensor


class Sot(nn.Module):
    def __init__(self, final_feat_dim: int, lambda_: float = 0.1, n_iter: int = 20) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.alpha_ = 1e5
        self.final_feat_dim = final_feat_dim
        self.paper_sot = PaperSOT(lambda_, n_iter)

    def forward(self, x: Tensor):
        if True:
            return self.paper_sot.call(x)
        else:
            # Normalize x
            V = x / x.norm(dim=1, keepdim=True)
            # Compute the cosine similarity matrix
            S = torch.mm(V, V.T)
            # Compute the squared Euclidean pairwise distance matrix, using the cosine similarity matrix
            D = 2 * (1 - S)
            # Set alpha_ in diagonal to act as infinite cost
            D.fill_diagonal_(self.alpha_)
            # Compute Sinkhorn in log-space
            #log_W = self.sinkhorn_log(D, self.lambda_, self.n_iter)
            log_W = ot.bregman.sinkhorn_log(torch.ones(D.size(0), device=D.device), torch.ones(D.size(0), device=D.device), D, 1/self.lambda_, numItermax=self.n_iter) # https://pythonot.github.io/gen_modules/ot.bregman.html#id108
            W = log_W.exp()
            # Set 1 in diagonal (prob of similarity between x_i and x_i is 1)
            W_clone = W.clone() # clone because of gradient, error on inplace operation
            W_clone.fill_diagonal_(1)
            W = W_clone
            return W


    def sinkhorn_log(self, D: Tensor, lambda_: float, num_iters: int) -> Tensor:
        log_k = -D * lambda_
        log_v = torch.zeros(D.size(1), device=D.device)  # constraint vector of all ones but to log so zeros

        for i in range(num_iters):
            log_u = -torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
            log_v = -torch.logsumexp(log_u.unsqueeze(-1) + log_k, dim=0)

        return log_u.unsqueeze(-1) + log_k + log_v.unsqueeze(0)



class PaperSOT(object):

    def __init__(self, lambda_: float = 0.1, n_iter: int = 10):
        super().__init__()
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.diagonal_val = 1e3

    def compute_cost(self, X: torch.Tensor) -> torch.Tensor:
        return 1 - self.cosine_similarity(X)

    def mask_diagonal(self, M: torch.Tensor, value: float):
        M_clone = M.clone()
        M_clone.fill_diagonal_(value)
        return M_clone

    def call(self, X: torch.Tensor) -> torch.Tensor:
        # get masked cost matrix
        C = self.compute_cost(X=X)
        M = self.mask_diagonal(C, value=self.diagonal_val)

        # compute self-OT
        z_log = self.log_sinkhorn(M=M, reg=self.lambda_, num_iters=self.n_iter)

        z = torch.exp(z_log)

        # set self-values to 1
        return self.mask_diagonal(z, value=1)

    def cosine_similarity(self, a: torch.Tensor):
        d_n = a / a.norm(dim=-1, keepdim=True)
        C = torch.mm(d_n, d_n.transpose(0, 1))
        return C

    def log_sum_exp(self, u: torch.Tensor, dim: int):
        u_max, __ = u.max(dim=dim, keepdim=True)
        log_sum_exp_u = torch.log(torch.exp(u - u_max).sum(dim)) + u_max.sum(dim)
        return log_sum_exp_u


    def log_sinkhorn(self, M: torch.Tensor, reg: float, num_iters: int):
        # Initialize dual variable v (u is implicitly defined in the loop)
        log_v = torch.zeros(M.size()[1]).to(M.device)  # ==torch.log(torch.ones(m.size()[1]))

        # Exponentiate the pairwise distance matrix
        log_K = -M / reg

        # Main loop
        for i in range(num_iters):
            # Match r marginals
            log_u = - self.log_sum_exp(log_K + log_v[None, :], dim=1)

            # Match c marginals
            log_v = - self.log_sum_exp(log_u[:, None] + log_K, dim=0)

        # Compute optimal plan, cost, return everything
        log_P = log_u[:, None] + log_K + log_v[None, :]
        return log_P