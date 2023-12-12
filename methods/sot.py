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

    def forward(self, x: Tensor):
        # Normalize x
        V = x / x.norm(dim=1, keepdim=True)
        # Compute the cosine similarity matrix
        S = torch.mm(V, V.T)
        # Compute the squared Euclidean pairwise distance matrix, using the cosine similarity matrix
        D = 2 * (1 - S)
        # Set alpha_ in diagonal to act as infinite cost
        D.fill_diagonal_(self.alpha_)
        # Compute Sinkhorn in log-space
        log_W = self.sinkhorn_log(D, self.lambda_, self.n_iter)
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
