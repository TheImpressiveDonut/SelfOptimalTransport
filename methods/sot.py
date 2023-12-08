import ot
import torch
import torch.nn as nn


class Sot(nn.Module):
    def __init__(self, final_feat_dim, lambda_: float = 0.1, n_iter: int = 10) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.alpha_ = 1e9
        self.final_feat_dim = final_feat_dim

    def forward(self, x):
        # Normalize x
        V = x / x.norm(dim=1, keepdim=True)
        # Compute the cosine similarity matrix
        S = torch.mm(V, V.T)
        # Compute the squared Euclidean pairwise distance matrix, using the cosine similarity matrix
        D = 2 * (1 - S)
        # Set alpha_ in diagonal to act as infinite cost
        D.fill_diagonal_(self.alpha_)
        # Compute Sinkhorn
        sum_row_constraint = torch.ones_like(D)  # @todo check if need vector or matrix
        sum_col_constraint = torch.ones_like(D)
        W = ot.sinkhorn(sum_row_constraint, sum_col_constraint, D, 1 / self.lambda_, numItermax=self.n_iter)
        # Set 1 in diagonal (prob of similarity between x_i and x_i is 1)
        W.fill_diagonal_(1)
        return W
