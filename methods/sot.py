import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from torch import Tensor

class SelfOptimalTransport(MetaTemplate):

    def __init__(self, backbone, n_way, n_support, lambda_) -> None:
        super(SelfOptimalTransport, self).__init__(backbone, n_way, n_support)
        self.sot = Sot(lambda_)

    
    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x, y):
        scores = self.forward(x)
        print(scores.shape)
        if self.type == 'classification':
            y = y.long().cuda()
        else:
            y = y.cuda()

        return self.loss_fn(scores, y)

    def forward(self, x):
        return self.sot(x)

    
class Sot(nn.Module):
    def __init__(self, lambda_=0.1, n_iter=10):
        super().__init__()
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.alpha_ = 1e9

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
        sum_row_constraint = torch.ones_like(D)
        sum_col_constraint = torch.ones_like(D)
        W = ot.sinkhorn(sum_row_constraint, sum_col_constraint, D, 1/self.lambda_, numItermax=self.n_iter)
        # Set 1 in diagonal (prob of similarity between x_i and x_i is 1)
        W.fill_diagonal_(1)
        return W