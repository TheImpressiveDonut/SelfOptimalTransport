import torch.nn.functional as F
import ot
from torch import Tensor

class SelfOptimalTransport(MetaTemplate):

    def __init__(self, backbone, n_way, n_support, lambda_) -> None:
        super(SelfOptimalTransport, self).__init__(backbone, n_way, n_support)
        self.lambda_ = lambda_

    
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

    def forward(self, x: Tensor) -> Tensor:
        V = x / x.norm(dim=1, keepdim=True)
        D = 2 * (torch.ones((V.size(0), V.size(0)), device=V.device) - (torch.mm(V, V.T)))
        D[torch.eye(D.size(0)).bool()] = 1e3
        a = torch.ones(D.size(0), device=D.device)
        b = torch.ones(D.size(0), device=D.device)
        reg = 1 / self.lambda_
        return ot.sinkhorn(a=a, b=b, M=D, reg=reg)

    