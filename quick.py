from torch import Tensor
import ot
import torch

def forward(x: Tensor) -> Tensor:
    V = x / x.norm(dim=1, keepdim=True)
    D = 2 * (torch.ones((V.size(0), V.size(0)), device=V.device) - (torch.mm(V, V.T)))
    D[torch.eye(D.size(0)).bool()] = 1e3
    a = torch.ones(D.size(0), device=D.device)
    b = torch.ones(D.size(0), device=D.device)
    reg = 1 / 0.1
    return ot.sinkhorn(a=a, b=b, M=D, reg=reg, numItermax=1000000)

if __name__ == '__main__':
    x = torch.rand(10, 50)
    F = forward(x)
    print(F.size())

#
#
#