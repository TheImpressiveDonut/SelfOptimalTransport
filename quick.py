from torch import Tensor, nn
import ot
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


def T(x: Tensor) -> Tensor:
    V = x / x.norm(dim=1, keepdim=True)
    D = 2 * (torch.ones((V.size(0), V.size(0)), device=V.device) - (torch.mm(V, V.T)))
    D[torch.eye(D.size(0)).bool()] = 1e9
    a = torch.ones(D.size(0), device=D.device)
    b = torch.ones(D.size(0), device=D.device)
    reg = 1 / 0.01
    return ot.sinkhorn(a=a, b=b, M=D, reg=reg, method='sinkhorn_log', numItermax=1000000)


if __name__ == '__main__':
    x = torch.rand(10, 50)
    target = torch.empty(10).random_(2)


    feature_extractor = nn.Linear(50, 100)
    classifier = nn.Linear(10, 1)

    opt = Adam(feature_extractor.parameters(), lr=1e-3)
    loss_fn = BCEWithLogitsLoss()

    opt.zero_grad()
    F = classifier(T(feature_extractor(x)))
    loss = loss_fn(F.squeeze(-1), target)
    loss.backward()
    opt.step()

    print(loss.grad_fn)