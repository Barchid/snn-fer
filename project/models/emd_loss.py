import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange


class EmdLoss(nn.Module):
    """Some Information about EmdLoss"""

    def __init__(self):
        super(EmdLoss, self).__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        return self._emd_loss(pred, gt)

    def _emd_loss(self, a: torch.Tensor, b: torch.Tensor):
        """Implementation of the efficient way of calculating the restricted EMD distance
        (shown in paper https://www.frontiersin.org/articles/10.3389/fncom.2019.00082/full)

        Args:
            a (torch.Tensor): dim=(T,B,C) the spike train a
            b (torch.Tensor): dim=(T,B,C) the spike train b
        """
        # normalize spike trains over spike numbers
        n_a = torch.count_nonzero(a, dim=0)
        a = a / (n_a + 1e-5)  # normalize w/ epsilon
        n_b = torch.count_nonzero(b, dim=0)
        b = b / (n_b + 1e-5)

        # cumsum
        cum_a = torch.cumsum(a, 0)  # cumulative function of spike train a
        cum_b = torch.cumsum(b, 0)  # cumulative function of spike train b

        # |elementiwe difference|
        diff = torch.abs(cum_a - cum_b)
        summation = torch.sum(diff, dim=0)

        # sum is the result
        return torch.mean(summation)


def emd(a: torch.Tensor, b: torch.Tensor):
    """Implementation of the efficient way of calculating the restricted EMD distance
    (shown in paper https://www.frontiersin.org/articles/10.3389/fncom.2019.00082/full)

    Args:
        a (torch.Tensor): dim=(T,B,C) the spike train a
        b (torch.Tensor): dim=(T,B,C) the spike train b
    """
    # normalize spike trains over spike numbers
    n_a = torch.count_nonzero(a, dim=0)
    a = a / (n_a + 1e-5)  # normalize w/ epsilon
    n_b = torch.count_nonzero(b, dim=0)
    b = b / (n_b + 1e-5)

    # cumsum
    cum_a = torch.cumsum(a, 0)  # cumulative function of spike train a
    cum_b = torch.cumsum(b, 0)  # cumulative function of spike train b

    # |elementiwe difference|
    diff = torch.abs(cum_a - cum_b)
    summation = torch.sum(diff, dim=0)

    # sum is the result
    return torch.mean(summation)


def naive_popu_emd(A, B):
    # A,B \in dim (T,B,C)
    P = A.shape[-1]

    emds = torch.zeros(P)
    for p in range(P):
        a = A[
            :,
            :,
        ]  # (T,B,C)
        b = B[p]
        per_neuron_emd = emd(a, b)
        emds[p] = per_neuron_emd

    return emds.mean()


def naive_loss(output: torch.Tensor, C: int, P: int):
    # output dim=(T, B, CxP)
    populations = rearrange(output, "T B (C P) -> T B C P", C=C, P=P)
    vals = torch.zeros(C)
    done = []
    i = 0
    for c1 in range(C):
        for c2 in range(C):
            if c1 == c2:
                continue

            if (f"{c1}{c2}" in done) or (f"{c2}{c1}" in done):
                continue

            popu1 = populations[:, :, c1, :]
            popu2 = populations[:, :, c2, :]
            val = naive_popu_emd(popu1, popu2)
            done.append(f"{c1}{c2}")
            done.append(f"{c2}{c1}")
            vals[i] = val
            i += 1

    return torch.tensor(vals).mean()
