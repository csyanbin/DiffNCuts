import torch
import math
from scipy.sparse.linalg import eigsh

def uniform_solution_direction(u, u_ref=None, uniform_solution_method='positive'):
    batch, m, n = u.shape
    direction_factor = 1.0

    if uniform_solution_method != 'skip':
        if u_ref is None:
            u_ref = u.new_ones(1, m, 1).detach()

        direction = torch.einsum('bmk,bmn->bkn', u_ref, u)

        if u_ref.shape[2] == n:
            direction = torch.diagonal(direction, dim1=1, dim2=2).view(batch, 1, n)

        if uniform_solution_method == 'positive':
            direction_factor = (direction >= 0).float()
        elif uniform_solution_method == 'negative':
            direction_factor = (direction <= 0).float()

    u = u * (direction_factor - 0.5) * 2

    return u


class EigenDecompositionFcnFast(torch.autograd.Function):
    """PyTorch autograd function for eigen decomposition real symmetric matrices. Returns all eigenvectors
    or just eigenvectors associated with the top-k eigenvalues."""

    eps = 1.0e-9 # tolerance to consider two eigenvalues equal

    @staticmethod
    def forward(ctx, X, top_k=None):
        B, M, N = X.shape
        assert N == M
        assert (top_k is None) or (1 <= top_k <= M)

        with torch.no_grad():
            lmd, Y = torch.linalg.eigh(0.5 * (X + X.transpose(1, 2)))
            Y = uniform_solution_direction(Y)

        ctx.save_for_backward(lmd, Y)
        return lmd, Y if top_k is None else Y[:, :, 0:top_k]

    @staticmethod
    def backward(ctx, dJdLmd, dJdY):
        lmd, Y = ctx.saved_tensors
        B, M, K = dJdY.shape

        zero = torch.zeros(1, dtype=lmd.dtype, device=lmd.device)
        L = lmd[:, 0:K].view(B, 1, K) - lmd.view(B, M, 1)
        L = torch.where(torch.abs(L) < EigenDecompositionFcnFast.eps, zero, 1.0 / L)
        dJdX = torch.bmm(torch.bmm(Y, L * torch.bmm(Y.transpose(1, 2), dJdY)), Y[:, :, 0:K].transpose(1, 2))

        dJdX = 0.5 * (dJdX + dJdX.transpose(1, 2))

        return dJdX, None