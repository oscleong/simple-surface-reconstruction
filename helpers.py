import numpy as np
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import ot

def pairwise_distance_matrix(X, Y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    X_norm = (X**2).sum(1).view(-1, 1)
    if Y is not None:
        Y_t = torch.transpose(Y, 0, 1)
        Y_norm = (Y**2).sum(1).view(1, -1)
    else:
        Y_t = torch.transpose(X, 0, 1)
        X_norm = X_norm.view(1, -1)
    
    Dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y_t)
    # Ensure diagonal is zero if X=Y
    # if Y is None:
    #     Dist = Dist - torch.diag(Dist.diag)
    return Dist #torch.clamp(Dist, 0.0, np.inf)

def project_stochastic_to_perm(P):
    max_in_rows = np.amax(P, axis = 1)
    max_list = [np.where(P == max_in_rows[ii]) for ii in range(P.shape[0])]
    for ii in range(len(max_list)):
        P[max_list[ii]] = 1
    Omega = np.array((P == 1))
    return np.multiply(Omega, P)

def get_permutation_matrix(a, b, C, maxiter, lam, method):
    if method == 'EMD':
        P = ot.emd(a, b, C, numItermax=maxiter, log=False)
    elif method == 'Sinkhorn':
        P = ot.sinkhorn(a, b, C, lam, method='sinkhorn', numItermax=maxiter, stopThr=1e-09)
    return P

def get_permutation_btwn_parametric(P1, P2_inv):
    return Variable(torch.from_numpy(np.matmul(P1, P2_inv)), requires_grad = False)