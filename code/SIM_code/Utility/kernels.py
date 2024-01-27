import torch
from . import settings


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def RBF_cov(X1, X2=None, alpha=1., beta=1.):
    """
    covarince matrix generated from RBF kernel with parameter alpha and beta
    :param X1: 2d tensor with dim N1 by M
    :param X2: 2d tensor with dim N2 by M
    :param alpha: scale parameter
    :param beta: lengthscale parameter
    :return: 2d tensor with dim N1 by N2
    """
    if X2 is None:
        X2 = X1
        cov_mat = torch.eye(X1.size(0)).type(settings.torchType)*settings.jitter
    else:
        cov_mat = torch.zeros(X1.size(0), X2.size(0)).type(settings.torchType)
    X1_std = X1/beta
    X2_std = X2/beta
    # print(X1_std.type(), X2_std.type())
    dist = pairwise_distances(X1_std, X2_std)
    cov_mat += torch.exp(-0.5*dist)*alpha**2
    return cov_mat


def Nonstationary_RBF_cov(X1, sigma1=None, ell1=None, X2=None, sigma2=None, ell2=None):
    """
    convariance matrix generated from nonstationary RBF kernel with parameter sigma and ell
    :param X1: 2d tensor with dim N1 by M
    :param sigmai: 1d tensor with length as Ni
    :param elli: 1d tensor with length as Ni
    :param X2: 2d tensor with dim N2 by M
    :return: scalar tensor
    """
    N1 = X1.size(0)
    if sigma1 is None:
        sigma1 = torch.ones(N1).type(settings.torchType)
    if ell1 is None:
        ell1 = torch.ones(N1).type(settings.torchType)
    if X2 is None:
        X2 = X1
        sigma2 = sigma1
        ell2 = ell1
        cov_mat = torch.eye(X1.size(0)).type(settings.torchType)*settings.jitter
    else:
        cov_mat = torch.zeros(X1.size(0), X2.size(0)).type(settings.torchType)
    N2 = X2.size(0)
    dist = pairwise_distances(X1, X2)
    A = (ell1**2).view([-1,1]).repeat([1,N2]) + (ell2**2).view([1,-1]).repeat([N1,1])
    B = ell1.view([-1,1]).repeat([1,N2]) * ell2.view([1,-1]).repeat([N1,1])
    C = sigma1.view([-1,1]).repeat([1,N2]) * sigma2.view([1,-1]).repeat([N1,1])
    cov_mat += C*torch.sqrt(2.*B/A)*torch.exp(-dist/A)
    return cov_mat


if __name__ == "__main__":
    # X1 = torch.randn([2,2])
    # X2 = torch.randn([3,2])
    # print("Covariance matrix between X1 and X2 via RBF: {}".format(RBF_cov(X1, X2)))
    # print("Covaraince matrix between X1 and X1 via RBF: {}".format(RBF_cov(X1)))

    X1 = torch.randn([2,2])
    X2 = torch.randn([3,2])
    ell1 = torch.exp(torch.randn(2))
    ell2 = torch.exp(torch.randn(3))
    sigma1 = torch.exp(torch.randn(2))
    sigma2 = torch.exp(torch.randn(3))

    print("Covariance matrix between X1 and X2 via nonstationary RBF: {}".format(Nonstationary_RBF_cov(X1, ell1, sigma1, X2, ell2, sigma2)))
    print("Covariance matrix between X1 and X1 via nonstationary RBF: {}".format(Nonstationary_RBF_cov(X1, ell1, sigma1)))
