import torch
import numpy as np
import time
from scipy.special import gammaln as lgamma

from . import kronecker_operation
from . import settings


def multivariate_normal_logpdf(y, mu, logdetSigma, invSigma):
    """
    Compute the un-normalized log pdf of multivariate normal distribution with parameter mu and Sigma.
    :param y: 1d tensor with length N
    :param mu: 1d tensor with length N
    :param logdetSigma: scalar tensor
    :param invSigma: 2d tenosr with dim N by N
    :return: scalar tensor
    """
    N = y.size()[0]
    y_bar = y - mu
    res = -0.5*N*(np.log(2.*np.pi)) - 0.5*logdetSigma - 0.5*torch.dot(y_bar, torch.mv(invSigma, y_bar))
    res = - 0.5*logdetSigma - 0.5*torch.dot(y_bar, torch.mv(invSigma, y_bar))
    return res


def multivariate_normal_logpdf0(y, mu, B, K, sigma2):
    """
    Efficiently compute the un-normalized log pdf of multivariate normal distribution with mean mu and covariance matrix B \otimes K + sigma2I
    :param y: 1d tensor with length NM
    :param mu: 1d tensor with length NM
    :param B: 2d tensor with dim M, M
    :param K: 2d tensor with dim N, N
    :param sigma2: scalar tensor
    :return: scalar tensor
    """
    # eigendecomposition
    w_B, v_B = torch.symeig(B, eigenvectors=True)
    # for the robustness of eigen-decomposition of K, we add jitters to K to avoid the same eigen values.
    # K += torch.diag(torch.rand(K.size(0)).type(settings.torchType)*settings.precision)
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    # compute U^Ty where U = v_B \otimes v_K
    tilde_y = y - mu
    a = kronecker_operation.kron_mv(v_B.t(), v_K.t(), tilde_y)
    # compute (sigma2I + T)^{-1}
    t = kronecker_operation.kronecker_product_diag(w_B, w_K)
    w = 1./(sigma2 + t)
    # compute y^T(sigma2I + B \times K)y
    b = torch.dot(a*w, a)
    # compute logdet(B \times T + sigma2I)
    c = torch.log(t + sigma2).sum()
    res = -0.5*c - 0.5*b
    return res


def multivariate_normal_logpdf1(y, mu, B, K, sigma2): # robust version
    """
    Efficiently compute the un-normalized log pdf of multivariate normal distribution with mean mu and covariance matrix B \otimes K + sigma2I
    :param y: 1d tensor with length NM
    :param mu: 1d tensor with length NM
    :param B: 2d tensor with dim M, M
    :param K: 2d tensor with dim N, N
    :param sigma2: scalar tensor
    :return: scalar tensor
    """
    # eigendecomposition
    B = B + torch.diag(torch.rand(B.size(0)).type(settings.torchType)*settings.precision)
    w_B, v_B = torch.symeig(B, eigenvectors=True)
    # for the robustness of eigen-decomposition of K, we add jitters to K to avoid the same eigen values.
    K = K + torch.diag(torch.rand(K.size(0)).type(settings.torchType)*settings.precision)
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    # more robust approach
    # K1 = K.detach()
    # eigvalng, eigvecng = torch.symeig(K1, eigenvectors=True)
    # np.random.seed(random_seed)
    # eps_eigvalues_array = np.sort(np.random.randn(K.size(0)))
    # eps_eigvalues = torch.from_numpy(eps_eigvalues_array).type(settings.torchType)
    # eps_matrix = torch.diag(eps_eigvalues*1)
    # eps_matrix = eigvecng.mm(eps_matrix)
    # eps_matrix = eps_matrix.mm(torch.transpose(eigvecng,0,1))
    # w_K_eps, v_K_eps = torch.symeig(K + eps_matrix, eigenvectors=True)
    # w_K = w_K_eps - eps_eigvalues
    # v_K = v_K_eps 

    # print(w_K, v_K.shape)
    # compute U^Ty where U = v_B \otimes v_K
    tilde_y = y - mu
    a = kronecker_operation.kron_mv(v_B.t(), v_K.t(), tilde_y)
    # compute (sigma2I + T)^{-1}
    t = kronecker_operation.kronecker_product_diag(w_B, w_K)
    w = 1./(sigma2 + t)
    # compute y^T(sigma2I + B \times K)y
    b = torch.dot(a*w, a)
    # compute logdet(B \times T + sigma2I)
    c = torch.log(t + sigma2).sum()
    res = -0.5*c - 0.5*b
    return res


def multivariate_normal_logpdf2(y, mu, B, K, sigma2):
    """
    Efficiently compute the un-normalized log pdf of multivariate normal distribution with mean mu and covariance matrix B \otimes K + sigma2I
    :param y: 1d tensor with length NM
    :param mu: 1d tensor with length NM
    :param B: 2d tensor with dim M, M
    :param K: 2d tensor with dim N, N
    :param sigma2: scalar tensor
    :return: scalar tensor
    """
    Sigma = kronecker_operation.kronecker_product(B, K) + sigma2*torch.eye(B.size(0)*K.size(0)).type(settings.torchType)
    logdetSigma = torch.logdet(Sigma)
    invSigma = torch.inverse(Sigma)
    res = multivariate_normal_logpdf(y, mu, logdetSigma, invSigma)
    return res


def inverse_gamma_logpdf_u(x, alpha=1., beta=1.): # ignore constant and treat alpha and beta known
    """
    Compute the un-normalized log pdf of inverse gamma distribution with parameter a and b.
    :param x: scalar tensor
    :param a: scale parameter
    :param b: rate parameter
    :return: scalar tensor
    """
    return (-alpha-1)*torch.log(x)-beta/x

def inverse_gamma_logpdf(x, alpha=1., beta=1.): # ignore constant and treat alpha and beta known
    """
    Compute the un-normalized log pdf of inverse gamma distribution with parameter a and b.
    :param x: scalar tensor
    :param a: scale parameter
    :param b: rate parameter
    :return: scalar tensor
    """
    return (-alpha-1)*torch.log(x)-beta/x+alpha*np.log(beta)-lgamma(alpha)

def gamma_logpdf(x, alpha=1., beta=1.):
    return (alpha-1)*torch.log(x)-beta*x+alpha*np.log(beta)-lgamma(alpha)

if __name__ == "__main__":
    # L = torch.randn([2, 2])
    # S = torch.mm(L, L.t())
    # logdetS = torch.logdet(S)
    # invS = torch.inverse(S)
    # y = torch.randn(2)
    # mu = torch.zeros(2)
    # print("logpdf: {}".format(multivariate_normal_logpdf(y, mu, logdetS, invS)))

    # from torch.distributions.multivariate_normal import MultivariateNormal
    # mvn = MultivariateNormal(mu, covariance_matrix=S)
    # print("logpdf: {}".format(mvn.log_prob(y)))

    # x = torch.exp(torch.randn(1))
    # print(inverse_gamma_logpdf_u(x))

    y = torch.randn(4)
    mu = torch.zeros(4)
    L_B = torch.randn([2, 2])
    B = torch.mm(L_B, L_B.t())
    L_K = torch.randn([2, 2])
    K = torch.mm(L_K, L_K.t())
    sigma2 = torch.exp(torch.randn(1))[0]
    ts = time.time()
    print(multivariate_normal_logpdf0(y, mu, B, K, sigma2))
    print("it costs {}s".format(time.time()-ts))
    ts = time.time()
    invS = kronecker_operation.kron_inv(sigma2, B, K)
    logdetS = kronecker_operation.kron_logdet(sigma2, B, K)
    print(multivariate_normal_logpdf(y, mu, logdetS, invS))
    print(time.time()-ts)
