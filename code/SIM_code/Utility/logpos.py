import torch
from torch.autograd import Variable
import time

# import private library
from . import utils
from . import kronecker_operation
from . import distributions
from . import kernels
from . import settings

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 1., "beta_tilde_l": 1., "mu_tilde_sigma": 0., "alpha_tilde_sigma": 1., "beta_tilde_sigma": 1., "a": 1, "b": 1, "c": 10}


def vec2pars(pars, N, M):
    """
    convert parameter vector to parameters.
    :param pars: parameter vector
    :param N: number of inputs
    :param M: number of tasks
    :return: four vectorized parameters
    """
    tilde_l = pars[:N]
    tilde_sigma = pars[N:2 * N]
    L_vec = pars[2 * N:2 * N + int(M * (M + 1) / 2)]
    tilde_sigma2_err = pars[-1]
    return tilde_l, tilde_sigma, L_vec, tilde_sigma2_err


def vec2pars_SVC(pars, N, M):
    """
    covert parameter vector to parameters
    :param pars: parameter vector
    :param N: number of input vectors
    :param M: number of tasks
    :return: three vectorized parameters
    """
    tilde_l = pars[:N]
    L_vecs = pars[N: N + N * int(M * (M + 1) / 2)]
    tilde_sigma2_err = pars[-1]
    return tilde_l, L_vecs, tilde_sigma2_err


def vec2pars_S(pars, M):
    """
    convert parameter vector to parameters.
    :param pars: parameter vector
    :param M: number of tasks
    :return: four vectorized parameters
    """
    tilde_l = pars[0]
    tilde_sigma = pars[1]
    L_vec = pars[2: 2 + int(M * (M + 1) / 2)]
    tilde_sigma2_err = pars[-1]
    return tilde_l, tilde_sigma, L_vec, tilde_sigma2_err


def vec2pars_hadamard_SVC(pars, N, M):
    """
    covert parameter vector to parameters
    :param pars: parameter vector
    :param N: number of inputs
    :param M: number of tasks
    :return: three vectorized parameters
    """
    tilde_l = pars[:N]
    L_vecs = pars[N: N+N*int(M*(M+1)/2)]
    tilde_sigma2_err = pars[-1]
    return tilde_l, L_vecs, tilde_sigma2_err


def generate_vectorized_indexes(indx1, indx2):
    """
    generate vectorized indexes
    :param indx: 1d tensor with length N
    :return: 1d tensor with length N*N
    """
    N1 = indx1.size(0)
    N2 = indx2.size(0)
    indx1v = indx1.view(-1, 1).repeat(1, N2).view(-1)
    indx2v = indx2.repeat(N1)
    return indx1v.type(torch.LongTensor), indx2v.type(torch.LongTensor)


def generate_K_index(B_f, indx):
    """
    generate covaraince matrix of task for individuals
    :param B_f: task covariance matrix with dim M, M.
    :param indx: task index for individual with length N
    :return: 2d tensor with dim N, N
    """
    N = indx.size(0)
    indx1, indx2 = generate_vectorized_indexes(indx, indx)
    # print(indx1.type(), indx2.type())
    K_vec = B_f[indx1, indx2]
    return K_vec.view([N, N])


# def generate_K_index_SVC_old(L_f_list):
#     """
#     generate covariance matrix of task for individuals
#     :param L_f_list: L_f list with length N
#     :return: 2d tensor with dim MN, MN
#     """
#     K_i = torch.cat([torch.cat([torch.matmul(L_f_i, L_f_j.t()) for L_f_j in L_f_list], dim=1) for L_f_i in L_f_list], dim=0)
#     return K_i


def generate_K_index_SVC(L_f_list):
    """
    generate covariance matrix of task for individuals
    :param L_f_list: L_f list with length N
    :return: 2d tensor with dim MN, MN
    """
    L = torch.cat(L_f_list, dim=0)
    return L.mm(L.t())


def generate_K_index_SVC_hadamard0(L_f_list, indexes):
    # compute summary statistics
    L = torch.stack([L_f[index, :]  for L_f, index in zip(L_f_list, indexes)])
    return torch.mm(L, L.t())


def generate_K_index_SVC_hadamard(L_f_list, indexes):

    """
    generate covariance matrix of task for individuals
    :param L_f_list: L_f list with length N
    :param indx: task index for individual with length N
    :return: 2d tensor with dim N, N
    """
    K_i = torch.cat([torch.cat([torch.dot(L_f_i[index_i, :], L_f_j[index_j, :]).view([1,1]) for L_f_j, index_j in
                                zip(L_f_list, indexes)], dim=1) for L_f_i, index_i in zip(L_f_list, indexes)], dim=0)
    return K_i


def show_covs(pars, Y, x):
    """
    show covariance matrices
    :param pars: parameter vector
    :return: B_f, K_x, sigma2_err
    """
    N, M = Y.size()
    tilde_l, tilde_sigma, L_vec, tilde_sigma2_err = vec2pars(pars, N, M)

    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    print("B_f: {}".format(B_f))
    print("K_x: {}".format(K_x))
    print("sigma2_err: {}".format(sigma2_err))


def show_covs_hadamard(pars, x, indx):
    """
    show covariance matrix for
    :param pars: parameter vector
    :return: B_f, sigma2_err
    """
    N = x.size(0)
    M = torch.unique(indx).size(0)
    tilde_l, tilde_sigma, L_vec, tilde_sigma2_err = vec2pars(pars, N, M)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    sigma2_err = torch.exp(tilde_sigma2_err)
    print("B_f: {}".format(B_f))
    print("sigma2_err: {}".format(sigma2_err))


def deviance_obj(pars, Y, x):
    """
    Objective function w.r.t. deviance
    :param pars: parameters including tilde_l tilde_sigma, L_vec, tilde_sigma2_err
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :return: scalar tensor
    """
    N, M = Y.size()
    tilde_l, tilde_sigma, L_vec, tilde_sigma2_err = vec2pars(pars, N, M)
    return deviance(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, Y, x)


def deviance(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, Y, x):
    """
    Deviance fuction
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma_err: scalar tensor
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :return: scalar tensor
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    # Compute log likelihood
    invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    dev = -2*loglik
    return dev


def nlogpos_obj(pars, Y, x, mu_tilde_l=0., alpha_tilde_l=1., beta_tilde_l=1., mu_tilde_sigma=0., alpha_tilde_sigma=1., beta_tilde_sigma=1., a=1, b=1, c=10, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior.
    :param pars: parameters including tilde_l tilde_sigma, L_vec, tilde_sigma2_err
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l: hyper-paramters of GP for l.
    :param mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for sigma
    :param a, b: hyper-parameters for sigma2_err.
    :param c: hyper-parameters for Ls.
    :return: scalar tensor
    """
    N, M = Y.size()
    tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err = vec2pars(pars, N, M)
    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_uL_vec, log_prior_sigma2_err = logpos(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c, verbose, Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_uL_vec, log_prior_sigma2_err
    else:
        return -logpos(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c, verbose, Prior)


def logpos(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c, verbose=False, Prior=True):
    """
    Compute the log joint posterior distribution
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c: hyper-parameters
    :return: scalar tensor
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vec = utils.uLvec2Lvec(uL_vec, M)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    res = 0
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    # Compute log likelihood
    # invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    # logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    # loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # print(torch.symeig(K_x))
    # print(sigma)
    # print("K_x:", K_x, "B_f:", B_f, "K:", kronecker_operation.kronecker_product(B_f, K_x))
    loglik = distributions.multivariate_normal_logpdf0(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    while loglik != loglik:
        loglik = distributions.multivariate_normal_logpdf1(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    res += loglik
    # Compute log prob for tilde_l
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    # print(x, alpha_tilde_l, beta_tilde_l)
    # print(torch.symeig(Sigma_l))
    log_prior_tilde_l = MultivariateNormal(mu_tilde_l*torch.ones_like(x), covariance_matrix=Sigma_l).log_prob(tilde_l)
    if Prior:
        res += log_prior_tilde_l
    # Compute log prob for tilde sigma
    Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    log_prior_tilde_sigma = MultivariateNormal(mu_tilde_sigma*torch.ones_like(x), covariance_matrix=Sigma_sigma).log_prob(tilde_sigma)
    if Prior:
        res += log_prior_tilde_sigma
    # Compute log prob for L_vec
    log_prior_uL_vec = torch.sum(Normal(0, c).log_prob(uL_vec))
    if Prior:
        res += log_prior_uL_vec
    # Compute log prob for sigma2_err
    log_prior_sigma2_err = distributions.inverse_gamma_logpdf(sigma2_err, alpha=a, beta=b)
    # log_prior_sigma2_err = distributions.gamma_logpdf(sigma2_err, alpha=a, beta=b)
    if Prior:
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_uL_vec, log_prior_sigma2_err
    else:
        return res


def nlogpos_obj_SVC(pars, Y, x, mu_tilde_l=0., alpha_tilde_l=5., beta_tilde_l=1., mu_L=0., alpha_L=5., beta_L=1., a=1, b=1, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior for SVC version
    :param pars: parameters including tilde_l, uL_vec, ilde_sigma2_err
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l: hyper-parametes of GP for tilde_l
    :param mu_L, alpha_L, beta_L: hyper-parameters of GP for L_ij
    :param a, b: hyper-parameters for sigma2_err
    :return: scalar tensor
    """
    N, M = Y.size()
    tilde_l, uL_vecs, tilde_sigma2_err = vec2pars_SVC(pars, N, M)
    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err = logpos_SVC(tilde_l, uL_vecs,
                                                                                                    tilde_sigma2_err, Y,
                                                                                                    x, mu_tilde_l,
                                                                                                    alpha_tilde_l,
                                                                                                    beta_tilde_l, mu_L,
                                                                                                    alpha_L, beta_L, a,
                                                                                                    b, verbose, Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err
    else:
        return -logpos_SVC(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l,
                             mu_L, alpha_L, beta_L, a, b, verbose, Prior)


def logpos_SVC(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, a, b, verbose=False, Prior=True):
    """
    Compute the log joint posterior distribution
    :param tilde_l: 1d tensor with length N
    :param uL_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b: hyper-parameters
    :return: scalar tensor
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vecs = utils.uLvecs2Lvecs(uL_vecs, N, M)
    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n+1) * int(M * (M + 1) / 2)] for n in range(N)]
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    l = torch.exp(tilde_l)
    sigma2_err = torch.exp(tilde_sigma2_err)
    res = 0
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l) # dim: N, N
    K_i = generate_K_index_SVC(L_f_list) # dim: NM , NM
    neworder = torch.arange(N * M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder] # dim: MN. MN
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
    # print("K_x:", K_x, "K_i:", K_i, "K:", K)
    # Compute log likelihood
    invS = torch.inverse(K + sigma2_err * torch.eye(N*M).type(settings.torchType))
    logdetS = torch.logdet(K + sigma2_err * torch.eye(N*M).type(settings.torchType))
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    res += loglik
    # Compute log prob for tilde_l
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    log_prior_tilde_l = MultivariateNormal(mu_tilde_l*torch.ones_like(x), covariance_matrix=Sigma_l).log_prob(tilde_l)
    if Prior:
        res += log_prior_tilde_l
    # Compute log prob for uL_vecs
    Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    order_matrix = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
    
    log_prior_uL_vecs = torch.sum(torch.stack([MultivariateNormal(mu_L*torch.ones_like(x), covariance_matrix=Sigma_L).log_prob(uL_vecs[order_matrix[:, m]]) for m in range(int(M*(M+1)/2))]))
    #import pdb
    #pdb.set_trace()
    if Prior:
        res += log_prior_uL_vecs
    # Compute log prob for sigma2_err
    log_prior_sigma2_err = distributions.inverse_gamma_logpdf(sigma2_err, alpha=a, beta=b)
    # log_prior_sigma2_err = distributions.gamma_logpdf(sigma2_err, alpha=a, beta=b)
    if Prior:
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err
    else:
        return res


def nlogpos_obj_S(pars, Y, x, mu_tilde_l, sigma_tilde_l, a=1, b=1, c=10, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior.
    :param pars: parameters including tilde_l tilde_sigma, L_vec, tilde_sigma2_err
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l: hyper-paramters of GP for l.
    :param mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for sigma
    :param a, b: hyper-parameters for sigma2_err.
    :param c: hyper-parameters for Ls.
    :return: scalar tensor
    """
    N, M = Y.size()
    tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err = vec2pars_S(pars, M)

    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err = logpos_S(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, mu_tilde_l, sigma_tilde_l, a, b, c, verbose, Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err
    else:
        return -logpos_S(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, mu_tilde_l, sigma_tilde_l, a, b, c, verbose, Prior)


def logpos_S(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, mu_tilde_l, sigma_tilde_l, a, b, c, verbose=False, Prior=True):
    """
    Compute the log joint posterior distribution
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param uL_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma_err: scalar tensor
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c: hyper-parameters
    :return: scalar tensor
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vec = utils.uLvec2Lvec(uL_vec, M)
    # import pdb
    # pdb.set_trace()
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l * torch.ones(N).type(settings.torchType))
    sigma = torch.exp(tilde_sigma * torch.ones(N).type(settings.torchType))
    # print(tilde_l, tilde_sigma, l.dtype, sigma.dtype)
    sigma2_err = torch.exp(tilde_sigma2_err)
    res = 0
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    # print(torch.eig(K_x))
    # Compute log likelihood
    # invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    # logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    # loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    loglik = distributions.multivariate_normal_logpdf0(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    while loglik != loglik:
        loglik = distributions.multivariate_normal_logpdf1(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    # loglik = distributions.multivariate_normal_logpdf1(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    # print("woodbury", distributions.multivariate_normal_logpdf0(y, torch.zeros_like(y), B_f, K_x, sigma2_err))
    # print("robust", distributions.multivariate_normal_logpdf1(y, torch.zeros_like(y), B_f, K_x, sigma2_err))
    # print("real", distributions.multivariate_normal_logpdf2(y, torch.zeros_like(y), B_f, K_x, sigma2_err))
    # print(kronecker_operation.kronecker_product(B_f, K_x))
    res += loglik
    # Compute log prior for tilde_l
    if Prior:
        log_prior_tilde_l = Normal(mu_tilde_l, sigma_tilde_l).log_prob(tilde_l)
        res += log_prior_tilde_l
    # Compute log prior for L_vec
    if Prior:
        log_prior_uL_vec = torch.sum(Normal(0, c).log_prob(uL_vec))
        res += log_prior_uL_vec
    # Compute log prior for sigma2_err
    if Prior:
        log_prior_sigma2_err = distributions.inverse_gamma_logpdf(sigma2_err, alpha=a, beta=b)
        # log_prior_sigma2_err = distributions.gamma_logpdf(sigma2_err, alpha=a, beta=b)
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err
    else:
        return res


def nlogpos_obj_hadamard(pars, x, indx, y, mu_tilde_l=0., alpha_tilde_l=1., beta_tilde_l=1., mu_tilde_sigma=0., alpha_tilde_sigma=1., beta_tilde_sigma=1., a=1, b=1, c=10, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior.
    :param pars: parameters including tilde_l tilde_sigma, L_vec, tilde_sigma2_err
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l: hyper-paramters of GP for tilde_l.
    :param mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for sigma
    :param a, b: hyper-parameters for sigma2_err.
    :param c: hyper-parameters for Ls.
    :return: scalar tensor
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    tilde_l, tilde_sigma, L_vec, tilde_sigma2_err = vec2pars(pars, N, M)
    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_L_vec, log_prior_sigma2_err = logpos_hadamard(tilde_l,
                                                                                                              tilde_sigma,
                                                                                                              L_vec,
                                                                                                              tilde_sigma2_err,
                                                                                                               x, indx, y,
                                                                                                              mu_tilde_l,
                                                                                                              alpha_tilde_l,
                                                                                                              beta_tilde_l,
                                                                                                              mu_tilde_sigma,
                                                                                                              alpha_tilde_sigma,
                                                                                                              beta_tilde_sigma,
                                                                                                              a, b, c,
                                                                                                              verbose,
                                                                                                              Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_L_vec, log_prior_sigma2_err
    else:
        return -logpos_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, mu_tilde_l, alpha_tilde_l, beta_tilde_l,
                       mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c, verbose, Prior)


def logpos_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c, verbose=False, Prior=True):
    """
    Compute the log joint posterior distribution
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma_err: scalar tensor
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c: hyper-parameters
    :return: scalar tensor
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    res = 0
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    K_i = generate_K_index(B_f, indx)
    K = K_x * K_i

    # Compute log likelihood
    invS = torch.inverse(K + sigma2_err*torch.eye(N).type(settings.torchType))
    logdetS = torch.logdet(K + sigma2_err*torch.eye(N).type(settings.torchType))
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    # logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    # loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # print(torch.symeig(K_x))
    # loglik = distributions.multivariate_normal_logpdf0(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    res += loglik
    # Compute log prob for tilde_l
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    # print(x, alpha_tilde_l, beta_tilde_l)
    # print(torch.symeig(Sigma_l))
    log_prior_tilde_l = MultivariateNormal(mu_tilde_l * torch.ones_like(x), covariance_matrix=Sigma_l).log_prob(tilde_l)
    if Prior:
        res += log_prior_tilde_l
    # Compute log prob for tilde sigma
    Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    log_prior_tilde_sigma = MultivariateNormal(mu_tilde_sigma * torch.ones_like(x),
                                               covariance_matrix=Sigma_sigma).log_prob(tilde_sigma)
    if Prior:
        res += log_prior_tilde_sigma
    # Compute log prob for L_vec
    log_prior_L_vec = torch.sum(Normal(0, c).log_prob(L_vec))
    if Prior:
        res += log_prior_L_vec
    # Compute log prob for sigma2_err
    log_prior_sigma2_err = distributions.inverse_gamma_logpdf_u(sigma2_err, alpha=a, beta=b)
    if Prior:
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_L_vec, log_prior_sigma2_err
    else:
        return res


def nlogpos_obj_hadamard_SVC(pars, x, indx, y, mu_tilde_l=0., alpha_tilde_l=1., beta_tilde_l=1., mu_L=0., alpha_L=1., beta_L=1., a=1, b=1, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior for SVC version
    :param pars: parameters including tilde_l, L_vec, tilde_sigma2_err
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l: hyper-parametes of GP for tilde_l
    :param mu_L, alpha_L, beta_L: hyper-parameters of GP for L_ij
    :param a, b: hyper-parameters for sigma2_err
    :return: scalar tensor
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    tilde_l, L_vecs, tilde_sigma2_err = vec2pars_hadamard_SVC(pars, N, M)
    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_L_vecs, log_prior_sigma2_err = logpos_hadamard_SVC(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, a, b, verbose, Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_L_vecs, log_prior_sigma2_err
    else:
        return -logpos_hadamard_SVC(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, a, b, verbose, Prior)


def logpos_hadamard_SVC(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, a, b, verbose=False, Prior=True):
    """
    Compute the log joint posterior distribution
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma_err: scalar tensor
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b, c: hyper-parameters
    :return: scalar tensor
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    L_vec_list = [L_vecs[n*int(M*(M+1)/2): (n+1)*int(M*(M+1)/2)] for n in range(N)]
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    l = torch.exp(tilde_l)
    sigma2_err = torch.exp(tilde_sigma2_err)
    # print("step1")
    res = 0
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)
    # print("step1.1")
    # ts = time.time()
    K_i = generate_K_index_SVC_hadamard0(L_f_list, indx)
    # print("efficient step1.2 costs {}s".format(time.time()-ts))
    # print(K_i[0, :10])
    # ts = time.time()
    # K_i = generate_K_index_SVC_hadamard(L_f_list, indx)
    # print("step1.2 costs {}s".format(time.time()-ts))
    # print(K_i[0, :10])
    # K_i = generate_K_index(B_f, indx)
    K = K_x * K_i
    # print("step2")
    # Compute log likelihood
    invS = torch.inverse(K + sigma2_err * torch.eye(N).type(settings.torchType))
    logdetS = torch.logdet(K + sigma2_err * torch.eye(N).type(settings.torchType))
    # print("step3")
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    # logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    # loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # print(torch.symeig(K_x))
    # loglik = distributions.multivariate_normal_logpdf0(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    res += loglik
    # print("step4")
    # Compute log prob for tilde_l
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    log_prior_tilde_l = MultivariateNormal(mu_tilde_l*torch.ones_like(x), covariance_matrix=Sigma_l).log_prob(tilde_l)
    if Prior:
        res += log_prior_tilde_l
    # print("step5")
    # Compute log prob for L_vecs
    Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    order_matrix = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
    log_prior_L_vecs = torch.sum(torch.stack([MultivariateNormal(mu_L*torch.ones_like(x), covariance_matrix=Sigma_L).log_prob(L_vecs[order_matrix[:, m]]) for m in range(int(M*(M+1)/2))]))
    # import pdb
    # pdb.set_trace()
    if Prior:
        res += log_prior_L_vecs
    # print("step6")
    # Compute log prob for sigma2_err
    log_prior_sigma2_err = distributions.inverse_gamma_logpdf_u(sigma2_err, alpha=a, beta=b)
    if Prior:
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    # print("step7")
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_L_vecs, log_prior_sigma2_err
    else:
        return res


def nlogpos_obj_hadamard_S(pars, x, indx, y, mu_tilde_l, sigma_tilde_l, a=1, b=1, c=10, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior for stationary version
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    tilde_l, tilde_sigma, L_vec, tilde_sigma2_err = vec2pars_S(pars, M)
    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_L_vecs, log_prior_sigma2_err = logpos_hadamard_S(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, mu_tilde_l, sigma_tilde_l, a, b, c, verbose, Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_L_vecs, log_prior_sigma2_err
    else:
        return -logpos_hadamard_S(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, mu_tilde_l, sigma_tilde_l, a, b, c, verbose, Prior)


def logpos_hadamard_S(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, mu_tilde_l, sigma_tilde_l, a, b, c, verbose=False, Prior=True):
    N = y.size(0)
    M = torch.unique(indx).size(0)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    res = 0
    K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
    K_i = generate_K_index(B_f, indx)
    K = K_x * K_i

    # Compute log likelihood
    invS = torch.inverse(K + sigma2_err * torch.eye(N).type(settings.torchType))
    logdetS = torch.logdet(K + sigma2_err * torch.eye(N).type(settings.torchType))
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    # logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    # loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    # print(torch.symeig(K_x))
    # loglik = distributions.multivariate_normal_logpdf0(y, torch.zeros_like(y), B_f, K_x, sigma2_err)
    res += loglik
    # Compute log prob for tilde_l
    log_prior_tilde_l = Normal(loc=mu_tilde_l, scale=sigma_tilde_l).log_prob(tilde_l)
    if Prior:
        res += log_prior_tilde_l
    # Compute log prob for L_vec
    log_prior_L_vec = torch.sum(Normal(0, c).log_prob(L_vec))
    if Prior:
        res += log_prior_L_vec
    # Compute log prob for sigma2_err
    log_prior_sigma2_err = distributions.inverse_gamma_logpdf_u(sigma2_err, alpha=a, beta=b)
    if Prior:
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_L_vec, log_prior_sigma2_err
    else:
        return res


if __name__ == "__main__":
    # x = torch.randn(5)
    # Y = torch.randn([5, 2])
    # tilde_sigma2_err = torch.randn(1)[0]
    # tilde_l = torch.randn(5)
    # tilde_sigma = torch.randn(5)
    # L_vec = torch.randn(3)
    # pars = Variable(torch.cat([tilde_l, tilde_sigma, L_vec, tilde_sigma2_err.view(1)]), requires_grad=True)
    # # print(nlogpos_obj(pars, Y, x, **hyper_pars))
    # # print(logpos(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, Y, x))
    # Target = nlogpos_obj(pars, Y, x, **hyper_pars)
    # Target.backward()
    # # print("gradient of pars: ", pars.grad)
    # pars.grad.data.zero_()

    # dev = deviance_obj(pars, Y, x)
    # dev.backward()
    # print("gradient of pars: ", pars.grad)

    # indx = torch.LongTensor([0, 0, 1, 1])
    # print(generate_vectorized_indexes(indx))
    # B_f = torch.randn([2,2])
    # print(generate_K_index(B_f, indx))

    # check ver2pars_SVC
    N = 5
    M = 2
    x = torch.randn(5).type(settings.torchType)
    Y = torch.randn([5, 2]).type(settings.torchType)
    tilde_l = torch.randn(5).type(settings.torchType)
    l = torch.exp(tilde_l)
    L_vecs = torch.randn(N*int(M*(M+1)/2)).type(settings.torchType)
    tilde_sigma2_err = torch.randn(1).type(settings.torchType)[0]
    print(tilde_l, L_vecs, tilde_sigma2_err)
    pars = Variable(torch.cat([tilde_l, L_vecs, tilde_sigma2_err.view(1)]), requires_grad=True)
    tilde_l, L_vecs, tilde_sigma2_err = vec2pars_SVC(pars, N, M)
    print(tilde_l, L_vecs, tilde_sigma2_err)
    # check generate_K_index_SVC()
    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n+1) * int(M * (M + 1) / 2)] for n in range(N)]
    # print(M, L_vec_list)
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    # print(L_f_list)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)  # dim: N, N
    print(K_x.shape)
    K_i = generate_K_index_SVC(L_f_list)  # dim: NM , NM
    # reorder K_I
    neworder = torch.arange(N*M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder]
    print(neworder)
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
    print(K.shape)
