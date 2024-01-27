import torch
from torch.distributions.normal import Normal
import numpy as np
import time

# import private library
from . import kernels
from . import kronecker_operation
from . import utils
from . import settings
from . import logpos


def vec2pars(pars_hist, N, M):
    """
    Convert pars_hist to tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist
    :param pars_hist:
    :return:
    """
    tilde_l_hist = pars_hist[:, :N]
    tilde_sigma_hist = pars_hist[:, N:2*N]
    L_vec_hist = pars_hist[:, 2*N:2*N+int(M*(M+1)/2)]
    tilde_sigma2_err_hist = pars_hist[:, -1]
    return tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist

def vec2list(y_vec, indx):
    """convert vector to list based on index"""
    M = np.unique(indx).shape[0]
    y_list = [y_vec[indx == i] for i in range(M)]
    return y_list

######################SNMGP############################
#### Sampling
def point_predsample(tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, N_sample, *args, **kwargs):
    """
    Sample the posterior predictor at new input x_star
    :param tilde_l_hist: 2d tensor with dim N_hist by N
    :param tilde_sigma_hist: 2d tensor with dim N_hist by N
    :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 2d tensor with length N_hist by M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    sampled_y_hist = []
    tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist = tilde_l_hist[-N_sample:], tilde_sigma_hist[-N_sample:], uL_vec_hist[-N_sample:], tilde_sigma2_err_hist[-N_sample:]
    for tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err in zip(tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist):
        # sample posterior of tilde_l_star
        L_vec = utils.uLvec2Lvec(uL_vec, M)
        Sigma_l = kernels.RBF_cov(x.view([-1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
        k_l = kernels.RBF_cov(x.view([-1,1]), x_star.view(1,1), alpha=alpha_tilde_l, beta= beta_tilde_l)
        proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
        mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l-mu_tilde_l))
        sigma2_l = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0,0] - torch.dot(proj_l, k_l.view(-1))
        if sigma2_l < 0:
            sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample()

        # sample posterior of tilde_sigma_star
        Sigma_sigma = kernels.RBF_cov(x.view([-1,1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        k_sigma = kernels.RBF_cov(x.view([-1,1]), x_star.view(1,1), alpha=alpha_tilde_sigma, beta= beta_tilde_sigma)
        proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
        mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma-mu_tilde_sigma))
        sigma2_sigma = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)[0,0] - torch.dot(proj_sigma, k_sigma.view(-1))
        if sigma2_sigma < 0:
            sigma2_sigma = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_sigma_star = Normal(loc=mu_sigma, scale=torch.sqrt(sigma2_sigma)).sample()

        # sample posterior of y
        sigma2_err = torch.exp(tilde_sigma2_err)
        l = torch.exp(tilde_l)
        sigma = torch.exp(tilde_sigma)
        sampled_l_star = torch.exp(sampled_tilde_l_star)
        sampled_sigma_star = torch.exp(sampled_tilde_sigma_star)
        L = utils.vec2lowtriangle(L_vec, M)
        B_f = torch.mm(L, L.t())
        K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)

        # First method
        # ts = time.time()
        # eigendecomposition
        w_B, v_B = torch.symeig(B_f, eigenvectors=True)
        w_K, v_K = torch.symeig(K_x, eigenvectors=True)
        k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1,1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]), sigma2=sampled_sigma_star.view(-1), ell2=sampled_l_star.view(-1))
        k_f = kronecker_operation.kronecker_product(B_f, k_x)
        t = kronecker_operation.kronecker_product_diag(w_B, w_K)
        w = 1./(sigma2_err + t)
        A = []
        for m in range(M):
            A.append(kronecker_operation.kron_mv(v_B.t(), v_K.t(), k_f[:, m]))
        A = torch.stack(A)
        b = kronecker_operation.kron_mv(v_B.t(), v_K.t(), y)
        mu_f = torch.mv(A, b*w)
        a2 = torch.diag(kronecker_operation.kronecker_product(B_f, kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=sampled_sigma_star.view(-1), ell1=sampled_l_star.view(-1))))
        # print(a2.size(), A.size(), w.size())
        sigma2_f = a2 - (A*w*A).sum(dim=1)
        sigma2_y = sigma2_f + sigma2_err
        # print(time.time()-ts)

        # # Second method
        # ts = time.time()
        # invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
        # k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
        #                                     sigma2=sampled_sigma_star.view(-1), ell2=sampled_l_star.view(-1))
        # k_f = kronecker_operation.kronecker_product(B_f, k_x)
        # mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
        # invL = torch.cholesky(invS)
        # T = torch.mm(k_f.t(), invL)
        # A = kronecker_operation.kronecker_product(B_f, kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=sampled_sigma_star.view(-1), ell1=sampled_l_star.view(-1)))
        # B = torch.mm(T, T.t())
        # Sigma_f = A - B
        # Sigma_y = Sigma_f + sigma2_err * torch.eye(M)
        # sigma2_y = torch.diagonal(Sigma_y)
        # print(time.time() - ts)

        # clip the sigma2_y.
        sigma2_y[sigma2_y <= 0] = settings.precision
        sampled_y = Normal(loc=mu_f, scale=torch.sqrt(sigma2_y)).sample()

        if sampled_y[0] != sampled_y[0]:
            import pdb
            pdb.set_trace()
        # record
        # import pdb
        # pdb.set_trace()
        sampled_y_hist.append(sampled_y)
    return torch.stack(sampled_y_hist)

def pointwise_predsample(tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist, Y, x, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, N_sample, *args, **kwargs):
    """
    Sample the posterior predictor at grids
    :param tilde_l_hist: 2d tensor with dim N_hist by N
    :param tilde_sigma_hist: 2d tensor with dim N_hist by N
    :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with length N_grid by N_hist by M
    """
    res = []

    # import pdb
    # pdb.set_trace()
    # point_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, Y, x, torch.tensor(0.58), mu_tilde_l,
    #                  alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
    for grid in grids:
        print(grid)
        sampled_y_hist = point_predsample(tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, N_sample, *args, **kwargs)
        res.append(sampled_y_hist)
    res = torch.stack(res)
    return res.numpy()

def test_predsample(tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist, Y, x, x_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, N_sample, *args, **kwargs):
    """
    Sample the posterior predictor at grids
    :param tilde_l_hist: 2d tensor with dim N_hist by N
    :param tilde_sigma_hist: 2d tensor with dim N_hist by N
    :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param x_test: 1d tensor with length N_test
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with length N_test by N_hist by M
    """
    res = []

    # import pdb
    # pdb.set_trace()
    # point_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, Y, x, torch.tensor(0.58),
    #                  mu_tilde_l,
    #                  alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
    for x_star in x_test:
        print(x_star)
        sampled_y_hist = point_predsample(tilde_l_hist, tilde_sigma_hist, uL_vec_hist, tilde_sigma2_err_hist, Y, x, x_star,
                                          mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma,
                                          beta_tilde_sigma, N_sample, *args, **kwargs)
        res.append(sampled_y_hist)
    res = torch.stack(res)
    return res.numpy()

#### MAP
def point_predmap_sampling(n_sample, tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec_hist: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err_hist: scaler tensor
    :param x: 1d tensor with length N
    :param Y: 2d tensor with dim N, M
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 2d tensor with  dim N_percentile, M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vec = utils.uLvec2Lvec(uL_vec, M)
    sampled_ys = list()
    for n in range(n_sample):
        # sample tilde_l_star
        Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
        k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
        proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
        mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
        sigma2_l = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0,0] - torch.dot(proj_l, k_l.view(-1))
        if sigma2_l < 0:
            sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
        tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample() 

        # sample tilde_sigma_star
        Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        k_sigma = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
        mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma - mu_tilde_sigma))
        sigma2_sigma = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)[0,0] - torch.dot(proj_sigma, k_sigma.view(-1))
        if sigma2_sigma < 0:
            sigma2_sigma = torch.tensor(settings.precision).type(settings.torchType)
        tilde_sigma_star = Normal(loc=mu_sigma, scale=torch.sqrt(sigma2_sigma)).sample() 

        # sample posterior of y
        sigma2_err = torch.exp(tilde_sigma2_err)
        l = torch.exp(tilde_l)
        sigma = torch.exp(tilde_sigma)
        l_star = torch.exp(tilde_l_star)
        sigma_star = torch.exp(tilde_sigma_star)
        L = utils.vec2lowtriangle(L_vec, M)
        B_f = torch.mm(L, L.t())
        K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
        # First method
        # ts = time.time()
        # eigendecomposition
        w_B, v_B = torch.symeig(B_f, eigenvectors=True)
        w_K, v_K = torch.symeig(K_x, eigenvectors=True)
        k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
                                            sigma2=sigma_star.view(-1), ell2=l_star.view(-1))

        D = kronecker_operation.kronecker_product_diag(w_B, w_K)
        w = 1. / (sigma2_err + D)
        t = kronecker_operation.kron_mv(v_B.t(), v_K.t(), y)
        c = w * t
        d = kronecker_operation.kron_mv(v_B, v_K, c)
        mu_f = kronecker_operation.kron_mv(B_f, k_x.t(), d)
        A = kronecker_operation.kronecker_product(torch.mm(B_f, v_B), torch.mm(k_x.t(), v_K))
        a2 = sigma_star**2*torch.diag(B_f)

        # k_f = kronecker_operation.kronecker_product(B_f, k_x)
        # D = kronecker_operation.kronecker_product_diag(w_B, w_K)
        # w = 1. / (sigma2_err + D)
        # A = []
        # for m in range(M):
        #     A.append(kronecker_operation.kron_mv(v_B.t(), v_K.t(), k_f[:, m]))
        # A = torch.stack(A)
        # b = kronecker_operation.kron_mv(v_B.t(), v_K.t(), y)
        # mu_f = torch.mv(A, b * w)
        # import pdb
        # pdb.set_trace() 
        # a2 = torch.diag(kronecker_operation.kronecker_product(B_f, kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]),sigma1=sigma_star.view(-1),ell1=l_star.view(-1))))
        
        sigma2_f = a2 - (A * w * A).sum(dim=1)
        sigma2_y = sigma2_f + sigma2_err
        sigma2_y[sigma2_y <= 0] = settings.precision
        sampled_y = Normal(loc=mu_f, scale=torch.sqrt(sigma2_y)).sample() 
        # import pdb
        # pdb.set_trace()
        sampled_ys.append(sampled_y)
    sampled_ys = torch.stack(sampled_ys).numpy()
    quantiles_y = np.percentile(sampled_ys, q = [2.5, 97.5], axis=0)
    mean_y = np.mean(sampled_ys, axis=0)
    std_y = np.std(sampled_ys, axis=0)
    return quantiles_y, mean_y, std_y

def pointwise_predmap_sampling(n_sample, tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior percentile at grids using map estimates
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with dim N_grid, N_percentile, M
    """
    quantiles_ys = list()
    mean_ys = list()
    std_ys = list()

    for grid in grids:
        print(grid)

        quantiles_y, mean_y, std_y= point_predmap_sampling(n_sample, tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
        quantiles_ys.append(quantiles_y)
        mean_ys.append(mean_y)
        std_ys.append(std_y)
    quantiles_ys = np.stack(quantiles_ys)
    mean_ys = np.stack(mean_ys)    
    std_ys = np.stack(std_ys)
    return quantiles_ys, mean_ys, std_ys

def test_predmap_sampling(n_sample, tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, x_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior quantiles on grids using MAP
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length  N
    :param L_vec_hist: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with length N, M
    :param x: 1d tensor with length N
    :param x_test: 1d tensor with length N_test
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with dim N_test, N_quantile, M
    """
    quantiles_ys = list()
    mean_ys = list()
    std_ys = list()

    for x_star in x_test:
        print(x_star)
        quantiles_y, mean_y, std_y= point_predmap_sampling(n_sample, tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
        quantiles_ys.append(quantiles_y)
        mean_ys.append(mean_y)
        std_ys.append(std_y)
    quantiles_ys = np.stack(quantiles_ys)
    mean_ys = np.stack(mean_ys)    
    std_ys = np.stack(std_ys)
    return quantiles_ys, mean_ys, std_ys

#### MAP_fast
def point_predmap(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec_hist: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err_hist: scaler tensor
    :param x: 1d tensor with length N
    :param Y: 2d tensor with dim N, M
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 2d tensor with  dim N_percentile, M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vec = utils.uLvec2Lvec(uL_vec, M)

    # estimate posterior of tilde_l_star
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
    proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
    mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
    est_tilde_l_star = mu_l

    # estimate posterior of tilde_sigma_star
    Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    k_sigma = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
    mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma - mu_tilde_sigma))
    est_tilde_sigma_star = mu_sigma

    # sample posterior of y
    sigma2_err = torch.exp(tilde_sigma2_err)
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    est_l_star = torch.exp(est_tilde_l_star)
    est_sigma_star = torch.exp(est_tilde_sigma_star)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)

    # First method
    # ts = time.time()
    # eigendecomposition
    w_B, v_B = torch.symeig(B_f, eigenvectors=True)
    w_K, v_K = torch.symeig(K_x, eigenvectors=True)
    k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
                                        sigma2=est_sigma_star.view(-1), ell2=est_l_star.view(-1))
    k_f = kronecker_operation.kronecker_product(B_f, k_x)
    t = kronecker_operation.kronecker_product_diag(w_B, w_K)
    w = 1. / (sigma2_err + t)
    A = []
    for m in range(M):
        A.append(kronecker_operation.kron_mv(v_B.t(), v_K.t(), k_f[:, m]))
    A = torch.stack(A)
    b = kronecker_operation.kron_mv(v_B.t(), v_K.t(), y)
    mu_f = torch.mv(A, b * w)
    a2 = torch.diag(kronecker_operation.kronecker_product(B_f, kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]),
                                                                                             sigma1=est_sigma_star.view(
                                                                                                 -1),
                                                                                             ell1=est_l_star.view(
                                                                                                 -1))))
    # print(a2.size(), A.size(), w.size())
    sigma2_f = a2 - (A * w * A).sum(dim=1)
    sigma2_y = sigma2_f + sigma2_err
    # print(time.time()-ts)

    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    percentile_y = torch.stack([mu_f-1.96*torch.sqrt(sigma2_y), mu_f, mu_f+1.96*torch.sqrt(sigma2_y)])

    return percentile_y

def pointwise_predmap(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior percentile at grids using map estimates
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with dim N_grid, N_percentile, M
    """
    res = []

    for grid in grids:
        # print(grid)
        percentiles_y = point_predmap(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma)
        res.append(percentiles_y)
    res = torch.stack(res)
    return res

def test_predmap(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, x_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior quantiles on grids using MAP
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length  N
    :param L_vec_hist: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with length N, M
    :param x: 1d tensor with length N
    :param x_test: 1d tensor with length N_test
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with dim N_test, N_quantile, M
    """
    res = []

    # import pdb
    # pdb.set_trace()
    # point_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, torch.tensor(0.58),
    #                  mu_tilde_l,
    #                  alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
    for x_star in x_test:
        # print(x_star)
        y_star_quantiles = point_predmap(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, x_star,
                                                          mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
        res.append(y_star_quantiles)
    res = torch.stack(res)
    return res

#### Hadamard Sampling
def point_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
        Sample the posterior predictor at new input x_star
        :param tilde_l_hist: 2d tensor with dim N_hist by N
        :param tilde_sigma_hist: 2d tensor with dim N_hist by N
        :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
        :param tilde_sigma2_err_hist: 1d tensor with length N_hist
        :param x: 1d tensor with length N
        :param indx: 1d tensor with length N
        :param y: 1d tensor with length N
        :param x_star: scalar tensor
        :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
        :return: 2d tensor with length N_hist by M
        """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    sampled_y_hist = []
    for tilde_l, tilde_sigma, L_vec, tilde_sigma2_err in zip(tilde_l_hist, tilde_sigma_hist, L_vec_hist,
                                                             tilde_sigma2_err_hist):
        # sample posterior of tilde_l_star
        Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
        k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
        proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
        mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
        sigma2_l = kernels.RBF_cov(x_star.view([1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0, 0] - torch.dot(
            proj_l, k_l.view(-1))
        if sigma2_l < 0:
            sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample()

        # sample posterior of tilde_sigma_star
        Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        k_sigma = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
        mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma - mu_tilde_sigma))
        sigma2_sigma = kernels.RBF_cov(x_star.view([1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)[
                           0, 0] - torch.dot(proj_sigma, k_sigma.view(-1))
        if sigma2_sigma < 0:
            sigma2_sigma = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_sigma_star = Normal(loc=mu_sigma, scale=torch.sqrt(sigma2_sigma)).sample()

        # sample posterior of y
        sigma2_err = torch.exp(tilde_sigma2_err)
        l = torch.exp(tilde_l)
        sigma = torch.exp(tilde_sigma)
        sampled_l_star = torch.exp(sampled_tilde_l_star)
        sampled_sigma_star = torch.exp(sampled_tilde_sigma_star)
        L = utils.vec2lowtriangle(L_vec, M)
        B_f = torch.mm(L, L.t())
        K_i = logpos.generate_K_index(B_f, indx)
        K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
        K = K_x * K_i

        # ts = time.time()
        # # First method
        # invS = torch.inverse(K + sigma2_err*torch.eye(N))
        # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
        # Second method
        w_K, v_K = torch.symeig(K, eigenvectors=True)
        w = 1./(sigma2_err + w_K)
        invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
        # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
        # import pdb
        # pdb.set_trace()

        k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
                                            sigma2=sampled_sigma_star.view(-1), ell2=sampled_l_star.view(-1))
        # k_f = kronecker_operation.kronecker_product(B_f, k_x)
        k_i = B_f[logpos.generate_vectorized_indexes(torch.arange(M), indx)].view([M, N])
        # print(k_i.size(), k_x.size())
        k_f = (k_i * (k_x.repeat([1, M]).t())).t()
        mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
        invL = torch.cholesky(invS)
        T = torch.mm(k_f.t(), invL)
        A = kronecker_operation.kronecker_product(B_f, kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=sampled_sigma_star.view(-1), ell1=sampled_l_star.view(-1)))
        B = torch.mm(T, T.t())
        Sigma_f = A - B
        Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
        sigma2_y = torch.diagonal(Sigma_y)
        # print(time.time() - ts)

        # clip the sigma2_y.
        sigma2_y[sigma2_y <= 0] = settings.precision
        sampled_y = Normal(loc=mu_f, scale=torch.sqrt(sigma2_y)).sample()

        if sampled_y[0] != sampled_y[0]:
            import pdb
            pdb.set_trace()
        # record
        # import pdb
        # pdb.set_trace()
        sampled_y_hist.append(sampled_y)
    return torch.stack(sampled_y_hist)

def pointwise_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Sample the posterior predictor at grids
    :param tilde_l_hist: 2d tensor with dim N_hist by N
    :param tilde_sigma_hist: 2d tensor with dim N_hist by N
    :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with length N_grid by N_hist by M
    """
    res = []

    # import pdb
    # pdb.set_trace()
    # point_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, torch.tensor(0.58),
    #                  mu_tilde_l,
    #                  alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
    for grid in grids:
        print(grid)
        sampled_y_hist = point_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, grid,
                                          mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma,
                                          beta_tilde_sigma, *args, **kwargs)
        res.append(sampled_y_hist)
    res = torch.stack(res)
    return res

def indexedpoint_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, x_star, indx_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Sample the posterior predictor at new input x_star
    :param tilde_l_hist: 2d tensor with dim N_hist by N
    :param tilde_sigma_hist: 2d tensor with dim N_hist by N
    :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param x_star: scalar
    :param indx_star: scalar
    :param y: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 1d tensor with length N_hist
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    sampled_y_hist = []
    for tilde_l, tilde_sigma, L_vec, tilde_sigma2_err in zip(tilde_l_hist, tilde_sigma_hist, L_vec_hist,
                                                             tilde_sigma2_err_hist):
        # sample posterior of tilde_l_star
        Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
        k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
        proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
        mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
        sigma2_l = kernels.RBF_cov(x_star.view([1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0, 0] - torch.dot(
            proj_l, k_l.view(-1))
        if sigma2_l < 0:
            sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample()

        # sample posterior of tilde_sigma_star
        Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        k_sigma = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
        proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
        mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma - mu_tilde_sigma))
        sigma2_sigma = kernels.RBF_cov(x_star.view([1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)[
                           0, 0] - torch.dot(proj_sigma, k_sigma.view(-1))
        if sigma2_sigma < 0:
            sigma2_sigma = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_sigma_star = Normal(loc=mu_sigma, scale=torch.sqrt(sigma2_sigma)).sample()

        # sample posterior of y
        sigma2_err = torch.exp(tilde_sigma2_err)
        l = torch.exp(tilde_l)
        sigma = torch.exp(tilde_sigma)
        sampled_l_star = torch.exp(sampled_tilde_l_star)
        sampled_sigma_star = torch.exp(sampled_tilde_sigma_star)
        L = utils.vec2lowtriangle(L_vec, M)
        B_f = torch.mm(L, L.t())
        K_i = logpos.generate_K_index(B_f, indx)
        K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
        K = K_x * K_i

        # ts = time.time()
        # # First method
        # invS = torch.inverse(K + sigma2_err*torch.eye(N))
        # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
        # Second method
        w_K, v_K = torch.symeig(K, eigenvectors=True)
        w = 1./(sigma2_err + w_K)
        invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
        # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
        # import pdb
        # pdb.set_trace()
        k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
                                            sigma2=sampled_sigma_star.view(-1), ell2=sampled_l_star.view(-1))
        # k_f = kronecker_operation.kronecker_product(B_f, k_x)
        k_i = B_f[logpos.generate_vectorized_indexes(indx_star.view([1]), indx)].view([N, 1])
        # print(k_i.size(), k_x.size())
        k_f = k_i*k_x
        mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
        invL = torch.cholesky(invS)
        T = torch.mm(k_f.t(), invL)

        A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=sampled_sigma_star.view(-1), ell1=sampled_l_star.view(-1)) * B_f[indx_star, indx_star]
        B = torch.mm(T, T.t())
        sigma2_f = (A-B)[0, 0]
        sigma2_y = sigma2_f + sigma2_err
        # print(time.time() - ts)

        # clip the sigma2_y.
        sigma2_y[sigma2_y <= 0] = settings.precision
        sampled_y = Normal(loc=mu_f, scale=torch.sqrt(sigma2_y)).sample()

        if sampled_y[0] != sampled_y[0]:
            import pdb
            pdb.set_trace()
        # record
        # import pdb
        # pdb.set_trace()
        sampled_y_hist.append(sampled_y)
    return torch.cat(sampled_y_hist)

def test_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, x_test, indx_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Sample the posterior predictor at grids
    :param tilde_l_hist: 2d tensor with dim N_hist by N
    :param tilde_sigma_hist: 2d tensor with dim N_hist by N
    :param L_vec_hist: 2d tensor with dim N_hist by M(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param x_test: 1d tensor with length N_test
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 2d tensor with length N_test by N_hist
    """
    res = []

    # import pdb
    # pdb.set_trace()
    # point_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, torch.tensor(0.58),
    #                  mu_tilde_l,
    #                  alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
    for x_star, indx_star in zip(x_test, indx_test):
        print(x_star)
        sampled_y_hist = indexedpoint_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y,
                                          x_star, indx_star,
                                          mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma,
                                          beta_tilde_sigma, *args, **kwargs)
        res.append(sampled_y_hist)
    res = torch.stack(res)
    return res

#### Hadamard MAP
def point_predmap_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec_hist: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err_hist: scaler tensor
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 2d tensor with  dim N_percentile, M
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)

    # estimate posterior of tilde_l_star
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
    proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
    mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
    est_tilde_l_star = mu_l

    # estimate posterior of tilde_sigma_star
    Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    k_sigma = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
    mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma - mu_tilde_sigma))
    est_tilde_sigma_star = mu_sigma

    # estimate posterior of y
    sigma2_err = torch.exp(tilde_sigma2_err)
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    est_l_star = torch.exp(est_tilde_l_star)
    est_sigma_star = torch.exp(est_tilde_sigma_star)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    K_i = logpos.generate_K_index(B_f, indx)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    K = K_x * K_i

    # ts = time.time()
    # # First method
    # invS = torch.inverse(K + sigma2_err*torch.eye(N))
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # Second method
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1./(sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # import pdb
    # pdb.set_trace()

    k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
                                        sigma2=est_sigma_star.view(-1), ell2=est_l_star.view(-1))
    # k_f = kronecker_operation.kronecker_product(B_f, k_x)
    k_i = B_f[logpos.generate_vectorized_indexes(torch.arange(M), indx)].view([M, N])
    # print(k_i.size(), k_x.size())
    k_f = (k_i * (k_x.repeat([1, M]).t())).t()
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)
    A = kronecker_operation.kronecker_product(B_f, kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=est_sigma_star.view(-1), ell1=est_l_star.view(-1)))
    B = torch.mm(T, T.t())
    Sigma_f = A - B
    Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
    sigma2_y = torch.diagonal(Sigma_y)
    # print(time.time() - ts)

    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    percentile_y = torch.stack([mu_f-1.96*torch.sqrt(sigma2_y), mu_f, mu_f+1.96*torch.sqrt(sigma2_y)])

    return percentile_y

def pointwise_predmap_hadmard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior percentile at grids using map estimates
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 3d tensor with dim N_grid, N_percentile, M
    """
    res = []

    for grid in grids:
        print (grid)
        percentiles_y = point_predmap_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma)
        res.append(percentiles_y)
    res = torch.stack(res)
    return res

def indexedpoint_predmap_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_star, indx_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Sample the posterior predictor at new input x_star
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length N
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param x_star: scalar
    :param indx_star: scalar
    :param y: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 1d tensor with length N_quantile
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    sampled_y_hist = []

    # sample posterior of tilde_l_star
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
    proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
    mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
    est_tilde_l_star = mu_l

    # sample posterior of tilde_sigma_star
    Sigma_sigma = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    k_sigma = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_sigma, beta=beta_tilde_sigma)
    proj_sigma = torch.solve(input=k_sigma, A=Sigma_sigma)[0].view(-1)
    mu_sigma = mu_tilde_sigma + torch.dot(proj_sigma, (tilde_sigma - mu_tilde_sigma))
    est_tilde_sigma_star = mu_sigma

    # sample posterior of y
    sigma2_err = torch.exp(tilde_sigma2_err)
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    est_l_star = torch.exp(est_tilde_l_star)
    est_sigma_star = torch.exp(est_tilde_sigma_star)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    K_i = logpos.generate_K_index(B_f, indx)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    K = K_x * K_i

    # ts = time.time()
    # # First method
    # invS = torch.inverse(K + sigma2_err*torch.eye(N))
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # Second method
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1./(sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # import pdb
    # pdb.set_trace()
    k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=sigma, ell1=l, X2=x_star.view([1, 1]),
                                        sigma2=est_sigma_star.view(-1), ell2=est_l_star.view(-1))
    # k_f = kronecker_operation.kronecker_product(B_f, k_x)
    k_i = B_f[logpos.generate_vectorized_indexes(indx_star.view([1]), indx)].view([N, 1])
    # print(k_i.size(), k_x.size())
    k_f = k_i*k_x
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)

    A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=est_sigma_star.view(-1), ell1=est_l_star.view(-1)) * B_f[indx_star, indx_star]
    B = torch.mm(T, T.t())
    sigma2_f = (A-B)[0, 0]
    sigma2_y = sigma2_f + sigma2_err
    # print(time.time() - ts)

    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    y_star_quantiles = torch.tensor([mu_f-1.96*torch.sqrt(sigma2_y), mu_f, mu_f+1.96*torch.sqrt(sigma2_y)]).type(settings.torchType)

    return y_star_quantiles

def test_predmap_harmard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_test, indx_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs):
    """
    Compute the posterior quantiles on grids using MAP
    :param tilde_l: 1d tensor with length N
    :param tilde_sigma: 1d tensor with length  N
    :param L_vec_hist: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param x: 1d tensor with length N
    :param indx: 1d tensor with length N
    :param y: 1d tensor with length N
    :param x_test: 1d tensor with length N_test
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma: hyper-parameters of GP for tilde_l and tilde_sigma
    :return: 2d tensor with length N_test by N_quantile
    """
    res = []
    for x_star, indx_star in zip(x_test, indx_test):
        # print(x_star)
        y_star_quantiles = indexedpoint_predmap_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_star, indx_star,
                                                          mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, *args, **kwargs)
        res.append(y_star_quantiles)
    res = torch.stack(res)
    return res

###############GNMGP######################
#### Inhomogeneous MAP
def point_predmap_inhomogeneous(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param uL_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scaler tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 2d tensor with  dim N_percentile, M and 1d tensor with dim M(M+1)/2
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vecs = utils.uLvecs2Lvecs(uL_vecs, N, M)
    # estimate posterior of tilde_l_star
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
    proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
    mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
    est_tilde_l_star = mu_l

    # estimate posterior of L_star
    Sigma_uL = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    k_uL = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_L, beta=beta_L)
    proj_uL = torch.solve(input=k_uL, A=Sigma_uL)[0].view(-1)
    matrix_indx = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
    mu_uL_vec = torch.stack([mu_L + torch.dot(proj_uL, (uL_vecs[matrix_indx[:, m]] - mu_L)) for m in range(int(M*(M+1)/2))])
    est_uL_vec_star = mu_uL_vec
    est_L_vec_star = utils.uLvec2Lvec(est_uL_vec_star, M)
    est_L_f_star = utils.vec2lowtriangle(est_L_vec_star, M)

    # estimate posterior of y
    sigma2_err = torch.exp(tilde_sigma2_err)
    l = torch.exp(tilde_l)
    est_l_star = torch.exp(est_tilde_l_star)

    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    l = torch.exp(tilde_l)
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)  # dim: N, N
    K_i = logpos.generate_K_index_SVC(L_f_list)  # dim: NM , NM (index by row)
    neworder = torch.arange(N * M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder]  # dim: MN. MN (index by column)
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i

    # ts = time.time()
    # # First method
    # invS = torch.inverse(K + sigma2_err*torch.eye(N))
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # Second method
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1./(sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())

    k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=torch.ones(N).type(settings.torchType), ell1=l, X2=x_star.view([1, 1]), sigma2=torch.ones(1).type(settings.torchType), ell2=est_l_star.view(-1))

    A_f = torch.cat([k_i * L_f for k_i, L_f in zip(k_x.view(-1), L_f_list)], dim=0)
    k_f = torch.mm(est_L_f_star, A_f.t()).t() # dim NM, M (index by row)
    k_f = k_f[neworder] # dim MN, M (index by column)
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)
    A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=torch.ones(1).type(settings.torchType), ell1=est_l_star.view(-1))*torch.mm(est_L_f_star, est_L_f_star.t())
    B = torch.mm(T, T.t())
    Sigma_f = A - B
    Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
    sigma2_y = torch.diagonal(Sigma_y)
    # print(time.time() - ts)
    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    percentile_y = torch.stack([mu_f-1.96*torch.sqrt(sigma2_y), mu_f, mu_f+1.96*torch.sqrt(sigma2_y)])
    # import pdb
    # pdb.set_trace()

    return percentile_y, est_L_vec_star

def pointwise_predmap_inhomogeneous(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param uL_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scaler tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 3d tensor with dim N_grid, N_percentile, M and 2d tensor with dim N_grid, M(M+1)/2.
    """
    pred_ys = list()
    pred_L_vecs = list()

    for grid in grids:
        print(grid)
        percentiles_y, pred_L_vec = point_predmap_inhomogeneous(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L)
        pred_ys.append(percentiles_y)
        pred_L_vecs.append(pred_L_vec)
    pred_ys = torch.stack(pred_ys)
    pred_L_vecs = torch.stack(pred_L_vecs)
    return pred_ys, pred_L_vecs

def test_predmap_inhomogeneous(tilde_l, L_vecs, tilde_sigma2_err, Y, x, x_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param L_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scaler tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param x_test: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 3d tensor with dim N_grid, N_percentile, M and 2d tensor with dim N_grid, M(M+1)/2.
    """
    pred_ys = list()
    pred_L_vecs = list()

    for grid in x_test:
        # print(grid)
        percentiles_y, pred_L_vec = point_predmap_inhomogeneous(tilde_l, L_vecs, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L)
        pred_ys.append(percentiles_y)
        pred_L_vecs.append(pred_L_vec)
    pred_ys = torch.stack(pred_ys)
    pred_L_vecs = torch.stack(pred_L_vecs)
    return pred_ys, pred_L_vecs

def point_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, pred_smoothness=False, pred_cov=False, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param L_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scaler tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 2d tensor with dim n_quantiles, M, 1d tensor with M and 1d tensor with dim M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vecs = utils.uLvecs2Lvecs(uL_vecs, N, M)
    sampled_ys = list()
    sampled_tilde_l_stars = list()
    sampled_L_f_stars = list()
    for n in range(n_sample):
        if pred_smoothness:
            # sample tilde_l_star
            Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
            k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
            proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
            mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
            sigma2_l = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0,0] - torch.dot(proj_l, k_l.view(-1))
            if sigma2_l < 0:
                sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
            tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample() 
            sampled_tilde_l_stars.append(tilde_l_star)
        elif pred_cov:
            # sample L_star
            Sigma_uL = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
            k_uL = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_L, beta=beta_L)
            proj_uL = torch.solve(input=k_uL, A=Sigma_uL)[0].view(-1)
            matrix_indx = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
            mu_uL_vec = torch.stack([mu_L + torch.dot(proj_uL, (uL_vecs[matrix_indx[:, m]] - mu_L)) for m in range(int(M*(M+1)/2))])
            sigma2_uL_vec = torch.stack([kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_L, beta=beta_L)[0,0] - torch.dot(proj_uL, k_uL.view(-1)) for m in range(int(M*(M+1)/2))])
            sigma2_uL_vec[sigma2_uL_vec < 0] = torch.tensor(settings.precision).type(settings.torchType)
            uL_vec_star = Normal(loc = mu_uL_vec, scale=torch.sqrt(sigma2_uL_vec)).sample()   
            L_vec_star = utils.uLvec2Lvec(uL_vec_star, M)
            L_f_star = utils.vec2lowtriangle(L_vec_star, M)
            sampled_L_f_stars.append(L_f_star)
        else:
            # sample tilde_l_star
            Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
            k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
            proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
            mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
            sigma2_l = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0,0] - torch.dot(proj_l, k_l.view(-1))
            if sigma2_l < 0:
                sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
            tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample() 
            sampled_tilde_l_stars.append(tilde_l_star)

            # sample L_star
            Sigma_uL = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
            k_uL = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_L, beta=beta_L)
            proj_uL = torch.solve(input=k_uL, A=Sigma_uL)[0].view(-1)
            matrix_indx = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
            mu_uL_vec = torch.stack([mu_L + torch.dot(proj_uL, (uL_vecs[matrix_indx[:, m]] - mu_L)) for m in range(int(M*(M+1)/2))])
            sigma2_uL_vec = torch.stack([kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_L, beta=beta_L)[0,0] - torch.dot(proj_uL, k_uL.view(-1)) for m in range(int(M*(M+1)/2))])
            sigma2_uL_vec[sigma2_uL_vec < 0] = torch.tensor(settings.precision).type(settings.torchType)
            uL_vec_star = Normal(loc = mu_uL_vec, scale=torch.sqrt(sigma2_uL_vec)).sample()   
            L_vec_star = utils.uLvec2Lvec(uL_vec_star, M)
            L_f_star = utils.vec2lowtriangle(L_vec_star, M)
            sampled_L_f_stars.append(L_f_star)

            # sample y
            sigma2_err = torch.exp(tilde_sigma2_err)
            l = torch.exp(tilde_l)
            l_star = torch.exp(tilde_l_star)
            L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
            L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
            l = torch.exp(tilde_l)
            sigma2_err = torch.exp(tilde_sigma2_err)
            K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)  # dim: N, N
            K_i = logpos.generate_K_index_SVC(L_f_list)  # dim: NM , NM (index by row)
            neworder = torch.arange(N * M).view([N, M]).t().contiguous().view(-1)
            K_i = K_i[:, neworder][neworder]  # dim: MN. MN (index by column)
            K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
            # ts = time.time()
            # # First method
            # invS = torch.inverse(K + sigma2_err*torch.eye(N))
            # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
            # Second method
            w_K, v_K = torch.symeig(K, eigenvectors=True)
            w = 1./(sigma2_err + w_K)
            invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
            k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=torch.ones(N).type(settings.torchType), ell1=l, X2=x_star.view([1, 1]), sigma2=torch.ones(1).type(settings.torchType), ell2=l_star.view(-1))
            A_f = torch.cat([k_i * L_f for k_i, L_f in zip(k_x.view(-1), L_f_list)], dim=0)
            k_f = torch.mm(L_f_star, A_f.t()).t() # dim NM, M (index by row)
            k_f = k_f[neworder] # dim MN, M (index by column)
            mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
            invL = torch.cholesky(invS)
            T = torch.mm(k_f.t(), invL)
            A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=torch.ones(1).type(settings.torchType), ell1=l_star.view(-1))*torch.mm(L_f_star, L_f_star.t())
            B = torch.mm(T, T.t())
            Sigma_f = A - B
            Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
            sigma2_y = torch.diagonal(Sigma_y)
            # print(time.time() - ts)
            # clip the sigma2_y.
            sigma2_y[sigma2_y <= 0] = settings.precision
            sampled_y = Normal(loc=mu_f, scale=torch.sqrt(sigma2_y)).sample() 
            # import pdb
            # pdb.set_trace()
            sampled_ys.append(sampled_y)

    if pred_smoothness:
        sampled_tilde_l_stars = torch.stack(sampled_tilde_l_stars).numpy()
        return sampled_tilde_l_stars 
    elif pred_cov:
        sampled_L_f_stars = torch.stack(sampled_L_f_stars).numpy()
        return sampled_L_f_stars
    else:
        sampled_ys = torch.stack(sampled_ys).numpy()
        quantiles_y = np.percentile(sampled_ys, q = [2.5, 97.5], axis=0)
        mean_y = np.mean(sampled_ys, axis=0)
        std_y = np.std(sampled_ys, axis=0)
        return quantiles_y, mean_y, std_y

def pointwise_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, pred_smoothness=False, pred_cov=False, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param L_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scaler tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 3d tensor with dim N_grid, n_quantiles, M, 2d tensor with dim N_grid, M and 2d tensor with dim N_grid, M
    """
    quantiles_ys = list()
    mean_ys = list()
    std_ys = list()
    sampled_tilde_ls = list()
    sampled_L_fs = list()

    for grid in grids:
        print(grid)
        if pred_smoothness:
            sampled_tilde_l_stars = point_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, pred_smoothness = True)
            sampled_tilde_ls.append(sampled_tilde_l_stars)
        elif pred_cov:
            sampled_L_f_stars = point_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, pred_cov = True)
            sampled_L_fs.append(sampled_L_f_stars)
        else:
            quantiles_y, mean_y, std_y = point_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L)
            quantiles_ys.append(quantiles_y)
            mean_ys.append(mean_y)
            std_ys.append(std_y)
    if pred_smoothness:
        sampled_tilde_ls = np.stack(sampled_tilde_ls)
        return sampled_tilde_ls
    elif pred_cov:
        sampled_L_fs = np.stack(sampled_L_fs)
        return sampled_L_fs
    else:
        quantiles_ys = np.stack(quantiles_ys)
        mean_ys = np.stack(mean_ys)    
        std_ys = np.stack(std_ys)
        return quantiles_ys, mean_ys, std_ys

def test_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, x_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using MAP estimates
    :param tilde_l: 1d tensor with length N
    :param L_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scaler tensor
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param x_test: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 3d tensor with dim N_grid, N_percentile, M and 2d tensor with dim N_grid, M(M+1)/2.
    """
    quantiles_ys = list()
    mean_ys = list()
    std_ys = list()

    for x_star in x_test:
        print(x_star)
        quantiles_y, mean_y, std_y= point_predmap_inhomogeneous_sampling(n_sample, tilde_l, uL_vecs, tilde_sigma2_err, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L)
        quantiles_ys.append(quantiles_y)
        mean_ys.append(mean_y)
        std_ys.append(std_y)
    quantiles_ys = np.stack(quantiles_ys)
    mean_ys = np.stack(mean_ys)    
    std_ys = np.stack(std_ys)
    return quantiles_ys, mean_ys, std_ys

#### Inhomogeneous Sampling
def point_predsample_inhomogeneous(tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, N_sample, *args, **kwargs):
    """
    Compute the posterior predictor at new input x_star using posterior sampling
    :param tilde_l_hist: 2d tensor with size N_hist, N
    :param L_vecs_hist: 2d tensor with size N_hist, NM(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist 
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param x_star: scalar tensor
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 2d tensor with dim N_hist, M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist = tilde_l_hist[-N_sample:], uL_vecs_hist[-N_sample:], tilde_sigma2_err_hist[-N_sample:]
    sampled_y_hist = []
    for tilde_l, uL_vecs, tilde_sigma2_err in zip(tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist):
        # ts = time.time()
        # sample posterior of tilde_l_star
        L_vecs = utils.uLvecs2Lvecs(uL_vecs, N, M)
        Sigma_l = kernels.RBF_cov(x.view([-1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
        k_l = kernels.RBF_cov(x.view([-1,1]), x_star.view(1,1), alpha=alpha_tilde_l, beta=beta_tilde_l)
        proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
        mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l-mu_tilde_l))
        sigma2_l = kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_tilde_l, beta=beta_tilde_l)[0,0] - torch.dot(proj_l, k_l.view(-1))
        if sigma2_l < 0:
            sigma2_l = torch.tensor(settings.precision).type(settings.torchType)
        sampled_tilde_l_star = Normal(loc=mu_l, scale=torch.sqrt(sigma2_l)).sample()
        sampled_l_star = torch.exp(sampled_tilde_l_star)
        # import pdb
        # pdb.set_trace()
        # print("sample tilde_l_star costs {}s".format(time.time()-ts))

        # ts = time.time()
        # sample posterior of L_star
        Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
        k_L = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_L, beta=beta_L)
        proj_L = torch.solve(input=k_L, A=Sigma_L)[0].view(-1)
        matrix_indx = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
        mu_L_vec = torch.stack([mu_L + torch.dot(proj_L, (L_vecs[matrix_indx[:, m]] - mu_L)) for m in range(int(M*(M+1)/2))])
        sigma2_L_vec = torch.stack([kernels.RBF_cov(x_star.view([1,1]), alpha=alpha_L, beta=beta_L)[0,0] - torch.dot(proj_L, k_L.view(-1)) for m in range(int(M*(M+1)/2))])
        sigma2_L_vec[sigma2_L_vec < 0] = torch.tensor(settings.precision).type(settings.torchType)
        sampled_L_vec_star = Normal(loc = mu_L_vec, scale=torch.sqrt(sigma2_L_vec)).sample()
        sampled_L_f_star = utils.vec2lowtriangle(sampled_L_vec_star, M)
        # import pdb
        # pdb.set_trace()
        # print("sample L_star costs {}s".format(time.time()-ts))

        # ts = time.time()
        # sample posterior of y
        L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
        l = torch.exp(tilde_l)
        sigma2_err = torch.exp(tilde_sigma2_err)
        K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)  # dim: N, N
        K_i = logpos.generate_K_index_SVC(L_f_list)  # dim: NM , NM (index by row)
        neworder = torch.arange(N * M).view([N, M]).t().contiguous().view(-1)
        K_i = K_i[:, neworder][neworder]  # dim: MN. MN (index by column)
        K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
        # First method
        w_K, v_K = torch.symeig(K, eigenvectors=True)
        w = 1./(sigma2_err + w_K)
        invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
        # # Second method
        # import pdb
        # pdb.set_trace()
        # invS = torch.inverse(K + sigma2_err*torch.eye(N*M))
        k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=torch.ones(N).type(settings.torchType), ell1=l, X2=x_star.view([1, 1]), sigma2=torch.ones(1).type(settings.torchType), ell2=sampled_l_star.view(-1))
        A_f = torch.cat([k_i * L_f for k_i, L_f in zip(k_x.view(-1), L_f_list)], dim=0)
        k_f = torch.mm(sampled_L_f_star, A_f.t()).t() # dim NM, M (index by row)
        k_f = k_f[neworder] # dim MN, M (index by column)
        mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
        invL = torch.cholesky(invS)
        T = torch.mm(k_f.t(), invL)
        A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), sigma1=torch.ones(1).type(settings.torchType), ell1=sampled_l_star.view(-1))*torch.mm(sampled_L_f_star, sampled_L_f_star.t())
        B = torch.mm(T, T.t())
        Sigma_f = A - B
        Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
        sigma2_y = torch.diagonal(Sigma_y)
        # clip the sigma2_y.
        sigma2_y[sigma2_y <= 0] = settings.precision
        sampled_y = Normal(loc=mu_f, scale=torch.sqrt(sigma2_y)).sample() 
        # if sampled_y[0] != sampled_y[0]:
        #     import pdb
        #     pdb.set_trace()
        sampled_y_hist.append(sampled_y)
        # import pdb
        # pdb.set_trace()
        # print("sample y costs {}s".format(time.time()-ts))
        # import pdb
        # pdb.set_trace()

    return torch.stack(sampled_y_hist)

def pointwise_predsample_inhomogeneous(tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist, Y, x, grids, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, N_sample, *args, **kwargs):
    """
    Sample the posterior predictor at grids
    :param tilde_l_hist: 2d tensor with size N_hist, N
    :param L_vecs_hist: 2d tensor with size N_hist, NM(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 3d tensor with dim N_grid, dim N_hist, M
    """
    res = []

    for grid in grids:
        print(grid)
        sampled_y_hist = point_predsample_inhomogeneous(tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist, Y, x, grid, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, N_sample=N_sample)
        res.append(sampled_y_hist)
    res = torch.stack(res)
    return res.numpy()

def test_predsample_inhomogeneous(tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist, Y, x, x_test, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, N_sample, *args, **kwargs):
    """
    Sample the posterior predictor at x_test
    :param tilde_l_hist: 2d tensor with size N_hist, N
    :param L_vecs_hist: 2d tensor with size N_hist, NM(M+1)/2
    :param tilde_sigma2_err_hist: 1d tensor with length N_hist
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param x_test: 1d tensor with length N_test
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L: hyper-parameters of GP for tilde_l and L_vecs
    :return: 3d tensor with size N_test, N_hist, M
    """
    res = []
    for x_star in x_test:
        print(x_star)
        sampled_y_hist = point_predsample_inhomogeneous(tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist, Y, x, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, N_sample=N_sample, *args, **kwargs)
        res.append(sampled_y_hist)
    res = torch.stack(res)
    return res.numpy()

#### Hadamard Inhomogeneous MAP
def point_predmap_SVC_hadamard(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, x_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, *args, **kwargs):
    N = y.size(0)
    M = torch.unique(indx).size(0)
    
    # estimate posterior of tilde_l_star
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
    proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
    mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
    est_tilde_l_star = mu_l
    # import pdb
    # pdb.set_trace()

    # estimate posterior of L_star
    Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    k_L = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_L, beta=beta_L)
    proj_L = torch.solve(input=k_L, A=Sigma_L)[0].view(-1)
    matrix_indx = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
    mu_L_vec = torch.stack([mu_L + torch.dot(proj_L, (L_vecs[matrix_indx[:, m]] - mu_L)) for m in range(int(M*(M+1)/2))])
    est_L_vec_star = mu_L_vec
    est_L_f_star = utils.vec2lowtriangle(est_L_vec_star, M)
    # import pdb
    # pdb.set_trace()

    # estimate posterior of y
    l = torch.exp(tilde_l)
    sigma2_err = torch.exp(tilde_sigma2_err)
    est_l_star = torch.exp(est_tilde_l_star)

    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)  # dim: N, N
    K_i = logpos.generate_K_index_SVC_hadamard0(L_f_list, indx)  # dim: N , N 
    K = K_x * K_i

    # ts = time.time()
    # # First method
    # invS = torch.inverse(K + sigma2_err*torch.eye(N))
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # Second method
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1./(sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))

    k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=torch.ones(N).type(settings.torchType), ell1=l, X2=x_star.view([1, 1]), sigma2=torch.ones(1).type(settings.torchType), ell2=est_l_star.view(-1)) # dim N, 1 
    # import pdb
    # pdb.set_trace()
    L = torch.stack([L_f[i, :] for i, L_f in zip(indx, L_f_list)]) # dim N, M
    k_i = torch.mm(L, est_L_f_star.t())# dim N, M
    k_f = k_x * k_i # dim N, M
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)
    A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), ell1=est_l_star.view(-1))*torch.mm(est_L_f_star, est_L_f_star.t())
    B = torch.mm(T, T.t())
    Sigma_f = A - B
    Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
    sigma2_y = torch.diagonal(Sigma_y)
    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    percentile_y = torch.stack([mu_f-1.96*torch.sqrt(sigma2_y), mu_f, mu_f+1.96*torch.sqrt(sigma2_y)])
    # import pdb
    # pdb.set_trace()
    return percentile_y

def pointwise_predmap_SVC_hadamard(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, grids, *args, **kwargs):
    """
    compute posterior predictive point-wise confidence intervals on grids
    :return: 3d tensor with dim N_grid, N_percentiles, M
    """
    res = []
    for grid in grids:
        print(grid)
        percentile_grid = point_predmap_SVC_hadamard(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, grid, *args, **kwargs)
        res.append(percentile_grid)
    res = torch.stack(res)
    return res

def indexedpoint_predmap_SVC_hadamard(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, x_star, indx_star, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, *args, **kwargs):
    """
    Compute posterior mean and std for new time stamp x_star and dimension indx.
    :return: 1d tensor with size 2
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    
    # estimate posterior of tilde_l_star
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    k_l = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_tilde_l, beta=beta_tilde_l)
    proj_l = torch.solve(input=k_l, A=Sigma_l)[0].view(-1)
    mu_l = mu_tilde_l + torch.dot(proj_l, (tilde_l - mu_tilde_l))
    est_tilde_l_star = mu_l
    # import pdb
    # pdb.set_trace()

    # estimate posterior of L_star
    Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    k_L = kernels.RBF_cov(x.view([-1, 1]), x_star.view(1, 1), alpha=alpha_L, beta=beta_L)
    proj_L = torch.solve(input=k_L, A=Sigma_L)[0].view(-1)
    matrix_indx = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])
    mu_L_vec = torch.stack([mu_L + torch.dot(proj_L, (L_vecs[matrix_indx[:, m]] - mu_L)) for m in range(int(M*(M+1)/2))])
    est_L_vec_star = mu_L_vec
    est_L_f_star = utils.vec2lowtriangle(est_L_vec_star, M)
    # import pdb
    # pdb.set_trace()

    # estimate posterior of y
    sigma2_err = torch.exp(tilde_sigma2_err)
    l = torch.exp(tilde_l)
    est_l_star = torch.exp(est_tilde_l_star)

    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    l = torch.exp(tilde_l)
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)  # dim: N, N
    K_i = logpos.generate_K_index_SVC_hadamard0(L_f_list, indx)  # dim: N , N 
    K = K_x * K_i

    # ts = time.time()
    # # First method
    # invS = torch.inverse(K + sigma2_err*torch.eye(N))
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))
    # Second method
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1./(sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())
    # print(torch.mm(invS, K + sigma2_err*torch.eye(N)))

    k_x = kernels.Nonstationary_RBF_cov(X1=x.view([-1, 1]), sigma1=torch.ones(N).type(settings.torchType), ell1=l, X2=x_star.view([1, 1]), sigma2=torch.ones(1).type(settings.torchType), ell2=est_l_star.view(-1)) # dim N, 1 
    # import pdb
    # pdb.set_trace()
    L = torch.stack([L_f[i, :] for i, L_f in zip(indx, L_f_list)]) # dim N, M 
    k_i = torch.mm(L, est_L_f_star[indx_star, :].view([-1, 1])) # dim N, 1
    k_f = k_x * k_i # dim N, 1
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)
    A = kernels.Nonstationary_RBF_cov(X1=x_star.view([1, 1]), ell1=est_l_star.view(-1))*torch.mm(est_L_f_star, est_L_f_star.t())
    B = torch.mm(T, T.t())
    Sigma_f = A - B
    Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
    sigma2_y = torch.diagonal(Sigma_y)
    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    return torch.cat([mu_f, sigma2_y.view(-1)])

def test_predmap_SVC_hadamard(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y, x_test, indx_test, *args, **kwargs):
    """
    Compute posterior mean and std for testing data
    :return: mean and std with 1d tensor of size N_test
    """
    res = []
    for x_star, indx_star in zip(x_test, indx_test):
        print(x_star)
        y_star_mean_std = indexedpoint_predmap_SVC_hadamard(tilde_l, L_vecs, tilde_sigma2_err, x, indx, y,
                                                         x_star, indx_star, *args, **kwargs)
        res.append(y_star_mean_std)
    res = torch.stack(res)
    return res[:, 0], res[:, 1]


##############LMC########################
#### Stationary MAP
def pointwise_predmap_S(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, grids, *args, **kwargs):
    """
    compute posterior predictive point-wise confidence intervals on grids
    :param tilde_l: scalar
    :param tilde_sigma: scalar
    :param uL_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param grids: 1d tensor with length N_grid
    :return: 2d tensor with dim N_grid, N_percentiles, M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vec = utils.uLvec2Lvec(uL_vec, M)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    # print("l = {}, sigma = {}, B_f = {}, sigma2_err = {}".format(l, sigma, B_f, sigma2_err))
    K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
    invS = torch.inverse(kronecker_operation.kronecker_product(B_f, K_x) + sigma2_err*torch.eye(N*M).type(settings.torchType))
    res = []
    for grid in grids:
        k_x = kernels.RBF_cov(x.view([-1, 1]), grid.view([1, 1]), alpha=sigma, beta=l)
        k_f = kronecker_operation.kronecker_product(B_f, k_x).t()
        mu_f = torch.mv(torch.mm(k_f, invS), y)
        sigma2_f = sigma**2*torch.diag(B_f) - torch.diagonal(torch.mm(torch.mm(k_f, invS), k_f.t()))
        sigma2_y = sigma2_f + sigma2_err
        sigma2_y[sigma2_y<0] = settings.precision
        res.append(torch.stack([mu_f - 1.96*torch.sqrt(sigma2_y), mu_f, mu_f + 1.96*torch.sqrt(sigma2_y)]))
    res = torch.stack(res)
    return res

def test_predmap_S(tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err, Y, x, test_x, *args, **kwargs):
    """
    compute posterior predictive point-wise confidence intervals on grids
    :param tilde_l: scalar
    :param tilde_sigma: scalar
    :param uL_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma2_err: scalar
    :param train_Y: 2d tensor with dim N by M
    :param train_x: 1d tensor with length N
    :param test_Y: 2d tensor with dim N_test, M
    :param test_x: 1d tensor with length N_test
    :return: pred_Y: tensor with dim N_test, M
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vec = utils.uLvec2Lvec(uL_vec, M)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    # print("l = {}, sigma = {}, B_f = {}, sigma2_err = {}".format(l, sigma, B_f, sigma2_err))
    K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
    invS = torch.inverse(kronecker_operation.kronecker_product(B_f, K_x) + sigma2_err * torch.eye(N * M).type(settings.torchType))
    res_mean = []
    res_std = []
    for x_star in test_x:
        k_x = kernels.RBF_cov(x.view([-1, 1]), x_star.view([1, 1]), alpha=sigma, beta=l)
        k_f = kronecker_operation.kronecker_product(B_f, k_x).t()
        mu_f = torch.mv(torch.mm(k_f, invS), y)
        sigma2_f = sigma ** 2 * torch.diag(B_f) - torch.diagonal(torch.mm(torch.mm(k_f, invS), k_f.t()))
        sigma2_y = sigma2_f + sigma2_err
        sigma2_y[sigma2_y < 0] = settings.precision
        res_mean.append(mu_f)
        res_std.append(torch.sqrt(sigma2_y))
    res_mean = torch.stack(res_mean)
    res_std = torch.stack(res_std)
    return res_mean, res_std

def pointwise_predsample_S(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs, Y, x, grids, *args, **kwargs):
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    samples = []
    for tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err in zip(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs):
        L_vec = utils.uLvec2Lvec(uL_vec, M)
        L = utils.vec2lowtriangle(L_vec, M)
        B_f = torch.mm(L, L.t())
        l = torch.exp(tilde_l)
        sigma = torch.exp(tilde_sigma)
        sigma2_err = torch.exp(tilde_sigma2_err)
        # print("l = {}, sigma = {}, B_f = {}, sigma2_err = {}".format(l, sigma, B_f, sigma2_err))
        K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
        invS = torch.inverse(kronecker_operation.kronecker_product(B_f, K_x) + sigma2_err*torch.eye(N*M).type(settings.torchType))
        res = []
        for grid in grids:
            k_x = kernels.RBF_cov(x.view([-1, 1]), grid.view([1, 1]), alpha=sigma, beta=l)
            k_f = kronecker_operation.kronecker_product(B_f, k_x).t()
            mu_f = torch.mv(torch.mm(k_f, invS), y)
            sigma2_f = sigma**2*torch.diag(B_f) - torch.diagonal(torch.mm(torch.mm(k_f, invS), k_f.t()))
            sigma2_y = sigma2_f + sigma2_err
            sigma2_y[sigma2_y<0] = settings.precision
            res.append(mu_f + np.random.randn()*torch.sqrt(sigma2_y))
        res = torch.stack(res)
        samples.append(res)
    return torch.stack(samples).numpy()

def test_predsample_S(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs, Y, x, test_x, *args, **kwargs):
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    samples = []
    for tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err in zip(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs):
        L_vec = utils.uLvec2Lvec(uL_vec, M)
        L = utils.vec2lowtriangle(L_vec, M)
        B_f = torch.mm(L, L.t())
        l = torch.exp(tilde_l)
        sigma = torch.exp(tilde_sigma)
        sigma2_err = torch.exp(tilde_sigma2_err)
        # print("l = {}, sigma = {}, B_f = {}, sigma2_err = {}".format(l, sigma, B_f, sigma2_err))
        K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
        invS = torch.inverse(kronecker_operation.kronecker_product(B_f, K_x) + sigma2_err*torch.eye(N*M).type(settings.torchType))
        res = []
        for x_star in test_x:
            k_x = kernels.RBF_cov(x.view([-1, 1]), x_star.view([1, 1]), alpha=sigma, beta=l)
            k_f = kronecker_operation.kronecker_product(B_f, k_x).t()
            mu_f = torch.mv(torch.mm(k_f, invS), y)
            sigma2_f = sigma**2*torch.diag(B_f) - torch.diagonal(torch.mm(torch.mm(k_f, invS), k_f.t()))
            sigma2_y = sigma2_f + sigma2_err
            sigma2_y[sigma2_y<0] = settings.precision
            res.append(mu_f + np.random.randn()*torch.sqrt(sigma2_y))
        res = torch.stack(res)
        samples.append(res)
    return torch.stack(samples).numpy()  

#### Hadamard Stationary MAP
def point_predmap_S_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_star, *args, **kwargs):
    N = y.size(0)
    M = torch.unique(indx).size(0)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
    K_i = logpos.generate_K_index(B_f, indx)
    K = K_x * K_i

    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1. / (sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())

    k_x = kernels.RBF_cov(X1=x.view([-1, 1]), X2=x_star.view([1, 1]), alpha=sigma, beta=l)
    # k_f = kronecker_operation.kronecker_product(B_f, k_x)
    k_i = B_f[logpos.generate_vectorized_indexes(torch.arange(M), indx)].view([M, N])
    # print(k_i.size(), k_x.size())
    k_f = (k_i * (k_x.repeat([1, M]).t())).t()
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)
    A = kronecker_operation.kronecker_product(B_f, kernels.RBF_cov(X1=x_star.view([1, 1]), alpha=sigma, beta=l))
    B = torch.mm(T, T.t())
    Sigma_f = A - B
    Sigma_y = Sigma_f + sigma2_err * torch.eye(M).type(settings.torchType)
    sigma2_y = torch.diagonal(Sigma_y)

    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    percentile_y = torch.stack([mu_f - 1.96 * torch.sqrt(sigma2_y), mu_f, mu_f + 1.96 * torch.sqrt(sigma2_y)])
    return percentile_y

def pointwise_predmap_S_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, grids, *args, **kwargs):
    """
    compute posterior predictive point-wise confidence intervals on grids
    :return: 3d tensor with dim N_grid, N_percentiles, M
    """
    res = []
    for grid in grids:
        percentile_grid = point_predmap_S_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, grid, *args, **kwargs)
        res.append(percentile_grid)
    res = torch.stack(res)
    return res

def indexedpoint_predmap_S_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_star, indx_star, *args, **kwargs):
    """
    Compute posterior mean and std for new time stamp x_star and dimension indx.
    :return: 1d tensor with size 2
    """
    N = y.size(0)
    M = torch.unique(indx).size(0)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l)
    sigma = torch.exp(tilde_sigma)
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.RBF_cov(x.view([-1, 1]), alpha=sigma, beta=l)
    K_i = logpos.generate_K_index(B_f, indx)
    K = K_x * K_i

    w_K, v_K = torch.symeig(K, eigenvectors=True)
    w = 1. / (sigma2_err + w_K)
    invS = torch.mm(torch.mm(v_K, torch.diag(w)), v_K.t())

    k_x = kernels.RBF_cov(X1=x.view([-1, 1]), X2=x_star.view([1, 1]), alpha=sigma, beta=l)
    # k_f = kronecker_operation.kronecker_product(B_f, k_x)
    k_i = B_f[logpos.generate_vectorized_indexes(indx_star.view([1]), indx)].view([N, 1])
    # print(k_i.size(), k_x.size())
    k_f = k_i * k_x
    mu_f = torch.mv(k_f.t(), torch.mv(invS, y))
    invL = torch.cholesky(invS)
    T = torch.mm(k_f.t(), invL)
    A = kronecker_operation.kronecker_product(B_f, kernels.RBF_cov(X1=x_star.view([1, 1]), alpha=sigma, beta=l))
    B = torch.mm(T, T.t())
    sigma2_f = (A - B)[0, 0]
    sigma2_y = sigma2_f + sigma2_err

    # clip the sigma2_y.
    sigma2_y[sigma2_y <= 0] = settings.precision
    y_star_mean_std = torch.tensor([mu_f, torch.sqrt(sigma2_y)]).type(settings.torchType)
    return y_star_mean_std

def test_predmap_S_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y, x_test, indx_test, *args, **kwargs):
    """
    Compute posterior mean and std for testing data
    :return: mean and std with 1d tensor of size N_test
    """
    res = []
    for x_star, indx_star in zip(x_test, indx_test):
        # print(x_star)
        y_star_mean_std = indexedpoint_predmap_S_hadamard(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, x, indx, y,
                                                         x_star, indx_star, *args, **kwargs)
        res.append(y_star_mean_std)
    res = torch.stack(res)
    return res[:, 0], res[:, 1]




if __name__ == "__main__":
    pass
