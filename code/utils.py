import torch
import numpy as np
import math
import matplotlib.pyplot as plt

TensorType = torch.DoubleTensor
tridiagonal_jitter = 1e-4
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev="cpu"
device = torch.device(dev)

def reparameterize(mean, var, z, full_cov=False, use_std=False):
    """
    Implenments the 'reparametyerization trick' for the Gausian, either full rank or diagnal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape _,N,N and the full covariane is used. Otherwise var must be _,N and the operation is elementwise

    :param mean: mean of shape _,N
    :param var: covariance of shae _,N or _,N,N
    :param z: samples from unit Gaussian of shape _,N
    :param full_cov: bool to indicate whether var is shape _,N or _,N,N
    :return sample from N(mean, var) of shape _,N
    """
    if var is None:
        return mean
    if full_cov is False:
        return mean + z * (var + tridiagonal_jitter)**0.5
    else:
        D =len(mean.size())
        N = mean.size()[-1]
        if D == 1:
            if use_std:
                chol = var
            else:
                I = tridiagonal_jitter * torch.eye(N).type(TensorType).to(mean.device)
                # try:
                #     chol = torch.cholesky(var + I)
                # except:
                #     import pdb
                #     pdb.set_trace()
                chol = torch.cholesky(var + I)
            z_v = z[:, None]
            f = mean + torch.matmul(chol, z_v)[:, 0]
        if D == 2:
            if use_std:
                chol = var
            else:
                I = tridiagonal_jitter * torch.eye(N)[None, :, :].type(TensorType).to(mean.device)
                chol = torch.cholesky(var + I)
            z_v = z[:, :, None]
            f = mean + torch.matmul(chol, z_v)[:, :, 0]
        if D == 3:
            if use_std:
                chol = var
            else:
                I = tridiagonal_jitter * torch.eye(N)[None, None, :, :].type(TensorType).to(mean.device)
                chol = torch.cholesky(var + I)
            z_v = z[:, :, :, None]
            f = mean + torch.matmul(chol, z_v)[:, :, :, 0]
        return f


def mat2ltri(X):
    X0 = X.clone()
    ii, jj = np.triu_indices(X0.size(-2), k=1, m=X0.size(-1))
    X0[..., ii, jj] = 0
    return X0


def squared_distance(X, X2):
    if X2 is not None:
        difference = X.unsqueeze(1) - X2.unsqueeze(0)
    else:
        difference = X.unsqueeze(1) - X2.unsqueeze(0)
    squared_distance = torch.sum(difference * difference, -1)
    return squared_distance


def squared_dist(X, X2, length_scales):
    if X2 is None:
        return squared_distance(X / length_scales, X / length_scales)
    else:
        return squared_distance(X / length_scales, X2 / length_scales)


def create_RBF(X, X2=None, scale2=1., length_scales=1.):
    # compute covariance matrix induced by RBF
    r2 = squared_dist(X, X2, length_scales)
    return scale2*torch.exp(-0.5*r2)


def create_Gibbs(X, X2, ell_X, ell_X2, scale2=1.):
    # compute covariance matrix induced by Gibbs
    r2 = squared_dist(X, X2, length_scales=1.)
    # import pdb # pdb.set_trace()
    ell_matrix = (ell_X**2).unsqueeze(1) + (ell_X2**2).unsqueeze(0)
    C = torch.sqrt(2*(ell_X.unsqueeze(1) * ell_X2.unsqueeze(0))/ell_matrix)
    return scale2*C*torch.exp(-r2/ell_matrix)    


def MGP_d(K12, K22, d11, mu, Sigma):
    """
    compute the marginalized element-wise GP :
    assume Y|X: N(K12K22^-1X, K11-K12K22^-1K21) and X: N(mu, Sigma)
    Y N(mu_Y, Sigma_Y)
    param K12 of shape N,M
    param K22 of shape M,M
    param d11 of shape N
    mu of shape of  ...,M
    Sigma of shape of ...,M,M
    """
    K21 = K12.t()
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1]).type(TensorType).to(mu.device) * tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t() # K12K22^{-1}
    mu_Y = torch.matmul(P, mu.unsqueeze(-1))[...,0]
    # import pdb; pdb.set_trace()
    sigma2_Y = d11 - torch.mul(P, K12).sum(-1) + torch.mul(P.matmul(Sigma), P).sum(-1)
    z = torch.randn(mu_Y.size()).type(TensorType).to(mu.device)
    sample = reparameterize(mu_Y, sigma2_Y, z, full_cov=False)
    return sample


def MGP_mu_sigma2(K12, K22, d11, mu, Sigma):
    """
       compute the summary statistics of the marginalized element-wise GP :
       assume Y|X: N(K12K22^-1X, K11-K12K22^-1K21) and X: N(mu, Sigma)
       Y N(mu_Y, Sigma_Y)
       param K12 of shape N,M
       param K22 of shape M,M
       param d11 of shape N
       mu of shape of  ...,M
       Sigma of shape of ...,M,M
       output: mu_Y of shape of ...,N and sigma2_Y of shape of ...,N
       """
    K21 = K12.t()
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1]).type(TensorType).to(device)*tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t()  # K12K22^{-1}
    mu_Y = torch.matmul(P, mu.unsqueeze(-1))[..., 0]
    sigma2_Y = d11 - torch.mul(P, K12).sum(-1) + torch.mul(P.matmul(Sigma), P).sum(-1)
    # import pdb; pdb.set_trace()
    return mu_Y, sigma2_Y


def MGP_mu(K12, K22, mu, device0=None):
    if device0 is None:
        device0 = device
    K21 = K12.t()
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1]).type(TensorType).to(device0) * tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t()  # K12K22^{-1}
    # import pdb; pdb.set_trace()
    mu_Y = torch.matmul(P, mu.unsqueeze(-1))[..., 0]
    return mu_Y


def MGP(K12, K22, K11, mu, Sigma):
    """
    compute the marginalized Gaussian:
    assume Y|X: N(K12K22^-1X, K11-K12K22^-1K21) and X: N(mu, Sigma)
    Y N(mu_Y, Sigma_Y)
    param K12 of shape N,M
    param K22 of shape M,M
    param K11 of shape N.N
    mu of shape of  ...,N
    Sigma of shape of ...,N,N
    """
    K21 = K12.t()
    K22 = K22 + torch.eye(K22.shape[-1])*tridiagonal_jitter
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1]) * tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t() # K12K22^{-1}
    # if size == 1:
    #     mu_Y = P.matmul(mu)
    #     Sigma_Y = K11 - P.mm(K21) + P.mm(Sigma).mm(P.t())
    # if size == 2:
    #     mu_Y = torch.matmul(P[None,:, :], mu.unsqueeze(-1))[:,:,0]
    #     Sigma_Y = K11 - P.mm(K21) + P[None,:, :].matmul(Sigma).matmul(P[None,:, :].transpose(-2,-1))
    mu_Y = torch.matmul(P, mu.unsqueeze(-1))[...,0]
    Sigma_Y = K11 - P.mm(K21) + P.matmul(Sigma).matmul(P.transpose(-2,-1))
    z = torch.randn(mu_Y.size()).type(TensorType).to(device)
    # print(torch.diagonal(Sigma_Y, dim1= -2, dim2=-1))
    sample = reparameterize(mu_Y, Sigma_Y, z, full_cov=True)
    return sample


def JGP(K12, K22, K11, mu, Sigma):
    """
    sample the joint Gaussian (Y, X):
    assume Y|X: N(K12K22^-1X, K11-K12K22^-1K21) and X: N(mu, Sigma)
    Y N(mu_Y, Sigma_Y)
    param K12 of shape N,M
    param K22 of shape M,M
    param K11 of shape N.N
    mu of shape of  ...,M
    Sigma of shape of ...,M,M
    """
    K21 = K12.t()
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1])*tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t() # K12K22^{-1}
    mu_Y = torch.cat([torch.matmul(P, mu.unsqueeze(-1))[...,0], mu])
    B = K11 - P.mm(K21)
    S22 = Sigma
    S11 = P.mm(Sigma).mm(P.t()) + B
    S12 = P.mm(Sigma)
    S21 = S12.t()
    Sigma_Y = torch.cat([torch.cat([S11, S12], axis=1), torch.cat([S21, S22], axis=1)], axis=0)
    z = torch.randn(mu_Y.size()).type(TensorType).to(device)
    # print(torch.diagonal(Sigma_Y, dim1= -2, dim2=-1))
    sample = reparameterize(mu_Y, Sigma_Y, z, full_cov=True)
    return sample


def JGP_S(K11_diag, K12, K22, mu, Sigma):
    """
    compute the joint Gaussian (Y, X) and Y_i are mutually independent given X:
    assume Y|X: N(K12K22^-1X, K11-K12K22^-1K21) and X: N(mu, Sigma)
    param K11_diag of shape N
    param K12 of shape N,M
    param K22 of shape M,M
    mu of shape of  ...,M
    Sigma of shape of ...,M,M
    """
    z_v = torch.randn(mu.size()).type(TensorType).to(mu.device)
    sampled_v = reparameterize(mu, Sigma, z_v, full_cov=True)
    K21 = K12.t()
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1]).type(TensorType).to(mu.device)*tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t()  # K12K22^{-1}
    # P = torch.solve(A=K22, input=K21)[0].t()  # K12K22^{-1}
    mu_Y = torch.matmul(P, sampled_v.unsqueeze(-1))[..., 0]
    sigma2_Y = K11_diag - torch.sum(torch.mul(P, K12), axis=1)
    z = torch.randn(mu_Y.size()).type(TensorType).to(mu.device)
    sampled_tildeell = reparameterize(mu_Y, sigma2_Y, z, full_cov=False)
    sample = torch.cat([sampled_tildeell, sampled_v])
    return sample


def CGP(K12, K22, K11, X):
    """
    compute the conditional Gaussian:
    assume Y|X: N(K12K22^-1X, K11-K12K22^-1K21)
    Y N(mu_Y, Sigma_Y)
    param K12 of shape N,M
    param K22 of shape M,M
    param K11 of shape N.N
    mu of shape of  ...,N
    Sigma of shape of ...,N,N
    """
    K21 = K12.t()
    K22_err = K22 + torch.eye(K22.shape[0], K22.shape[1])*tridiagonal_jitter
    P = torch.solve(A=K22_err, input=K21)[0].t() # K12K22^{-1}
    # if size == 1:
    #     mu_Y = P.matmul(mu)
    #     Sigma_Y = K11 - P.mm(K21) + P.mm(Sigma).mm(P.t())
    # if size == 2:
    #     mu_Y = torch.matmul(P[None,:, :], mu.unsqueeze(-1))[:,:,0]
    #     Sigma_Y = K11 - P.mm(K21) + P[None,:, :].matmul(Sigma).matmul(P[None,:, :].transpose(-2,-1))
    mu_Y = torch.matmul(P, X.unsqueeze(-1))[...,0]
    Sigma_Y = K11 - P.matmul(K12.t())
    z = torch.randn(mu_Y.size()).type(TensorType).to(device)
    # print(torch.diagonal(Sigma_Y, dim1= -2, dim2=-1))
    sample = reparameterize(mu_Y, Sigma_Y, z, full_cov=True)
    return sample


def Normal_logprob(loc, scale, y):
    var = scale**2
    log_scale = torch.log(scale)
    log_pdf = -((y - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    return torch.sum(log_pdf)


def log_determinant_halfpower(K):
    scale_tril = torch.cholesky(K)
    return scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def batch_trace_XXT(bmat):
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    n = bmat.size(-1)
    m = bmat.size(-2)
    flat_trace = bmat.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(bmat.shape[:-2])


def batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.
    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]
    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)
    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0].pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b
    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def KL_Gaussian(X_mu, X_Sigma, X2_mu, X2_Sigma, device0=None):
    """
    Compute Kullback-Leibler divergence of multivaraite Gaussian distributions
    X_mu: mean of Gaussian distributions of shape ...,N
    X_Sigma: Variance of Gaussian distributions of shape ...,N,N
    X2_mu: mean of a Gaussian distribution of shape N
    X2_Sigma: Variance of a Gaussian distribution of shape N,N
    """
    # import pdb; pdb.set_trace()
    if device0 is None:
        device0 = device
    X_Sigma = X_Sigma + torch.eye(X_Sigma.shape[-1]).type(TensorType).to(device0)*tridiagonal_jitter
    X2_Sigma = X2_Sigma + torch.eye(X2_Sigma.shape[-1]).type(TensorType).to(device0)*tridiagonal_jitter
    n = X_mu.shape[-1]
    half_term1 = log_determinant_halfpower(X2_Sigma) - log_determinant_halfpower(X_Sigma)
    scale_tril_X = torch.cholesky(X_Sigma)
    scale_tril_X2 = torch.cholesky(X2_Sigma) 
    term2 = batch_trace_XXT(torch.triangular_solve(input = scale_tril_X, A=scale_tril_X2)[0])
    term3 = batch_mahalanobis(scale_tril_X2, X2_mu - X_mu)
    return half_term1 + 0.5*(term2 + term3 - n)


def plot_corrs(x, sampled_corrs, dim1=0, dim2=1, directory=None):
    correlations_mean = np.mean(sampled_corrs[:, :, dim1, dim2], axis=0)
    correlations_quantiles = np.quantile(sampled_corrs[:, :, dim1, dim2], q=(0.025, 0.975), axis=0)
    fig = plt.figure()
    plt.plot(x, correlations_mean, color="b")
    plt.plot(x, correlations_quantiles.T, color="r", linestyle="dashed")
    if directory is None:
        fig.savefig("correlation_dim{}vsdim{}.png".format(dim1, dim2))
    else:
        fig.savefig(directory + "/correlation_dim{}vsdim{}.png".format(dim1, dim2))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # # test reparameterize
    # mean = torch.rand(5,5)
    # q_sqrt = torch.nn.Parameter(torch.randn(5, 5, 5))
    # q_sqrt0 = mat2ltri(q_sqrt)
    # z = torch.randn(5,5)
    # f = reparameterize(mean, torch.matmul(q_sqrt0, q_sqrt0.permute(0, 2, 1)), z, full_cov=True)  
    # obj = torch.sum(f)
    # print(obj)
    # obj.backward()
    # print(q_sqrt.grad)

    # # test create_RBF
    # N0 = 5
    # N1 = 10
    # M = 3
    # X0 = torch.randn(N0, M)
    # X1 = torch.randn(N1, M)
    # C = create_RBF(X0, X1)
    # print(C)

    # # test MGP and MGP_d
    # N0 = 10
    # N1 = 5
    # M = 3
    # D = 20
    # X0 = torch.randn(N0, M).type(TensorType)
    # X1 = torch.randn(N1, M).type(TensorType)
    # K12 = create_RBF(X0, X1)
    # K22 = create_RBF(X1)
    # K11 = create_RBF(X0)
    # mu = torch.randn(D, N1).type(TensorType)
    # q_sqrt = torch.nn.Parameter(torch.randn(D, N1, N1).type(TensorType))
    # q_sqrt0 = mat2ltri(q_sqrt)
    # Sigma = torch.matmul(q_sqrt0, q_sqrt0.transpose(-2,-1))
    # sample = MGP(K12, K22, K11, mu, Sigma)
    # print(sample.shape)
    # sample = MGP_d(K12, K22, torch.diag(K11), mu, Sigma)
    # print(sample.shape)


    import pdb
    pdb.set_trace()



    