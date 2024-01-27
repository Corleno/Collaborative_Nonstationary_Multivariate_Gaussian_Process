# Simulate multi-output data
import numpy as np
from scipy.stats import multivariate_normal
import pyGPs
import pickle
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# import private libraries
from Utility import kernels
from Utility import settings
from Utility import logpos
from Utility import kronecker_operation

def SIM_MNTS(M, N, save_dir=None, folder_name=None, file_name="sim_MNTS.pickle", verbose=True, seed = 0):
    # np.random.seed(22)
    # torch.manual_seed(22)
    # Generate N time stamps on (0,1)
    x = torch.from_numpy(np.sort(np.random.rand(N))).type(settings.torchType)

    # Generate length-scale function ##### a determinant function
    tilde_l = 3*(x-1)**3 - 3
    l = torch.exp(tilde_l)
    if verbose:
        fig = plt.figure()
        plt.plot(x.numpy(), tilde_l.numpy())
        plt.savefig(save_dir + folder_name + "true_log_l.png")
        plt.close(fig)

    # Generate std processes ##### determinant functions
    stds = torch.stack([1+x**2, 2-x**2], axis=1)
    std_array = stds.numpy()
    if verbose:
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), std_array[:, m], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "std.png")
        plt.close(fig)

    # Generate correlation process via a cos function ##### a determinant function
    cors = torch.cos(x*np.pi)
    if verbose:
        fig = plt.figure()
        plt.plot(x.numpy(), cors.numpy())
        plt.savefig(save_dir + folder_name + "true_log_R_{}{}.png".format(0, 1))
        plt.close(fig)

    # generate covariance processes via combining std process and correlation process
    L_f_list = []
    for n in range(N):
        D_f = torch.diag(stds[n, :])
        R_f = torch.eye(M).type(settings.torchType)
        R_f[0, 1] = cors[n]
        R_f[1, 0] = cors[n]
        B_f = D_f.mm(R_f).mm(D_f)
        L_f = torch.cholesky(B_f)
        L_f_list.append(L_f)
    L_vecs = torch.cat([L_f[[0, 1, 1], [0, 0, 1]] for L_f in L_f_list])

    # Generate sigma2_err 
    sigma2_err = 1e-2

    # Generate y
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)
    K_i = logpos.generate_K_index_SVC(L_f_list)
    neworder = torch.arange(N*M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder]
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
    torch.manual_seed(seed)
    y = MultivariateNormal(loc=torch.zeros(M*N).type(settings.torchType), covariance_matrix=K + sigma2_err *
                                            torch.diag(torch.ones(M*N).type(settings.torchType))).sample()
    Y = y.view([M, N])
    if verbose:
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), Y.numpy()[m, :], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "Y.png")
        plt.close(fig)

    with open(save_dir + folder_name + file_name, "wb") as res:
        pickle.dump([x.numpy(), l.numpy(), L_vecs.numpy(), sigma2_err, Y.numpy().T], res)
    return Y

def SIM_illustration_varying_freq(save_dir=None, folder_name=None, file_name="sim_illustration_varying_freq.pickle", verbose=True, seed = 22):
    np.random.seed(seed)
    # Generate time stamps
    X1 = np.random.rand(100)[:, None];
    X1 = X1 * 0.8
    X2 = np.random.rand(100)[:, None];
    X2 = X2 * 0.8 + 0.2
    Xt1 = np.random.rand(100)[:, None]
    Xt2 = np.random.rand(100)[:, None]
    X_list = [X1, X2]
    Xt_list = [Xt1, Xt2]

    f_output1 = lambda x: 5 * np.cos(2*np.pi*x*x*5) + np.random.rand(x.size)[:,None]
    f_output2 = lambda x: 5 * ((1 - x)*np.cos(2*np.pi*x*x*5) - x*np.cos(2*np.pi*x*x*5)) + np.random.rand(x.size)[:,None]
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)
    Y_list = [Y1, Y2]
    Yt_list = [Yt1, Yt2]

    plt.figure()
    plt.scatter(X1, Y1, label="Train set")
    plt.scatter(Xt1, Yt1, label="Test set")
    plt.title("Output 1")
    plt.legend()
    plt.show()
    plt.figure()
    plt.scatter(X2, Y2, label="Train set")
    plt.scatter(Xt2, Yt2, label="Test set")
    plt.title("Output 2")
    plt.legend()
    plt.show()

    with open(save_dir + folder_name + file_name, "wb") as res:
        pickle.dump([X_list, Y_list, Xt_list, Yt_list], res)

    return X_list, Y_list, Xt_list, Yt_list

def SIM_illustration_low_freq(save_dir=None, folder_name=None, file_name="sim_illustration_low_freq.pickle", verbose=True, seed = 22):
    np.random.seed(seed)
    # Generate time stamps
    X1 = np.random.rand(100)[:, None];
    X1 = X1 * 0.8
    X2 = np.random.rand(100)[:, None];
    X2 = X2 * 0.8 + 0.2
    Xt1 = np.random.rand(100)[:, None]
    Xt2 = np.random.rand(100)[:, None]
    X_list = [X1, X2]
    Xt_list = [Xt1, Xt2]

    f_output1 = lambda x: 5 * np.cos(2*np.pi*x*2) + np.random.rand(x.size)[:,None]
    f_output2 = lambda x: 5 * ((1 - x)*np.cos(2*np.pi*x*2) - x*np.cos(2*np.pi*x*2)) + np.random.rand(x.size)[:,None]
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)
    Y_list = [Y1, Y2]
    Yt_list = [Yt1, Yt2]

    plt.figure()
    plt.scatter(X1, Y1, label="Train set")
    plt.scatter(Xt1, Yt1, label="Test set")
    plt.title("Output 1")
    plt.legend()
    plt.show()
    plt.figure()
    plt.scatter(X2, Y2, label="Train set")
    plt.scatter(Xt2, Yt2, label="Test set")
    plt.title("Output 2")
    plt.legend()
    plt.show()

    with open(save_dir + folder_name + file_name, "wb") as res:
        pickle.dump([X_list, Y_list, Xt_list, Yt_list], res)

    return X_list, Y_list, Xt_list, Yt_list

def SIM_illustration_high_freq(save_dir=None, folder_name=None, file_name="sim_illustration_high_freq.pickle", verbose=True, seed = 22):
    np.random.seed(seed)
    # Generate time stamps
    X1 = np.random.rand(100)[:, None];
    X1 = X1 * 0.8
    X2 = np.random.rand(100)[:, None];
    X2 = X2 * 0.8 + 0.2
    Xt1 = np.random.rand(100)[:, None]
    Xt2 = np.random.rand(100)[:, None]
    X_list = [X1, X2]
    Xt_list = [Xt1, Xt2]

    f_output1 = lambda x: 5 * np.cos(2*np.pi*x*5) + np.random.rand(x.size)[:,None]
    f_output2 = lambda x: 5 * ((1 - x)*np.cos(2*np.pi*x*5) - x*np.cos(2*np.pi*x*5)) + np.random.rand(x.size)[:,None]
    Y1 = f_output1(X1)
    Y2 = f_output2(X2)
    Yt1 = f_output1(Xt1)
    Yt2 = f_output2(Xt2)
    Y_list = [Y1, Y2]
    Yt_list = [Yt1, Yt2]

    plt.figure()
    plt.scatter(X1, Y1, label="Train set")
    plt.scatter(Xt1, Yt1, label="Test set")
    plt.title("Output 1")
    plt.legend()
    plt.show()
    plt.figure()
    plt.scatter(X2, Y2, label="Train set")
    plt.scatter(Xt2, Yt2, label="Test set")
    plt.title("Output 2")
    plt.legend()
    plt.show()

    with open(save_dir + folder_name + file_name, "wb") as res:
        pickle.dump([X_list, Y_list, Xt_list, Yt_list], res)

    return X_list, Y_list, Xt_list, Yt_list

if __name__ == "__main__":
    # simulate nonstationary multivariate time series with varying covariance process
    hyper_pars = {"mu_tilde_l": -3, "alpha_tilde_l": 3., "beta_tilde_l": 0.4, "mu_L": 0., "alpha_L": 5., "beta_L": 1
        , "a": 1., "b": 1.}

    save_dir = "../../data/"
    folder_name = "simulation/"

    # N = 200
    # M = 2
    # ts = SIM_MNTS(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS.pickle", seed=2222, verbose=True, **hyper_pars)
    
    # N_rep = 100
    # for n in range(N_rep):
    #     print(n)
    #     ts = SIM_MNTS(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS_{}.pickle".format(n), seed=n, verbose=False, **hyper_pars)

    # N = 2000
    # M = 2
    # ts = SIM_MNTS(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS_L.pickle", seed=2222,
    #               verbose=True)

    # N_rep = 10
    # for n in range(N_rep):
    #     print(n)
    #     ts = SIM_MNTS_S(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS_L_{}.pickle".format(n), seed=n, verbose=False)

    # N = 200
    # M = 2
    # ts = SIM_MNTS(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS.pickle", seed=2222, verbose=True, **hyper_pars)

    ts = SIM_illustration_varying_freq(save_dir=save_dir, folder_name=folder_name)

    # ts = SIM_illustration_low_freq(save_dir=save_dir, folder_name=folder_name)

    # ts = SIM_illustration_high_freq(save_dir=save_dir, folder_name=folder_name)
