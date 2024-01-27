"""
Empirical estimation for model parameters via window size estimation.
"""
import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from skgstat import Variogram
# %set_env SKG_SUPPRESS=true
import os
# os.environ["SKG_SUPPRESS"]="True"

# import private libraries
from . import utils
from . import settings

def load_syndata():
    # Load synthetic data
    # with open("../data/sim/sim_MNTS.pickle", "rb") as res:
    with open("../data/sim_large/sim_MNTS.pickle", "rb") as res:
        x, true_l, true_L_vecs, true_sigma2_err, Y = pickle.load(res)
    trend = 0.
    scale = 1.
    x_scale = 1.
    x_train = x
    Y_train = Y
    x_test = None
    Y_test = None
    return x_train, Y_train, x_test, Y_test, trend, scale, x_scale


def SV(x, Y, indx):
    '''
    Experimental variogram for a collection of lags
    '''
    # V = Variogram(x, Y[:, indx], normalize=False)
    # xdata = V.bins
    # ydata = V.experimental
    # fig = plt.figure()
    # V.plot()
    # plt.savefig("check.png")
    # import pdb
    # pdb.set_trace()

    sv = list()
    lag = list()
    N = x.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            lag.append(x[j] - x[i])
            sv.append(0.5*np.mean((Y[j, indx] - Y[i, indx])**2))
    return np.array(lag), np.array(sv)
    # return xdata, ydata


def variogram_Gaussian(s, sigma, l):
    return sigma**2*(1 - np.exp(-0.5*s**2/l**2))


def global_estimation(x, Y):
    _, M = Y.shape
    S = np.cov(Y.T)
    L_f = np.linalg.cholesky(S)
    L_vec = utils.lowtriangle2vec(L_f, M)
    return S, L_vec


def local_estimation(x, Y, window_size=30, save_dir=None, folder_name=None, subfolder_name=None, check=False):
    N, M = Y.shape
    est_sigmas = list()
    est_ls = list()
    est_B = list()
    est_L_vecs = list()
    est_stds = list()
    est_R = list()
    # estimate lengthscale process and covariance process
    for n in range(N):
        start = max(0, n - window_size)
        end = min(n + window_size, N-1)
        xn_seg = x[start: end]
        Yn_seg = Y[start: end]
        cofs = list()
        for m in range(M):
            lag, sv = SV(xn_seg, Yn_seg, m)
            cof_u, cov = curve_fit(variogram_Gaussian, lag, sv, maxfev=2000)
            cofs.append(cof_u)
            if check:
                yi = np.array(list(map(lambda x: variogram_Gaussian(x, *cof_u), lag)))
                fig = plt.figure()
                plt.plot(lag, sv, 'rD')
                order = np.argsort(lag)
                plt.plot(lag[order], yi[order], '--r')
                if subfolder_name is None:
                    plt.savefig(save_dir + folder_name + "check.png")
                else:
                    plt.savefig(save_dir + folder_name + subfolder_name + "check.png")
                plt.close(fig)
                import pdb
                pdb.set_trace()
        cof = np.mean(np.stack(cofs), axis=0)
        est_sigmas.append(np.abs(cof[0]))
        est_ls.append(np.abs(cof[1]))
        try:
            # S = np.cov(Yn_seg.T).rehshape(M,M)
            S = np.matmul(Yn_seg.T, Yn_seg)/(Yn_seg.shape[0]-1)
            est_B.append(S)
            est_L_f = np.linalg.cholesky(S)
        except:
            # S = np.cov(Yn_seg.T).reshape(M,M) + np.diag(np.ones(M) * settings.precision)
            S = np.matmul(Yn_seg.T, Yn_seg)/(Yn_seg.shape[0]-1) + np.diag(np.ones(M) * settings.precision)
            est_B.append(S)
            est_L_f = np.linalg.cholesky(S)
        est_L_vec = utils.lowtriangle2vec(est_L_f, M).reshape(-1)
        est_L_vecs.append(est_L_vec)
        D = np.sqrt(np.diag(S)) 
        est_stds.append(D)
        est_R.append(np.diag(1./D).dot(S).dot(np.diag(1./D)))
    est_B = np.stack(est_B) # dim N, M, M
    est_R = np.stack(est_R) # dim N, M, M
    est_stds = np.stack(est_stds)
    est_tilde_sigma2_err = -4
    est_ls = np.array(est_ls)
    smooth_ls = list()
    # smooth est_ls
    for n in range(N):
        start = max(0, n - 10)
        end = min(n + 10, N-1)
        smooth_ls.append(np.mean(est_ls[start: end]))
    smooth_ls = np.array(smooth_ls) 
    return np.array(est_sigmas), est_ls, smooth_ls, est_stds, est_R, est_B, np.concatenate(est_L_vecs), est_tilde_sigma2_err


def visualization(x, Y, est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=None, folder_name=None, subfolder_name=None, attributes=None):
    N, M = Y.shape
    if attributes is None:
        attributes = ["Dim {}".format(m+1) for m in range(M)]
    fig = plt.figure()
    plt.plot(x, np.log(est_ls))
    plt.plot(x, np.log(smooth_ls))
    # plt.savefig(save_dir + folder_name_separable + "empirical_log_l.png")
    if subfolder_name is not None:
        plt.savefig(save_dir + folder_name + subfolder_name + "empirical_log_l.png")
    else:
        plt.savefig(save_dir + folder_name + "empirical_log_l.png")
    plt.close(fig)
    # fig = plt.figure()
    # plt.plot(x, est_sigmas)
    # plt.savefig(save_dir + folder_name_separable + "empirical_sigma.png")
    # plt.close(fig)
    fig = plt.figure()
    for m in range(M):
        plt.plot(x, est_stds[:, m], label="Dim {}".format(m + 1))
    plt.legend()
    if subfolder_name is not None:
        plt.savefig(save_dir + folder_name + subfolder_name + "empirical_std.png")
    else:
        plt.savefig(save_dir + folder_name + "empirical_std.png")
    plt.close(fig)
    for i in range(M):
        for j in range(i+1, M):
            fig = plt.figure()
            plt.plot(x, est_R[:, i, j])
            if subfolder_name is not None:
                plt.savefig(save_dir + folder_name + subfolder_name + "empirical_R_{}_{}.png".format(attributes[i], attributes[j]))
            else:
                plt.savefig(save_dir + folder_name + "empirical_R_{}_{}.png".format(attributes[i], attributes[j]))
            plt.close(fig)
    fig = plt.figure()
    est_L_f = est_L_vecs.reshape([-1, int(M * (M + 1) / 2)])
    for i in range(M):
        for j in range(i, M):
            plt.plot(x, est_L_f[:, int(i * (i + 1) / 2) + j], label="L_{}_{}".format(attributes[i], attributes[j]))
    plt.legend()
    if subfolder_name is not None:
        plt.savefig(save_dir + folder_name + subfolder_name + "empirical_L.png")
    else:
        plt.savefig(save_dir + folder_name + "empirical_L.png")
    plt.close(fig)


def save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=None, folder_name=None, subfolder_name=None):
    if subfolder_name is not None:
        with open(save_dir + folder_name + subfolder_name + "empirical_est.pickle", "wb") as res:
            pickle.dump([np.log(est_ls), np.log(smooth_ls), est_L_vecs, est_tilde_sigma2_err], res)
    else:
        with open(save_dir + folder_name + "empirical_est.pickle", "wb") as res:
            pickle.dump([np.log(est_ls), np.log(smooth_ls), est_L_vecs, est_tilde_sigma2_err], res)


if __name__ == "__main__":
    save_dir = "../res/"
    # folder_name_separable = "sim_separable/"
    # folder_name_nonseparable = "sim_nonseparable/"
    folder_name_separable = "sim_large_separable/"
    folder_name_nonseparable = "sim_large_nonseparable/"
    x, Y, _, _, _, _, _ = load_syndata()

    est_sigmas, est_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = local_estimation(x, Y)
    # est_S, est_L_vec = global_estimation(x, Y)

    visualization(x, Y, est_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir, folder_name=folder_name_nonseparable)
    save_res(est_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name_nonseparable)


