"""
Visualize posterior estimates and posterior samples
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from . import utils


def vec2pars_est(pars, N, M): # SNMGP
    """
    Convert pars to tilde_l, tilde_sigma, L_vec, tilde_sigma2_err
    """
    tilde_l = pars[:N]
    tilde_sigma = pars[N:2*N]
    L_vec = pars[2*N:2*N+int(M*(M+1)/2)]
    tilde_sigma2_err = pars[-1]
    return tilde_l, tilde_sigma, L_vec, tilde_sigma2_err


def vec2pars_est_SVC(pars, N): # GNMGP
    """
    Convert pars to tilde_l, tilde_sigma, L_vec, tilde_sigma2_err
    """
    tilde_l = pars[:N]
    L_vecs = pars[N:-1]
    tilde_sigma2_err = pars[-1]
    return tilde_l, L_vecs, tilde_sigma2_err


def vec2pars_est_S(pars, M): # LMC
    """
    Convert pars to tilde_l, tilde_sigma, L_vec, tilde_sigma2_err
    """
    tilde_l = pars[:1]
    tilde_sigma = pars[1:2]
    L_vec = pars[2:2+int(M*(M+1)/2)]
    tilde_sigma2_err = pars[-1]
    return tilde_l, tilde_sigma, L_vec, tilde_sigma2_err


def cov2cor(S):
    """
    Convert covariance matrix to correlation matrix
    :param S: 2d array with dim N, N
    :return: 2d array with dim N, N
    """
    d = np.sqrt(np.diagonal(S))
    DInv = np.diag(1./d)
    R = np.matmul(np.matmul(DInv, S), DInv)
    return R


def vec2pars(pars_hist, N, M): #SNMGP
    """
    Convert pars_hist to tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist
    """
    tilde_l_hist = pars_hist[:, :N]
    tilde_sigma_hist = pars_hist[:, N:2*N]
    L_vec_hist = pars_hist[:, 2*N:2*N+int(M*(M+1)/2)]
    tilde_sigma2_err_hist = pars_hist[:, -1]
    return tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist


def vec2pars_SVC(pars_hist, N, M): #GNMGP
    """
    Convert pars_hist to tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist
    """
    tilde_l_hist = pars_hist[:, :N]
    L_vecs_hist = pars_hist[:, N: N + N*int(M * (M + 1) / 2)]
    tilde_sigma2_err_hist = pars_hist[:, -1]
    return tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist

def vec2pars_S(pars_hist, M): # LMC
    """
    Convert pars to tilde_l, tilde_sigma, L_vec, tilde_sigma2_err
    """
    tilde_l = pars_hist[:, :1]
    tilde_sigma = pars_hist[:, 1:2]
    L_vec = pars_hist[:, 2:2+int(M*(M+1)/2)]
    tilde_sigma2_err = pars_hist[:, -1]
    return tilde_l, tilde_sigma, L_vec, tilde_sigma2_err


def samples2quantiles(pos_sample, percentiles=[2.5, 50., 97.5]):
    """
    compute posterior sample point-wise quantiles
    :param pos_sample: 2d array with dim N_samples, N
    :param percentiles: list with size N_percentiles
    :return: 2d array with dim N_percentiles, N
    """
    res = np.percentile(pos_sample, q=percentiles, axis=0)
    return res


def plot_mean_and_CI(x, mean, lb, ub, color_mean="b", color_shading="r"):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb, color=color_shading, alpha=.5, label="predictive 95% confidence interval")
    # plot the mean on top
    plt.plot(x, mean, color=color_mean, label="predictive mean")


def visualization_pos(x, tilde_l_hist, tilde_sigma_hist=None, L_vecs_hist=None, N=None, M=None, save_dir=None, folder_name=None, subfolder_name=None, attributes=None):
    """
    Visualize the posterior of lengthscale and scale parameters on a log scale.
    :param x: 1d array with length N
    :param tilde_l_hist: 2d array with dim N_hist, N
    :param tilde_sigma_hist: 2d array with dim N_hist, N
    :param L_vecs_hist: 2d array with dim N_hist, N*M*(M+1)/2
    :return: None
    """
    # reoder x
    order = np.argsort(x)
    x = x[order]
    # compute percentiles for tilde_l_hist and tilde_sigma_hist
    tilde_l_hist = tilde_l_hist[:, order]
    tilde_l_percentiles = samples2quantiles(tilde_l_hist)
    if tilde_sigma_hist is not None:
        tilde_sigma_hist = tilde_sigma_hist[:, order]
        tilde_sigma_percentiles = samples2quantiles(tilde_sigma_hist)
    if L_vecs_hist is not None:
        L_vec_hist_list = [[L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)] for L_vecs in L_vecs_hist]
        # import pdb
        # pdb.set_trace()
        L_f_hist_list =[np.stack([utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list])[order] for L_vec_list in L_vec_hist_list]
        B_f_hist_array = np.stack([np.stack([np.matmul(L_f, L_f.T) for L_f in L_f_list]) for L_f_list in L_f_hist_list])
        R_f_hist_array = np.stack([np.stack([cov2cor(np.matmul(L_f, L_f.T)) for L_f in L_f_list]) for L_f_list in L_f_hist_list])
        B_f_means = np.mean(B_f_hist_array, axis=0) # T, M, M
        B_f_stds = np.std(B_f_hist_array, axis=0)
        R_f_means = np.mean(R_f_hist_array, axis=0) # T, M, M
        R_f_stds = np.std(R_f_hist_array, axis=0)

    fig = plt.figure()
    plt.plot(x, tilde_l_percentiles[1, :], color='b')
    plt.plot(x, tilde_l_percentiles[[0, 2], :].T, color='r')
    if subfolder_name is None:
        plt.savefig(save_dir + folder_name + "pos_logl_HMC.png")
    else:
        plt.savefig(save_dir + folder_name + subfolder_name + "pos_logl_HMC.png")
    plt.close(fig)
    if tilde_sigma_hist is not None:
        fig = plt.figure()
        plt.plot(x, tilde_sigma_percentiles[1, :], color='b')
        plt.plot(x, tilde_sigma_percentiles[[0, 2], :].T, color='r')
        if subfolder_name is None:
            plt.savefig(save_dir + folder_name + "pos_logsigma_HMC.png")
        else:
            plt.savefig(save_dir + folder_name + subfolder_name + "pos_logsigma_HMC.png")
        plt.close(fig)
    if L_vecs_hist is not None:
        for i in range(M):
            for j in range(i, M):
                fig = plt.figure()
                plot_mean_and_CI(x, B_f_means[:, i, j], B_f_means[:, i, j] - 2 * B_f_stds[:, i, j], B_f_means[:, i, j] + 2 * B_f_stds[:, i, j])
                if subfolder_name is None:
                    plt.savefig(save_dir + folder_name + "Covariance_{}_{}_HMC.png".format(attributes[i], attributes[j]))
                else:
                    plt.savefig(save_dir + folder_name + subfolder_name + "Covariance_{}_{}_HMC.png".format(attributes[i], attributes[j]))
                plt.close(fig)
                fig = plt.figure()
                plot_mean_and_CI(x, R_f_means[:, i, j], R_f_means[:, i, j] - 2 * R_f_stds[:, i, j], R_f_means[:, i, j] + 2 * R_f_stds[:, i, j])
                plt.xlabel('time (hour)', fontsize=22)
                plt.ylabel('correlation coefficient', fontsize=22)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.legend(fontsize=16)
                plt.tight_layout()
                if subfolder_name is None:
                    plt.savefig(save_dir + folder_name + "Correlation{}_{}_HMC.png".format(attributes[i], attributes[j]))
                else:
                    plt.savefig(save_dir + folder_name + subfolder_name + "Correlation_{}_{}_HMC.png".format(attributes[i], attributes[j]))
                plt.close(fig)


def visualization_pos_map(x, tilde_l, L_vecs=None, N=None, M=None, save_dir=None, folder_name=None, subfolder_name=None, attributes=None):
    # reoder x
    order = np.argsort(x)
    x = x[order]
    # compute percentiles for tilde_l_hist and tilde_sigma_hist
    tilde_l = tilde_l[order]
    if L_vecs is not None:
        L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        L_f_list = np.stack([utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list])[order]
        B_fs = np.stack([np.matmul(L_f, L_f.T) for L_f in L_f_list]) # size N, M, M
        R_fs = np.stack([cov2cor(np.matmul(L_f, L_f.T)) for L_f in L_f_list]) # size N, M, M

    fig = plt.figure()
    plt.plot(x, tilde_l, color='b')
    if subfolder_name is None:
        plt.savefig(save_dir + folder_name + "pos_logl_MAP.png")
    else:
        plt.savefig(save_dir + folder_name + subfolder_name + "pos_logl_MAP.png")
    plt.close(fig)

    if L_vecs is not None:
        for i in range(M):
            for j in range(i, M):
                fig = plt.figure()
                plt.plot(x, B_fs[:, i, j])
                if subfolder_name is None:
                    plt.savefig(save_dir + folder_name + "Covariance_{}_{}_MAP".format(attributes[i], attributes[j]))
                else:
                    plt.savefig(save_dir + folder_name + subfolder_name + "Covariance_{}_{}_MAP".format(attributes[i], attributes[j]))
                plt.close(fig)
                fig = plt.figure()
                plt.plot(x, R_fs[:, i, j])
                if subfolder_name is None:
                    plt.savefig(
                        save_dir + folder_name + "Correlation_{}_{}_MAP".format(attributes[i], attributes[j]))
                else:
                    plt.savefig(
                        save_dir + folder_name + subfolder_name + "Correlation_{}_{}_MAP".format(attributes[i], attributes[j]))
                plt.close(fig)


def visualization_pos_map_heatmap(x, L_vecs=None, N=None, M=None, save_dir=None, folder_name=None, subfolder_name=None, attributes=None):
    order = np.argsort(x)
    x = x[order]
    # compute percentiles for tilde_l_hist and tilde_sigma_hist
    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    L_f_list = np.stack([utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list])[order]
    # B_fs = np.stack([np.matmul(L_f, L_f.T) for L_f in L_f_list])  # size N, M, M
    R_fs = np.stack([cov2cor(np.matmul(L_f, L_f.T)) for L_f in L_f_list])  # size N, M, M
    if not os.path.exists(save_dir + folder_name + subfolder_name + "correlation_process"):
        os.mkdir(save_dir + folder_name + subfolder_name + "correlation_process")
    for n in range(N):
        df_cor = pd.DataFrame(R_fs[n], columns=attributes, index=attributes)
        fig = plt.figure()
        ax = sns.heatmap(
            df_cor,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        plt.savefig(save_dir + folder_name + subfolder_name + "correlation_process/{}.png".format(x[n]))
        plt.close(fig)

def visualization_pos_map_heatmap_withR_s(x, R_fs=None, save_dir=None, folder_name=None, subfolder_name=None, attributes=None):
    order = np.argsort(x)
    x = x[order]
    if not os.path.exists(save_dir + folder_name + subfolder_name + "correlation_process"):
        os.mkdir(save_dir + folder_name + subfolder_name + "correlation_process")
    for n in range(len(x)):
        df_cor = pd.DataFrame(R_fs[n], columns=attributes, index=attributes)
        fig = plt.figure()
        ax = sns.heatmap(
            df_cor,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        plt.savefig(save_dir + folder_name + subfolder_name + "correlation_process/{}.png".format(x[n]))
        plt.close(fig)

if __name__ == "__main__":
    pass
