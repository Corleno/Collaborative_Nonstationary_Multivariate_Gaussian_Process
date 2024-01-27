"""
Generate figures for model validation of synthetic data
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

# import private libraries
import sys
sys.path.append("..")
from Utility import posterior_analysis
from Utility import utils

if __name__ == "__main__":
    save_dir = "../../res/"
    data_dir = "../../data/simulation/"
    folder_MAP = "sim_MCMC/"
    folder_name = "sim_validation/"

    do_obs_plots = True
    do_MCMC_plot = False

    # Load synthetic data
    with open(data_dir + "sim_MNTS.pickle", "rb") as res:
        x, true_l, true_L_vecs, true_sigma2_err, Y = pickle.load(res)
    true_tilde_l = np.log(true_l)
    N, M = Y.shape
    true_L_vec_list = [true_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    true_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in true_L_vec_list]
    true_B_list = [np.matmul(L_f, L_f.T) for L_f in true_L_f_list]
    true_R_list = [posterior_analysis.cov2cor(B) for B in true_B_list]
    true_stds = np.stack([np.sqrt(np.diag(B)) for B in true_B_list])
    true_R_01 = np.cos(x * np.pi)

    # plot different runs
    indexes = np.arange(10)
    for index in indexes:
        with open(data_dir + "sim_MNTS_{}.pickle".format(index), "rb") as res:
            x, true_l, true_L_vecs, true_sigma2_err, Y = pickle.load(res)
        fig = plt.figure()
        plt.plot(x, Y)
        fig.savefig(save_dir + folder_MAP + "runs/{}_obs.png".format(index))

    # explore MCMC results
    if do_MCMC_plot:
        # Load initialization results
        with open(save_dir + folder_MAP + "empirical_est.pickle", "rb") as res:
            init_tilde_l, smooth_init_tilde_l, init_L_vecs, init_tilde_sigma2_err = pickle.load(res)
        init_L_vec_list = [init_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        init_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in init_L_vec_list]
        init_B_list = [np.matmul(L_f, L_f.T) for L_f in init_L_f_list]
        init_R_list = [posterior_analysis.cov2cor(B) for B in init_B_list]
        init_stds = np.stack([np.sqrt(np.diag(B)) for B in init_B_list])
        init_R_01 = np.stack([R_f[0, 1] for R_f in init_R_list])

        # Load inference results
        with open(save_dir + folder_MAP + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_uL_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_L_vecs = utils.uLvecs2Lvecs(est_uL_vecs, N, M)
        est_L_vec_list = [est_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        est_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in est_L_vec_list]
        est_B_list = [np.matmul(L_f, L_f.T) for L_f in est_L_f_list]
        est_R_list = [posterior_analysis.cov2cor(B) for B in est_B_list]
        est_stds = np.stack([np.sqrt(np.diag(B)) for B in est_B_list])
        est_R_01 = np.stack([R_f[0, 1] for R_f in est_R_list])

        # Load predictive results
        with open(save_dir + folder_MAP + "pred_grid_map.pickle", "rb") as res:
            gridy_quantiles, gridy_mean, gridy_std = pickle.load(res)
        gridy_quantiles = gridy_quantiles
        grids = np.linspace(0., 1., 201)

        # Plot prediction results.
        # for m in range(M):
        #     fig = plt.figure()
        #     plt.plot(x, Y[:, m], label="true")
        #     plt.plot(grids, gridy_quantiles[:, 1, m], label="predicted")
        #     plt.fill_between(grids, y1=gridy_quantiles[:, 0, m], y2=gridy_quantiles[:, 2, m], color="r", alpha=0.2, label="predictive 95% confidence interval")
        #     plt.legend()
        #     plt.xlabel("x", fontsize=18)
        #     plt.ylabel("y", fontsize=18)
        #     plt.savefig(save_dir + folder_name + "predictive_process_D{}.png".format(m+1))
        #     plt.close(fig)

        # Plot real data results
        for m in range(M):
            fig = plt.figure()
            plt.plot(x, Y[:, m], '--')
            plt.scatter(x, Y[:, m], color='r')
            plt.xlabel("x", fontsize=22)
            plt.ylabel("y", fontsize=22)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(save_dir + folder_name + "synthetic_data_dim{}.png".format(m+1))
            plt.close(fig)

        # Plot log lengthscale process
        fig = plt.figure()
        plt.plot(x, true_tilde_l, label="true")
        plt.plot(x, init_tilde_l, '--' , label="initialized")
        plt.plot(x, est_tilde_l, '--' , label="inferred")
        plt.legend(fontsize=22)
        plt.xlabel("x", fontsize=22)
        plt.ylabel("log lengthscalse", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(save_dir + folder_name + "predictive_log_lengthscale_process")
        plt.close(fig)

        # Plot correlation process
        fig = plt.figure()
        plt.plot(x, true_R_01, label="true")
        plt.plot(x, init_R_01, '--', label="initialized")
        plt.plot(x, est_R_01, '--', label="inferred")
        plt.legend(fontsize=22)
        plt.xlabel("x", fontsize=22)
        plt.ylabel("correlation", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(save_dir + folder_name + "predictive_correlation_process")
        plt.close(fig)

        # Plot standard deviation process
        for m in range(M):
            fig= plt.figure()
            plt.plot(x, true_stds[:, m], label="true")
            plt.plot(x, init_stds[:, m], "--", label="initialized")
            plt.plot(x, est_stds[:, m], "--", label="inferred")
            plt.legend(fontsize=22)
            plt.xlabel("x", fontsize=22)
            plt.ylabel("standard deviation", fontsize=22)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(save_dir + folder_name + "predictive_std_D{}".format(m + 1))
            plt.close(fig)
