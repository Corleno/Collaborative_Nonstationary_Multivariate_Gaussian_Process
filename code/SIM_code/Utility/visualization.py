'''
Visualization for posterior analysis
'''

import numpy as np
import matplotlib.pyplot as plt


def samples2quantiles(pos_sample, percentiles = [2.5, 50., 97.5]):
    """
    convert samples to quantiles
    :param pos_sample: 3d array with dim N_grid, N_samples, M
    :param percentiles: list
    :return: 3d array with dim N_percentiles, N_grids, M
    """
    res = np.percentile(pos_sample, q=percentiles, axis=1)
    return res


def Plot_posterior(x, Y, grids, pos_quantile, save_dir=None, folder_name=None, subfolder_name=None, attributes=None, type="MAP"):
    """
    Plot the pointwise posterior predictive process
    :param x: 1d array with length N
    :param Y: 2d array with dim N, M
    :param grids: 1d array with dim N_grid
    :param pos_quantile: 3d array with dim N_percentile, N_grid, M
    :return:
    """
    _, M = Y.shape
    for m in range(M):
        fig = plt.figure()
        plt.scatter(x, Y[:, m])
        plt.plot(grids, pos_quantile[1, :, m], color='b')
        plt.plot(grids, pos_quantile[[0,2], :, m].T, color="r", linestyle="dashed")
        plt.xlabel("x", fontsize=22)
        plt.ylabel("y{}".format(m+1), rotation=0, fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if subfolder_name is None:
            plt.savefig(save_dir + folder_name + "Pos_pred_{}_{}.png".format(attributes[m], type))
        else:
            plt.savefig(save_dir + folder_name + subfolder_name + "Pos_pred_{}_{}.png".format(attributes[m], type))
        plt.close(fig)


def Plot_posterior_hadamard(x, indx, y, grids, pos_quantile, save_dir=None, folder_name=None, subfolder_name=None, attributes=None):
    cat_list = np.unique(indx)
    for m in cat_list:
        fig = plt.figure()
        plt.scatter(x[indx == m], y[indx == m])
        plt.plot(grids, pos_quantile[1, :, int(m)], color='b')
        plt.plot(grids, pos_quantile[[0, 2], :, int(m)].T, color="r")
        plt.xlabel("x", fontsize=22)
        plt.ylabel("y{}".format(m + 1), rotation=0, fontsize=22)
        plt.legend(fontsize=16, loc='upper right')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if subfolder_name is None:
            plt.savefig(save_dir + folder_name + "Pos_pred_{}.png".format(attributes[m]))
        else:
            plt.savefig(save_dir + folder_name + subfolder_name + "Pos_pred_{}.png".format(attributes[m]))
        plt.close(fig)


def Plot_posterior_trainandtest(x, Y, grids, pos_quantile, x_test, Y_test, Y_pred, save_dir=None, folder_name=None,
                                subfolder_name=None, with_obs=True, attributes=None, type="MAP"):
    """
    Plot the pointwise posterior predictive process
    :param x: 1d array with length N
    :param Y: 2d array with dim N, M
    :param grids: 1d array with dim N_grid
    :param pos_quantile: 3d array with dim N_percentile, N_grid, M
    :return: None
    """
    _, M = Y.shape
    if attributes is None:
        attributes = np.arange(M) + 1
    for m in range(M):
        fig = plt.figure()
        plt.scatter(x, Y[:, m], label="training data")
        if with_obs:
            plt.scatter(x_test, Y_test[:, m], label="ground truth data")
            plt.scatter(x_test, Y_pred[:, m], label="predicted data")
        plt.plot(grids, pos_quantile[1, :, m], color='orange', label="predictive mean")
        plt.fill_between(grids, pos_quantile[0, :, m], pos_quantile[2, :, m], color="r", alpha =0.2, label="preditive 95% confidence interval")
        s = np.max(Y[:, m]) - np.min(Y[:, m])
        plt.ylim(np.min(Y[:, m]) - 0.15*s, np.max(Y[:, m]) + 0.8*s)
        plt.xlabel("time (hour)", fontsize=22)
        plt.ylabel("{}".format(attributes[m]), fontsize=22)
        plt.legend(fontsize=16, loc='upper right')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if subfolder_name is None:
            plt.savefig(save_dir + folder_name + "Pos_pred_{}_{}.png".format(attributes[m], type))
        else:
            plt.savefig(save_dir + folder_name + subfolder_name + "Pos_pred_{}_{}.png".format(attributes[m], type))
        # plt.close(fig)


def Plot_posterior_trainandtest_non(x_train_list, y_train_list, grids, y_grids_quantile_list, x_test_list, y_test_list, pred_test_list, save_dir=None, folder_name=None, attributes=None, type="MAP"):
    """
    Plot the pointwise posterior predictive process for nonstationary model
    :return: None
    """
    dim_index = 0
    for x_train, y_train, y_grids_quantile, x_test, y_test, y_pred in zip(x_train_list, y_train_list, y_grids_quantile_list, x_test_list, y_test_list, pred_test_list):
        fig = plt.figure()
        plt.scatter(x_train, y_train, label="training data")
        plt.scatter(x_test, y_test, label="ground truth data")
        plt.scatter(x_test, y_pred, label="predicted data")
        plt.plot(grids, y_grids_quantile[1, :], color='b')
        plt.plot(grids, y_grids_quantile[[0, 2], :].T, color="r")
        s = np.max(y_train) - np.min(y_train)
        plt.ylim(np.min(y_train) - 2 * s, np.max(y_train) + 2 * s)
        plt.xlabel("x", fontsize=15)
        plt.ylabel("y{}".format(dim_index), rotation=0, fontsize=15)
        plt.legend(fontsize=12, loc=1)
        plt.savefig(save_dir + folder_name + "Pos_pred_{}_{}.png".format(attributes[dim_index], type))
        plt.close(fig)
        dim_index = dim_index + 1


if __name__ == "__main__":
    pass
