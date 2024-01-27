import pickle
import numpy as np
import matplotlib.pyplot as plt


def orig2adj(x, Y):
    """
    preprocess for mimic data: detrend and normalized for each feature.
    :param x: 1d array with length N
    :param Y: 2d array with dim N, N_feature
    :return: adjY, trend and scale
    """
    trend = np.mean(Y, axis=0)
    adjY = Y - trend
    scale = np.std(adjY, axis=0)
    adjY = adjY/scale
    return adjY, trend, scale


def adj2orig(adjY, trend, scale):
    """
    convert adjusted time series back to original time series
    :param adj_Y: 2d array with dim N, N_feature
    :param trend: 1d array with length N_feature
    :param scale: 1d array with length N_feature
    :return:
    """
    adjY = adjY * scale
    Y = adjY + trend
    return Y


def orig2adj_non(y_list):
    """
    preprocess for mimic data: detrend and normalized for each feature.
    :param y_list: list
    :return: adj_y_list, trend_list and scale_list
    """
    adj_y_list = []
    trend_list = []
    scale_list = []
    for y in y_list:
        trend = np.mean(y)
        adj_y = y - trend
        scale = np.std(adj_y)
        adj_y = adj_y/scale
        adj_y_list.append(adj_y)
        trend_list.append(trend)
        scale_list.append(scale)
    return adj_y_list, trend_list, scale_list


def adj2orig_non(adj_y_list, trend_list, scale_list):
    """
    convert adjusted time series back to original time series
    :param adj_y_list: list
    :param trend_list: list
    :param scale_list: list
    :return: y_list
    """
    y_list = []
    for adj_y, trend, scale in zip(adj_y_list, trend_list, scale_list):
        y = adj_y * scale
        y_list.append(y + trend)
    return y_list


if __name__ == "__main__":
    # with open("../data/mimic_1p.pickle", "rb") as res:
    #     x, Y = pickle.load(res)
    # fig = plt.figure()
    # plt.plot(x, Y)
    # plt.savefig("mimic/raw_1p.png")
    # plt.show()
    #
    # # preprocess
    # adjY, trend, scale = orig2adj(x, Y)
    # fig = plt.figure()
    # plt.plot(x, adjY)
    # plt.savefig("mimic/adj_1p.png")
    # plt.show()
    #
    # backY = adj2orig(adjY, trend, scale)
    # assert (Y == backY).all(), "conversion failed!"

    with open("../data/patient_record.pickle", "rb") as res:
        x_list, y_list = pickle.load(res)
    print(y_list)
    indx_list = [i * np.ones_like(x) for i, x in enumerate(x_list)]
    print(indx_list)
    adj_y_list, trend_list, scale_list = orig2adj_non(y_list)
    print(adj_y_list)
    y_list = adj2orig_non(adj_y_list, trend_list, scale_list)
    print(y_list)