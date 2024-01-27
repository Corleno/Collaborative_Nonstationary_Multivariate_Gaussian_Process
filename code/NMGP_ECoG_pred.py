import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import time
import math
import torch
from utils import *
TensorType = torch.DoubleTensor
from nmgp_dsvi import *
import statsmodels.api as sm
from sklearn import preprocessing


def scale_datasets(Y_list):
    M = len(Y_list)
    Y_scaled_list = list()
    for i in range(M):
        ts = Y_list[i]
        Y_scaled_list.append(ts/(np.max(ts) - np.min(ts)))
    return Y_scaled_list


def create_datasets(X_list, Y_list, test_channel_index=0, channel_indexes=None, train_rate=0.8, random=True):
    np.random.seed(22)
    N = Y_list[0].shape[0]
    M = len(Y_list)
    if random:
        train_index = np.random.choice(N, size=int(train_rate * N), replace=False)
    else:
        train_index = np.arange(int(train_rate * N))
    test_index = np.array(list(set(np.arange(N)) - set(train_index)))

    Xtrain_list = list()
    Ytrain_list = list()
    Xtest_list = list()
    Ytest_list = list()
    if channel_indexes is None:
        channel_indexes = np.arange(M)
    for i in channel_indexes:
        if i == channel_indexes[test_channel_index]:
            Xtrain_list.append(X_list[i][train_index])
            Ytrain_list.append(Y_list[i][train_index])
            Xtest_list.append(X_list[i][test_index])
            Ytest_list.append(Y_list[i][test_index])
        else:
            Xtrain_list.append(X_list[i])
            Ytrain_list.append(Y_list[i])
            Xtest_list.append(np.array([]))
            Ytest_list.append(np.array([]))
    return Xtrain_list, Ytrain_list, Xtest_list, Ytest_list


def plot_resps(times, resps, time_trials):
    fig = plt.figure()
    resp_mean = np.mean(resps, axis=1)
    resp_std = np.std(resps, axis=1)
    resp_upper, resp_lower = resp_mean + resp_std, resp_mean - resp_std
    plt.plot(times, resp_mean)
    plt.plot(times, np.stack([resp_lower, resp_upper], axis=1), linestyle='--', color='r')
    if time_trials is not None:
        for time_trail in time_trials:
            plt.vlines(time_trail[0], ymin=-2, ymax=10)
            plt.vlines(time_trail[1], ymin=-2, ymax=10)
    plt.show()
    plt.close(fig)


def plot_resps_fb(times, resps, channel_indexes, data_directory=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # import pdb; pdb.set_trace()
    res = sm.graphics.fboxplot(resps.T, wfactor=1000., labels=channel_indexes, ax=ax)
    ax.set_xlabel("Time (second)", fontsize=18)
    ax.set_ylabel("Z-score", fontsize=18)
    ax.set_xticks(np.arange(times.shape[0], step=100))
    ax.set_xticklabels(np.arange(times.shape[0], step=100)/400.)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    fig.savefig(data_directory + "/functional_boxplot.png")
    plt.show()


def extract_trails(time_trails, time_range):
    res_list = []
    for time_trail in time_trails:
        if ((time_trail[0] > time_range[0]) and (time_trail[0] < time_range[1])) or ((time_trail[1] > time_range[0]) and (time_trail[1] < time_range[1])):
            res_list.append([np.max([time_trail[0], time_range[0]]), np.min([time_trail[1], time_range[1]])])
    return np.stack(res_list)


def time2timestamp(x, rate):
    return np.round(x * rate).astype(int)


def load_ECoG(data_file, n_channel=None, channel_indexes=None, do_plot_raw_data=True, time_start=14., time_stop=16., rate=400.):
    data = "ECoG"
    if not os.path.exists("../res/{}".format(data)):
        os.mkdir("../res/{}".format(data))
    # Default setting
    times_test = np.arange(160) / rate
    # Upload Data
    time_trials = None
    if data_file == "data_R32_B7.pickle":
        data_loc = "../data/ECoG/R32_B7_Hilb_54bands_ECoG_high_gamma.pickle"
        with open(data_loc, "rb") as res:
            times, band_resps, time_trials = pickle.load(res)
    if data_file == "data_R32_B8.pickle":
        data_loc = "../data/ECoG/R32_B8_Hilb_54bands_ECoG_high_gamma.pickle"
        with open(data_loc, "rb") as res:
            times, band_resps = pickle.load(res)
    # take subsamples
    time_interval_name = "{}s_{}s".format(int(time_start), int(time_stop))
    time_interval = np.linspace(time_start, time_stop, num=int((time_stop - time_start) * (rate)), endpoint=False)
    timestamps_interval = time2timestamp(time_interval, rate)
    if data_file == "data_R32_B7.pickle":
        time_range = np.array([time_interval[0], time_interval[-1]])
        time_trials = extract_trails(time_trials, time_range)
    times = times[timestamps_interval]
    band_resps = band_resps[timestamps_interval]
    # import pdb; pdb.set_trace()

    N, M = band_resps.shape
    X_list = list()
    Y_list = list()
    for i in range(M):
        X_list.append(np.arange(N))
        Y_list.append(preprocessing.scale(band_resps[:, i]))
    # sample channels
    X_list, Y_list, Xt_list, Yt_list = create_datasets(X_list, Y_list, channel_indexes=channel_indexes)
    if n_channel is None:
        n_channel = len(X_list)
    n_dims = n_channel
    pre_name = "pred_data{}".format(n_channel)
    # convert data format
    X_train_origin_list = [x[:, None] for x in X_list]
    X_test_origin_list = [x[:, None] for x in Xt_list]
    Y_train_origin_list = [y[:, None] for y in Y_list]
    Y_test_origin_list = [y[:, None] for y in Yt_list]

    X_train_list = X_train_origin_list[:n_dims]
    X_test_list = X_test_origin_list[:n_dims]
    Y_train_list = Y_train_origin_list[:n_dims]
    Y_test_list = Y_test_origin_list[:n_dims]

    X_train_vec = np.concatenate(X_train_list)
    X_test_vec = np.concatenate(X_test_list)
    Y_train_vec = np.concatenate(Y_train_list)
    Y_test_vec = np.concatenate(Y_test_list)

    train_index = np.concatenate([np.ones_like(Y_train_list[i]) * i for i in range(n_dims)]).astype(int)
    test_index = np.concatenate([np.ones_like(Y_test_list[i]) * i for i in range(n_dims)]).astype(int)
    t_max = np.max(np.concatenate(X_list))

    if do_plot_raw_data:
        n_illustration = 10
        gap=10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(n_illustration):
            ax.plot(X_train_list[i], Y_train_list[i] - gap * i, 'k', mew=1.5)
            ax.plot(X_test_list[i], Y_test_list[i] - gap * i, 'r', mew=1.5)
        ax.legend(loc="upper left")
        plt.show()

        standard_band_resps = preprocessing.scale(band_resps, axis=0)
        standard_band_resps = standard_band_resps[:, channel_indexes]
        plot_resps(times, standard_band_resps, time_trials)
        plot_resps_fb(times, standard_band_resps, channel_indexes, data_directory="../data/ECoG")

    return X_train_list, X_test_list, Y_train_list, Y_test_list, X_train_vec, X_test_vec, Y_train_vec, Y_test_vec, \
           train_index, test_index, t_max, rate, times_test, data, pre_name, time_interval_name


#%% Functions
def CNMGP(directory, X_train_list, Y_train_list, M, hyperpars={}, initpars={}, batchsize=0, lr=0.01, itnum=2000, do_inference=True, do_earlystop=False, do_stop_criterion=False,
          verbose=False, PATH="model.pt", continuous_training=False, show_ELBO=True, save_model=False):
    z = np.linspace(0, t_max, num=M)
    dim_outputs = len(X_train_list)

    if batchsize == 0:
        batch_size = X_train_vec.shape[0]
    else:
        batch_size = batchsize

    if do_inference:
        NMGP_model, loss_list, time_list = inference(X_train_list, Y_train_list, z, batch_size, dim_outputs,
            lr=lr, itnum=itnum, hyperpars=hyperpars, do_stop_criterion=do_stop_criterion,
            verbose=verbose, PATH=PATH, continuous_training=continuous_training, show_ELBO=show_ELBO,
            save_model=save_model, **initpars)
        fig = plt.figure()
        plt.plot(np.array(time_list), np.array(loss_list))
        plt.title('loss_trace')
        plt.show()
        plt.close(fig)

        # save results
        with open(directory, "wb") as res:
            pickle.dump([NMGP_model, loss_list, time_list], res)
    else:
        with open(directory, "rb") as res:
            NMGP_model, loss_list, time_list = pickle.load(res)
        return NMGP_model, loss_list, time_list


def plot_sample(times, samples):
    mean_curve = np.mean(samples, axis=0)
    std_curve = np.std(samples, axis=0)
    # plt.plot(times, mean_curve, color="b")
    plt.fill_between(times, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    # plt.plot(times, mean_curve - std_curve, linestyle="--", color='r')
    # plt.plot(times, mean_curve + std_curve, linestyle="--", color='r')


def plot_channels(channel, timestamps, true_ts, est_ts, num_tr=800):
    fig = plt.figure()
    plt.plot(true_ts[timestamps + (channel-1)*num_tr + 640], label="true")
    plt.plot(est_ts[timestamps + (channel-1)*num_tr + 640], label="estimate")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


#%% main
if __name__ == "__main__":
    test_channel_index = 0

    #%% load ECoG Data
    with open("../data/ECoG/78_channel_indexes.pickle", "rb") as res:
    # with open("../data/ECoG/102_channel_indexes.pickle", "rb") as res:
    # with open("../data/ECoG/48_channel_indexes.pickle", "rb") as res:
        channel_indexes = pickle.load(res)
    n_channels = channel_indexes.shape[0]
    data_file = "data_R32_B7.pickle"
    # data_file = "data_R32_B8.pickle"
    time_start = 14.
    time_stop = 16.
    rate = 400.
    X_train_list, X_test_list, Y_train_list, Y_test_list, X_train_vec, X_test_vec, Y_train_vec, Y_test_vec, \
    train_index, test_index, t_max, rate, times_test, data, pre_name, time_interval_name = load_ECoG(data_file, time_start=time_start,
        time_stop=time_stop, rate=400., channel_indexes=channel_indexes)
    # time_interval_name = "14s_19s_50"

    print("Inference starts.")
    # import pdb; pdb.set_trace()
    do_NMGP25 = False
    do_NMGP50 = True
    do_NMGP100 = True
    do_NMGP200 = True


    #%% #NMGP25
    if do_NMGP25:
        M = 25
        batchsize = 1000
        lr = 0.01
        itnum = 100
        directory = "../res/{}/{}/prediction_res_M{}_B{}.pickle".format(data, pre_name, M, batchsize)

        # CNMGP(directory, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True)
        NMGP_model, loss_list50, rmse_test_list50, time_list50 = CNMGP(directory, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
        fig = plt.figure()
        plt.plot(np.array(time_list50), np.array(loss_list50))
        plt.title('loss_trace')
        plt.show()
        plt.close(fig)
        fig = plt.figure()
        plt.plot(np.array(time_list50), np.array(rmse_test_list50))
        plt.title('rmse_test_trace')
        plt.show()
        plt.close(fig)

        print("training time per batch: {}s".format(time_list50[-1]/itnum/math.ceil(X_train_vec.shape[0]/batchsize)))
        print("training time per epoch: {}s".format(time_list50[-1]/itnum))
        ts = time.time()
        est_Y_test = predict_Y(NMGP_model, X_test_list)
        rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
        print("testing time:{}s".format(time.time() - ts))
        print("rmse: {}".format(rmse_test))

        fig = plt.figure()
        plt.plot(Y_test_vec.reshape(-1), label="true")
        plt.plot(est_Y_test, label="estimate")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        est_Y_train = predict_Y(NMGP_model, X_train_list)
        fig = plt.figure()
        plt.plot(Y_train_vec.reshape(-1)[:640], label="true")
        plt.plot(est_Y_train[:640], label="estimate")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        sampled_Y_test = sample_Y(NMGP_model, X_test_list)[0]
        fig = plt.figure()
        plt.plot(times_test, Y_test_vec.reshape(-1), label="true")
        plt.plot(times_test, est_Y_test, label="estimate")
        plot_sample(times_test, sampled_Y_test)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        import pdb; pdb.set_trace()


    #%% #NMGP50
    if do_NMGP50:
        M = 50
        # batchsize = 2048
        batchsize=512
        lr = 0.005
        itnum = 20
        directory = "../res/{}/{}/{}".format(data, pre_name, time_interval_name)
        print("directory: {}".format(directory))
        directory_inference = directory + "/inference_res_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                           test_channel_index)
        directory_prediction = directory + "/predictive_res_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                             test_channel_index)
        directory_prediction_summary = directory + "/predictive_res_s_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                             test_channel_index)
        hyperpars = {"length_scales_L0_log": 10, "length_scales_L1_log": 10, "length_scales_tildeell_log": 2,
                     "sigma2_err_log": -5}
        # hyperpars = {"length_scales_L0_log": 10, "length_scales_L1_log": 10, "length_scales_tildeell_log": 5,
        #              "sigma2_err_log": -5}
        initpars = {"mu_v": 1 * np.ones(M)}
        ## do inference
        # CNMGP(directory_inference, X_train_list, Y_train_list, M, hyperpars=hyperpars, initpars=initpars,
        #       batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True, do_earlystop=False, do_stop_criterion=False,
        #       PATH=directory + "/model50.pt", continuous_training=False, verbose=True, show_ELBO=False, save_model=True)
        print("inference completed!")
        ### load inference results
        NMGP_model, loss_list50, time_list50 = CNMGP(directory_inference, X_train_list, Y_train_list, M,
                                                batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
        fig = plt.figure()
        plt.plot(np.array(time_list50), np.array(loss_list50))
        plt.title('loss_trace')
        plt.show()
        plt.close(fig)

        # # reconstruction
        number_train = int((time_stop - time_start)*rate)
        grids_train = np.arange(number_train)
        # ### do reconstruction
        # sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train = sample_FY(NMGP_model, grids_train)
        # # ### save reconstruction results
        # # with open(directory_prediction, "wb") as res:
        # #     pickle.dump([sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train], res)
        # ### load reconstruction results
        # # with open(directory_prediction, "rb") as res:
        # #     sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train = pickle.load(res)
        # est_Y_train = np.mean(sampled_Ys_train, axis=0)
        # quantile_Y_train = np.quantile(sampled_Ys_train, q=(0.025, 0.975), axis=0)
        # gridlls_mean = np.mean(sampled_tilde_ells_train, axis=0)
        # gridlls_quantiles = np.percentile(sampled_tilde_ells_train, q=(2.5, 97.5), axis=0)
        # est_corrs_train = np.mean(sampled_corrs_train, axis=0)
        # ## save reconstruction summarization results
        # with open(directory_prediction_summary, "wb") as res:
        #     pickle.dump([est_Y_train, quantile_Y_train, gridlls_mean, gridlls_quantiles, est_corrs_train], res)
        ### load reconstruction summarization results
        with open(directory_prediction_summary, "rb") as res:
            est_Y_train, quantile_Y_train, gridlls_mean, gridlls_quantiles, est_corrs_train = pickle.load(res)

        print("training time per epoch: {}s".format(time_list50[-1] / itnum ))
        ts = time.time()
        est_Y_test = predict_Y(NMGP_model, X_test_list)
        rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
        print("testing time:{}s".format(time.time() - ts))
        print("rmse: {}".format(rmse_test))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Output 1')
        ax1.plot(X_test_list[0], est_Y_test, 'rx', mew=1.5, label='Prediction set')
        ax1.plot(X_test_list[0], np.vstack(Y_test_list), 'kx', mew=1.5, label='test set')
        ax1.legend()
        plt.show()

    #%% #NMGP100
    if do_NMGP100:
        M = 100
        # batchsize = 2048
        batchsize = 512
        lr = 0.005
        itnum = 40
        directory = "../res/{}/{}/{}".format(data, pre_name, time_interval_name)
        print("directory: {}".format(directory))
        directory_inference = directory + "/inference_res_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                           test_channel_index)
        directory_prediction = directory + "/predictive_res_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                             test_channel_index)
        directory_prediction_summary = directory + "/predictive_res_s_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                                       test_channel_index)
        hyperpars = {"length_scales_L0_log": 10, "length_scales_L1_log": 10, "length_scales_tildeell_log": 2,
                     "sigma2_err_log": -5}
        # hyperpars = {"length_scales_L0_log": 10, "length_scales_L1_log": 10, "length_scales_tildeell_log": 5,
        #              "sigma2_err_log": -5}
        initpars = {"mu_v": 1 * np.ones(M)}
        ## do inferences
        # CNMGP(directory_inference, X_train_list, Y_train_list, M, hyperpars=hyperpars, initpars=initpars,
        #       batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True, do_earlystop=False, do_stop_criterion=False,
        #       PATH=directory + "/model100.pt", continuous_training=True, verbose=True, show_ELBO=False, save_model=True)
        print("inference completed!")
        ### load inference results
        NMGP_model, loss_list100, time_list100 = CNMGP(directory_inference, X_train_list, Y_train_list, M,
                                                     batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
        fig = plt.figure()
        plt.plot(np.array(time_list100), np.array(loss_list100))
        plt.title('loss_trace')
        plt.show()
        plt.close(fig)

        # # reconstruction
        number_train = int((time_stop - time_start) * rate)
        grids_train = np.arange(number_train)
        # ### do reconstruction
        # sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train = sample_FY(NMGP_model, grids_train)
        # # ### save reconstruction results
        # # with open(directory_prediction, "wb") as res:
        # #     pickle.dump([sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train], res)
        # ### load reconstruction results
        # # with open(directory_prediction, "rb") as res:
        # #     sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train = pickle.load(res)
        # est_Y_train = np.mean(sampled_Ys_train, axis=0)
        # quantile_Y_train = np.quantile(sampled_Ys_train, q=(0.025, 0.975), axis=0)
        # gridlls_mean = np.mean(sampled_tilde_ells_train, axis=0)
        # gridlls_quantiles = np.percentile(sampled_tilde_ells_train, q=(2.5, 97.5), axis=0)
        # est_corrs_train = np.mean(sampled_corrs_train, axis=0)
        # ## save reconstruction summarization results
        # with open(directory_prediction_summary, "wb") as res:
        #     pickle.dump([est_Y_train, quantile_Y_train, gridlls_mean, gridlls_quantiles, est_corrs_train], res)
        ### load reconstruction summarization results
        with open(directory_prediction_summary, "rb") as res:
            est_Y_train, quantile_Y_train, gridlls_mean, gridlls_quantiles, est_corrs_train = pickle.load(res)

        print("training time per epoch: {}s".format(time_list100[-1] / itnum ))
        ts = time.time()
        est_Y_test = predict_Y(NMGP_model, X_test_list)
        rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
        print("testing time:{}s".format(time.time() - ts))
        print("rmse: {}".format(rmse_test))

        # import pdb; pdb.set_trace()

    #%% #NMGP200
    if do_NMGP200:
        M = 200
        # batchsize = 2048
        batchsize = 512
        lr = 0.005
        itnum = 20
        directory = "../res/{}/{}/{}".format(data, pre_name, time_interval_name)
        print("directory: {}".format(directory))
        directory_inference = directory + "/inference_res_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                           test_channel_index)
        directory_prediction = directory + "/predictive_res_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                             test_channel_index)
        directory_prediction_summary = directory + "/predictive_res_s_M{}_B{}_channel{}.pickle".format(M, batchsize,
                                                                                                       test_channel_index)
        hyperpars = {"length_scales_L0_log": 10, "length_scales_L1_log": 10, "length_scales_tildeell_log": 2,
                     "sigma2_err_log": -5}
        # hyperpars = {"length_scales_L0_log": 10, "length_scales_L1_log": 10, "length_scales_tildeell_log": 5,
        #              "sigma2_err_log": -5}
        initpars = {"mu_v": 1 * np.ones(M)}
        ## do inference
        # CNMGP(directory_inference, X_train_list, Y_train_list, M, hyperpars=hyperpars, initpars=initpars,
        #       batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True, do_earlystop=False, do_stop_criterion=False,
        #       PATH=directory + "/model200.pt", continuous_training=True, verbose=True, show_ELBO=False, save_model=True)
        print("inference completed!")
        ### load inference results
        NMGP_model, loss_list200, time_list200 = CNMGP(directory_inference, X_train_list, Y_train_list, M,
                                                     batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
        fig = plt.figure()
        plt.plot(np.array(time_list200), np.array(loss_list200))
        plt.title('loss_trace')
        plt.show()
        plt.close(fig)

        # # reconstruction
        number_train = int((time_stop - time_start) * rate)
        grids_train = np.arange(number_train)
        ### do reconstruction
        # sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train = sample_FY(NMGP_model, grids_train)
        # # ### save reconstruction results
        # # with open(directory_prediction, "wb") as res:
        # #     pickle.dump([sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train], res)
        # ### load reconstruction results
        # # with open(directory_prediction, "rb") as res:
        # #     sampled_tilde_ells_train, sampled_Ys_train, sampled_corrs_train = pickle.load(res)
        # est_Y_train = np.mean(sampled_Ys_train, axis=0)
        # quantile_Y_train = np.quantile(sampled_Ys_train, q=(0.025, 0.975), axis=0)
        # gridlls_mean = np.mean(sampled_tilde_ells_train, axis=0)
        # gridlls_quantiles = np.percentile(sampled_tilde_ells_train, q=(2.5, 97.5), axis=0)
        # est_corrs_train = np.mean(sampled_corrs_train, axis=0)
        # ## save reconstruction summarization results
        # with open(directory_prediction_summary, "wb") as res:
        #     pickle.dump([est_Y_train, quantile_Y_train, gridlls_mean, gridlls_quantiles, est_corrs_train], res)
        ### load reconstruction summarization results
        with open(directory_prediction_summary, "rb") as res:
            est_Y_train, quantile_Y_train, gridlls_mean, gridlls_quantiles, est_corrs_train = pickle.load(res)

        print("training time per epoch: {}s".format(time_list200[-1] / itnum ))
        ts = time.time()
        est_Y_test = predict_Y(NMGP_model, X_test_list)
        rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
        print("testing time:{}s".format(time.time() - ts))
        print("rmse: {}".format(rmse_test))

        # import pdb; pdb.set_trace()

    #%% Reconstruction analysis

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Output 1')
    ax1.plot(X_test_list[0], est_Y_test, 'rx', mew=1.5, label='Prediction set')
    ax1.plot(X_test_list[0], np.vstack(Y_test_list), 'kx', mew=1.5, label='test set')
    ax1.legend()
    plt.show()

    import pdb; pdb.set_trace()

    fig = plt.figure()
    plt.plot(grids_train, gridlls_mean[:number_train], color='b')
    plt.plot(grids_train, gridlls_quantiles.T[:number_train], color='r', linestyle="dashed")
    plt.plot(NMGP_model.Z.cpu().numpy().reshape(-1), NMGP_model.mu_v.cpu().detach().numpy(), 'o')
    plt.xlabel("x", fontsize=22)
    plt.ylabel("y", rotation=0, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.legend()
    fig.savefig(directory + "/gridlls.png")
    plt.show()

    for m in range(n_channels):
        fig = plt.figure()
        plt.scatter(X_train_list[m], Y_train_list[m], label="train data")
        plt.plot(grids_train, est_Y_train[:, m], color='b')
        plt.plot(grids_train, quantile_Y_train[:, :, m].T, color="r", linestyle="dashed")
        plt.xlabel("Time (second)", fontsize=15)
        plt.ylabel("Z-score", fontsize=15)
        # plt.xticks(ticks=np.linspace(0,2000,num=6), labels=np.linspace(0,5,num=6) ,fontsize=15)
        plt.xticks(ticks=np.linspace(0, 800, num=6), labels=np.round(np.linspace(0, 2, num=6), 2), fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-2, 8)
        plt.tight_layout()
        plt.legend(loc="upper right")
        fig.savefig(directory + "/obs/obs_{}.png".format(m))
        plt.show()

    # plot_corrs(grids_train, sampled_corrs_train, dim1=0, dim2=1, directory=directory)
    import pdb; pdb.set_trace()
    #%% Post analysis
    print("Post analysis starts.")
    # import pdb; pdb.set_trace()
    # channel_indexes
    channel_dict = {index : i for i, index in enumerate(channel_indexes)}
    matrix_index = np.array([[53, 51, 49, 26, 18], [52, 50, 48, 24, 16], [74, 76, 78, 102, 110], [75, 77, 79, 100, 108], [91, 93, 95, 98, 106]])
    n_grid = 5


    do_direction_analysis = True
    do_distance_analysis = True

    if do_direction_analysis:
        ### direction analysis
        right_corrs = list()
        top_corrs = list()
        left_corrs = list()
        bottom_corrs = list()
        for i in range(n_grid):
            for j in range(n_grid):
                center = [i, j]
                center_index = channel_dict[matrix_index[center[0], center[1]]]
                if i-1 >= 0:
                    left = [i-1, j]
                    left_index = channel_dict[matrix_index[left[0], left[1]]]
                    left_corrs.append(est_corrs_train[:, center_index, left_index])
                if j+1 < 5:
                    top = [i, j+1]
                    top_index = channel_dict[matrix_index[top[0], top[1]]]
                    top_corrs.append(est_corrs_train[:, center_index, top_index])
                if i+1 < 5:
                    right = [i+1, j]
                    right_index = channel_dict[matrix_index[right[0], right[1]]]
                    right_corrs.append(est_corrs_train[:, center_index, right_index])
                if j-1 >= 0:
                    bottom = [i, j-1]
                    bottom_index = channel_dict[matrix_index[bottom[0], bottom[1]]]
                    bottom_corrs.append(est_corrs_train[:, center_index, bottom_index])
        left_corrs = np.stack(left_corrs)
        top_corrs = np.stack(top_corrs)
        right_corrs = np.stack(right_corrs)
        bottom_corrs = np.stack(bottom_corrs)
        fig, axes = plt.subplots(nrows=3, ncols=3)
        axes[0, 1].plot(top_corrs.mean(axis=0))
        axes[0, 1].set_ylim(0.3, 0.7)
        axes[1, 0].plot(left_corrs.mean(axis=0))
        axes[1, 0].set_ylim(0.3, 0.7)
        axes[1, 2].plot(right_corrs.mean(axis=0))
        axes[1, 2].set_ylim(0.3, 0.7)
        axes[2, 1].plot(bottom_corrs.mean(axis=0))
        axes[2, 1].set_ylim(0.3, 0.7)
        fig.savefig(directory + "/corrs.png")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    if do_distance_analysis:
        def analysis_dist(dist):
            dist_corrs = list()
            for i in range(n_grid - dist):
                for j in range(n_grid - dist):
                    center = [i, j]
                    center_index = channel_dict[matrix_index[center[0], center[1]]]
                    right = [i + dist, j]
                    right_index = channel_dict[matrix_index[right[0], right[1]]]
                    dist_corrs.append(est_corrs_train[:, center_index, right_index])
                    top = [i, j + dist]
                    top_index = channel_dict[matrix_index[top[0], top[1]]]
                    dist_corrs.append(est_corrs_train[:, center_index, top_index])
            dist_corrs = np.stack(dist_corrs)
            return dist_corrs.mean(axis=0)
        dists = np.array([1, 2, 3])
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for index, dist in enumerate(dists):
            axes[index].plot(analysis_dist(dist))
            axes[index].set_ylim(0, 0.6)
            # axes[index].set_xticks(ticks=np.linspace(0, 800, num=6))
            # axes[index].set_xticklabels(labels=np.round(np.linspace(0, 2, num=6),2))
            axes[index].set_xticks(ticks=np.linspace(0, 2000, num=6))
            axes[index].set_xticklabels(labels=np.round(np.linspace(0, 5, num=6),2))
            axes[index].set_xlabel("Time (second)", fontsize=15)
            axes[index].set_ylabel("Correlation coefficient", fontsize=15)

        plt.tight_layout()
        fig.savefig(directory + "/corrs_dist.png")
        plt.show()
        plt.close(fig)

    import pdb; pdb.set_trace()

