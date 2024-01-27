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

#%% PM25 subset

data = "HCP"
data_file = "data.pickle"
if not os.path.exists("../res/{}/{}".format(data, data_file)):
    os.mkdir("../res/{}/{}".format(data, data_file))

# Upload Data
with open("../data/{}/{}".format(data, data_file), "rb") as res:
    X_list, Y_list, Xt_list, Yt_list = pickle.load(res)

n_dims = len(X_list)

# convert data format
X_train_list = [x[:, None] for x in X_list]
X_test_list = [x[:, None] for x in Xt_list]
X_train_vec = np.concatenate(X_train_list)
X_test_vec = np.concatenate(X_test_list)
Y_train_list = [y[:, None] for y in Y_list]
Y_test_list = [y[:, None] for y in Yt_list]
Y_train_vec = np.concatenate(Y_train_list)
Y_test_vec = np.concatenate(Y_test_list)
train_index = np.concatenate([np.ones_like(Y_train_list[i])*i for i in range(n_dims)]).astype(int)
test_index = np.concatenate([np.ones_like(Y_test_list[i])*i for i in range(n_dims)]).astype(int)

t_max = np.max([np.max(np.concatenate(X_list)), np.max(np.concatenate(Xt_list))])

do_plot_raw_data = True

if do_plot_raw_data:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Output 1')
    ax1.plot(X_train_list[0], Y_train_list[0],'kx',mew=1.5,label='Train set')
    ax1.plot(X_test_list[0], Y_test_list[0],'rx',mew=1.5,label='Test set')
    ax1.legend()
    plt.show()

#%%
def VTVLCM(data, M, batchsize=0, lr=0.01, itnum=2000, do_inference=True, do_test=False):
    z = np.linspace(0, t_max, num=M)
    dim_outputs = len(X_list)

    if batchsize == 0:
        batch_size = X_train_vec.shape[0]
    else:
        batch_size = batchsize

    if do_inference:
        hyperpars = {"length_scales_L0_log": 5, "length_scales_L1_log": 5, "length_scales_tildeell_log": 5}
        initpars = {"mu_v": 1*np.ones(M)}
        if do_test:
            NMGP_model, loss_list, rmse_test_list, time_list = inference(X_train_list, Y_train_list, z, batch_size, dim_outputs, lr=lr,
                                itnum=itnum, hyperpars=hyperpars, verbose=True, show_ELBO=False,
                                X_test_list = X_test_list, Y_test_list = Y_test_list, **initpars)
        else:
            NMGP_model, loss_list, time_list = inference(X_train_list, Y_train_list, z, batch_size, dim_outputs, lr=lr,
                                itnum=itnum, hyperpars=hyperpars, verbose=True, show_ELBO=False, **initpars)

        fig = plt.figure()
        plt.plot(np.array(time_list), np.array(loss_list))
        plt.title('loss_trace')
        plt.show()
        plt.close(fig)

        if do_test:
            fig = plt.figure()
            plt.plot(np.array(time_list), np.array(rmse_test_list))
            plt.title('rmse_test_trace')
            plt.show()
            plt.close(fig)

        # import pdb; pdb.set_trace()
        # # prediction
        # grids = np.linspace(0, 1, number_grids)
        # predicted_Ys, sampled_Ls, sampled_Gs = predict_Y(NMGP_model, [grids, grids])
        # # reconstruction
        # predicted_Y_train, _, _ = predict_Y(NMGP_model, X_train_list)
        # est_Y_train = np.mean(predicted_Y_train, axis=0)
        # # testing
        # est_Y_test = predict_Y(NMGP_model, X_test_list)
        # print("training time per epoch: {}s".format(time_list[-1] / itnum))
        # ts = time.time()
        # est_Y_test = predict_Y(NMGP_model, X_test_list)
        # rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
        # print("testing time:{}s".format(time.time() - ts))
        # print("rmse: {}".format(rmse_test))

        # import pdb; pdb.set_trace()
        # save results
        if do_test:
            with open("../res/{}/prediction_res_M{}_B{}.pickle".format(data, M, batchsize), "wb") as res:
                pickle.dump([NMGP_model, loss_list, rmse_test_list, time_list], res)
        else:
            with open("../res/{}/prediction_res_M{}_B{}.pickle".format(data, M, batchsize), "wb") as res:
                pickle.dump([NMGP_model, loss_list, time_list], res)
    else:
        if do_test:
            with open("../res/{}/prediction_res_M{}_B{}.pickle".format(data, M ,batchsize), "rb") as res:
                NMGP_model, loss_list, rmse_test_list, time_list = pickle.load(res)
        else:
            with open("../res/{}/prediction_res_M{}_B{}.pickle".format(data, M ,batchsize), "rb") as res:
                NMGP_model, loss_list, time_list = pickle.load(res)
    if do_test:
        return NMGP_model, loss_list, rmse_test_list, time_list
    else:
        return NMGP_model, loss_list, time_list

# #%% NMGP M = 50
#
# M = 50
# batchsize = 1024
# lr = 0.01
# itnum = 10
#
# # VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True)
# NMGP_model, loss_list50, time_list50 = VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
# fig = plt.figure()
# plt.plot(np.array(time_list50), np.array(loss_list50))
# plt.title('loss_trace')
# plt.show()
# plt.close(fig)
#
# print("training time per epoch: {}s".format(time_list50[-1]/itnum))
# ts = time.time()
# est_Y_test = predict_Y(NMGP_model, X_test_list)
# rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
# print("testing time:{}s".format(time.time() - ts))
# print("rmse: {}".format(rmse_test))
#
# # import pdb; pdb.set_trace()
# #%% NMGP M = 100
#
#
# M = 100
# batchsize = 1024
# lr = 0.01
# itnum = 10
#
# # VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True)
# # NMGP_model, loss_list100, time_list100 = VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
# fig = plt.figure()
# plt.plot(np.array(time_list100), np.array(loss_list100))
# plt.title('loss_trace')
# plt.show()
# plt.close(fig)
#
# print("training time per epoch: {}s".format(time_list100[-1]/itnum))
# ts = time.time()
# est_Y_test = predict_Y(NMGP_model, X_test_list)
# rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
# print("testing time:{}s".format(time.time() - ts))
# print("rmse: {}".format(rmse_test))
#
# # import pdb; pdb.set_trace()
# #%% NMGP M = 200
#
# M = 200
# batchsize = 1024
# lr = 0.01
# itnum = 10
#
# # VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True)
# NMGP_model, loss_list200, time_list200 = VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False)
# fig = plt.figure()
# plt.plot(np.array(time_list200), np.array(loss_list200))
# plt.title('loss_trace')
# plt.show()
# plt.close(fig)
#
# print("training time per epoch: {}s".format(time_list200[-1]/itnum))
# ts = time.time()
# est_Y_test = predict_Y(NMGP_model, X_test_list)
# rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
# print("testing time:{}s".format(time.time() - ts))
# print("rmse: {}".format(rmse_test))
#
# # import pdb; pdb.set_trace()
#%%
# time50 = np.array(time_list50)
# time100 = np.array(time_list100)
# time200 = np.array(time_list200)
# rmse_test50 = np.array(rmse_test_list50)
# rmse_test100 = np.array(rmse_test_list100)
# rmse_test200 = np.array(rmse_test_list200)
#
# fig = plt.figure()
# plt.plot(time50[time50<175], rmse_test50[time50<175], label="IP=50")
# plt.plot(time100[time100<210], rmse_test100[time100<210], label="IP=100")
# plt.plot(time200, rmse_test200, label="IP=200")
# plt.xlabel("seconds")
# plt.ylabel("RMSE")
# plt.legend()
# plt.tight_layout()
# plt.savefig("train_trace_IP.png")
# plt.show()

#%%
M = 100
batchsize = 1000
lr = 0.01
itnum = 50

# VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True, do_test=True)
NMGP_model, loss_list_1000, rmse_test_list_1000, time_list_1000 = VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False, do_test=True)

#%%
M = 100
batchsize = 2000
lr = 0.01
itnum = 50

# VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True, do_test=True)
NMGP_model, loss_list_2000, rmse_test_list_2000, time_list_2000 = VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False, do_test=True)
#%%
M = 100
batchsize = 5000
lr = 0.01
itnum = 50

# VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=True, do_test=True)
NMGP_model, loss_list_5000, rmse_test_list_5000, time_list_5000 = VTVLCM(data, M, batchsize=batchsize, lr=lr, itnum=itnum, do_inference=False, do_test=True)
#%%

time_1000 = np.array(time_list_1000)
time_2000 = np.array(time_list_2000)
time_5000 = np.array(time_list_5000)
rmse_test_1000 = np.array(rmse_test_list_1000)
rmse_test_2000 = np.array(rmse_test_list_2000)
rmse_test_5000 = np.array(rmse_test_list_5000)
#
fig = plt.figure()
# plt.plot(time100[time100<210], rmse_test100[time100<210], label="BS=1000")
# plt.plot(time100, rmse_test100, label="BS=1000")
# plt.plot(time_2000, rmse_test_2000, label="BS=2000")
# plt.plot(time_5000, rmse_test_5000, label="BS=5000")
plt.plot(time_1000[time_1000<500], rmse_test_1000[time_1000<500], label="BS=1000")
plt.plot(time_2000[time_2000<350], rmse_test_2000[time_2000<350], label="BS=2000")
plt.plot(time_5000[time_5000<600], rmse_test_5000[time_5000<600], label="BS=5000")
plt.xlabel("Time (second)", fontsize=22)
plt.ylabel("RMSE", fontsize=22)
plt.xticks(ticks=np.linspace(0, 600, num=6), fontsize=22)
plt.yticks(ticks=np.linspace(0.98, 1.02, num=5), fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("train_trace_BS_HCP.png")
plt.show()
