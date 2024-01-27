import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import time

import os

import torch
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader

from utils import *

from pre_nmgp import pre_estimation_partial as pre_estimation

TensorType = torch.DoubleTensor
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev="cpu"
device = torch.device(dev)

def print_mem(itnum, bnum = 1):
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**20
    return 'iteration: {} batchnum {} memory use: {}MB'.format(itnum, bnum, memoryUse)


class Model(torch.nn.Module):
    """
    Customized Model class for all GP objects
    """

    def forward(self):
        return None

    # Three functions wrap:
    # _get_param_array, _set_parameters,_loss_and_grad

    def _get_param_array(self):
        param_array = []
        for param in self.parameters():
            if param.requires_grad:
                param_array.append(param.data.numpy().flatten())
        return np.concatenate(param_array)

    def _set_parameters(self, param_array):
        idx_current = 0
        for param in self.parameters():
            if param.requires_grad:
                idx_next = idx_current + np.prod(param.data.size())
                param_np = np.reshape(param_array[idx_current: idx_next], param.data.numpy().shape)
                idx_current = idx_next
                param.data = TensorType(param_np)

    def _loss_and_grad(self, param_array):
        # 1) set parametes of the model from a 1D param_array
        self._set_prarmeters(param_array)
        for name, param in self.named_parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        # 2) Compute loss function
        loss = self.compute_loss()
        loss.backward()
        # 3) Return loss value and gradients
        grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad.append(param.grad.data.numpy().flatten())
        print('loss: {}.'.format(loss.data.numpy()))
        grad = np.concatenate(grad)
        grad_isfinite = np.isfinite(grad)
        if np.all(grad_isfinite):
            return loss.data.numpy(), grad
        else:
            print("Waring: inf or nan in gradient: replacing with zeros.")
            return loss.data.numpy(), np.where(grad_isfinite, grad, 0.)


class trainData(Dataset):
    def __init__(self, X_data, Y_data, I):
        self.X_data = X_data
        self.Y_data = Y_data
        self.I = I

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index], self.I[index]

    def __len__(self):
        return len(self.X_data)


class NMGP(Model):

    def __init__(self, number_observations, dim_outputs, Z, minibatch_size=None, mu_v=None, mu_W=None, mu_U=None, sqrt_v=None, sqrt_W=None,
                 sqrt_U=None, seed=22):
        """
            mu_U: variational mean of coefficients (D x M)
            mu_W: variational mean of latent functions (D x D x M)
            mu_v: variational mean of length-scale function (M)
        """
        super(NMGP, self).__init__()
        self.Z = Z
        self.M = Z.shape[0]
        self.N = number_observations
        self.D = dim_outputs
        self.batch_size = minibatch_size

        torch.random.manual_seed(seed)
        # parameters
        sqrt_scale = 0.1
        if mu_W is None:
            self.mu_W = Parameter(0.1 * torch.randn(self.D, self.M).type(TensorType))
            # self.mu_W = Parameter(torch.zeros(self.D, self.M).type(TensorType))
        else:
            self.mu_W = Parameter(torch.from_numpy(mu_W).type(TensorType))
        if sqrt_W is None:
            self.sqrt_W = Parameter(sqrt_scale * torch.randn(self.D, self.M, self.M).type(TensorType))
        else:
            self.sqrt_W = Parameter(torch.from_numpy(sqrt_W).type(TensorType))
        if mu_v is None:
            self.mu_v = Parameter(-4 * torch.ones(self.M).type(TensorType))
        else:
            self.mu_v = Parameter(torch.from_numpy(mu_v).type(TensorType))
        if sqrt_v is None:
            self.sqrt_v = Parameter(sqrt_scale * torch.randn(self.M, self.M).type(TensorType))
        else:
            self.sqrt_v = Parameter(torch.from_numpy(sqrt_v).type(TensorType))
        if mu_U is None:
            self.mu_U = Parameter(0.1 * torch.randn(self.D, self.D, self.M).type(TensorType))
            # self.mu_U = Parameter(torch.zeros(self.D, self.D, self.M).type(TensorType))
        else:
            self.mu_U = Parameter(torch.from_numpy(mu_U).type(TensorType))
        if sqrt_U is None:
            self.sqrt_U = Parameter(sqrt_scale * torch.randn(self.D, self.D, self.M, self.M).type(TensorType))
        else:
            self.sqrt_U = Parameter(torch.from_numpy(sqrt_U).type(TensorType))
        # hyperparameters       
        self.sigma2_g = 1
        # GP hyper-parameters for the length scale function
        self.sigma2_tildeell_log = Parameter(torch.tensor(0.).type(TensorType))
        self.length_scales_tildeell_log = Parameter(torch.tensor(-4.).type(TensorType))
        # GP hyper-parameters for coefficient functions
        self.sigma2_L0_log = Parameter(torch.tensor(0.).type(TensorType))
        self.length_scales_L0_log = Parameter(torch.tensor(-4.).type(TensorType))
        self.sigma2_L1_log = Parameter(torch.tensor(0.).type(TensorType))
        self.length_scales_L1_log = Parameter(torch.tensor(-4.).type(TensorType))
        # hyper-parameters for error
        self.sigma2_err_log = Parameter(torch.tensor(-2.).type(TensorType))

    def forward(self, inputs_list, outputs_list, index=None, verbose=False):
        """
           index: selected dimension index
        """
        if verbose:
            t1 = time.time()
        if index is not None:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, index)])
        else:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, range(self.D))])
        output_index = torch.from_numpy(np.stack([np.arange(I.shape[0]), I], axis=1)).type(torch.long)
        inputs = torch.cat(inputs_list).view(-1, 1)
        outputs = torch.cat(outputs_list).view(-1, 1)

        # reparameterization of covariance matrices
        l_sqrt_W = mat2ltri(self.sqrt_W)
        self.Sigma_W = torch.matmul(l_sqrt_W, l_sqrt_W.permute(0, 2, 1))
        l_sqrt_v = mat2ltri(self.sqrt_v)
        self.Sigma_v = torch.matmul(l_sqrt_v, l_sqrt_v.permute(1, 0))
        l_sqrt_U = mat2ltri(self.sqrt_U)
        self.Sigma_U = torch.matmul(l_sqrt_U, l_sqrt_U.permute(0, 1, 3, 2))
        # reparameterization of hyper-parameters
        # GP hyper-parameters for the length scale function
        self.sigma2_tildeell = torch.exp(self.sigma2_tildeell_log)
        self.length_scales_tildeell = torch.exp(self.length_scales_tildeell_log)
        # GP hyper-parameters for coefficient functions
        self.sigma2_L0 = torch.exp(self.sigma2_L0_log)
        self.length_scales_L0 = torch.exp(self.length_scales_L0_log)
        self.sigma2_L1 = torch.exp(self.sigma2_L1_log)
        self.length_scales_L1 = torch.exp(self.length_scales_L1_log)
        # hyper-parameters for error
        self.sigma2_err = torch.exp(self.sigma2_err_log)
        if verbose:
            print("reparameterization costs {}s".format(time.time() - t1))

        batch_size = inputs.shape[0]

        if verbose:
            t1 = time.time()
        #### sample the parameters for the length-scale function at both training inputs and inducing inputs.
        # compute the covariance matrix for the length-scale function
        K_tildeell_11_diag = torch.ones(batch_size).type(TensorType).to(device)*self.sigma2_tildeell
        K_tildeell_12 = create_RBF(inputs, self.Z, scale2=self.sigma2_tildeell,
                                   length_scales=self.length_scales_tildeell)
        K_tildeell_22 = create_RBF(self.Z, scale2=self.sigma2_tildeell, length_scales=self.length_scales_tildeell)
        # # Approach 1
        # z_v = torch.randn(self.mu_v.size()).type(TensorType)
        # sampled_v = reparameterize(self.mu_v, self.Sigma_v, z_v, full_cov=True)
        # sampled_tilde_ell = CGP(K_tildeell_12, K_tildeell_22, K_tildeell_11, sampled_v)
        # # Approach 2
        # K_tildeell_11 = create_RBF(inputs, scale2=self.sigma2_tildeell, length_scales=self.length_scales_tildeell)
        # sampled_v_tilde_ell = JGP(K_tildeell_12, K_tildeell_22, K_tildeell_11, self.mu_v, self.Sigma_v)
        # sampled_tilde_ell = sampled_v_tilde_ell[:batch_size]
        # sampled_v = sampled_v_tilde_ell[batch_size:]
        # Approach 3
        sampled_v_tilde_ell = JGP_S(K_tildeell_11_diag, K_tildeell_12, K_tildeell_22, self.mu_v, self.Sigma_v)
        sampled_tilde_ell = sampled_v_tilde_ell[:batch_size]
        sampled_v = sampled_v_tilde_ell[batch_size:]
        sampled_ell_Z = torch.exp(sampled_v)
        sampled_ell_X = torch.exp(sampled_tilde_ell)

        #### sample the coefficients at training inputs
        # compute the covariance matrix for the coefficient functions.
        K_L0_11_diag = torch.ones(batch_size).type(TensorType).to(device) * self.sigma2_L0
        K_L0_12 = create_RBF(inputs, self.Z, scale2=self.sigma2_L0, length_scales=self.length_scales_L0)
        K_L0_22 = create_RBF(self.Z, scale2=self.sigma2_L0, length_scales=self.length_scales_L0)
        K_L1_11_diag = torch.ones(batch_size).type(TensorType).to(device) * self.sigma2_L1
        K_L1_12 = create_RBF(inputs, self.Z, scale2=self.sigma2_L1, length_scales=self.length_scales_L1)
        K_L1_22 = create_RBF(self.Z, scale2=self.sigma2_L1, length_scales=self.length_scales_L1)
        ## Approach 1
        sampled_L = torch.zeros(self.D, self.D, batch_size).type(TensorType).to(device)
        for i in range(self.D):
            for j in range(i+1):
                if i == j:
                    sampled_logLii = MGP_d(K_L1_12, K_L1_22, K_L1_11_diag, self.mu_U[i, j, :],
                                           self.Sigma_U[i, j, :, :])
                    sampled_L[i, j, :] = torch.exp(sampled_logLii)
                else:
                    sampled_Lij = MGP_d(K_L0_12, K_L0_22, K_L0_11_diag, self.mu_U[i, j, :],
                                        self.Sigma_U[i, j, :, :])
                    sampled_L[i, j, :] = sampled_Lij
        sampled_l = sampled_L.permute(2, 0, 1)[output_index[:, 0], output_index[:, 1]]
        ## Approach 2
        # sampled_l = torch.zeros(batch_size, self.D).type(TensorType).to(device)
        # for index, dim_index in enumerate(I):
        #     for j in range(dim_index):
        #         sampled_l[index, j] = MGP_d(K_L0_12[index].view(1, self.M), K_L0_22, K_L0_11_diag[index].view(1,1),
        #                                     self.mu_U[dim_index, j, :], self.Sigma_U[dim_index, j, :, :])
        #     sampled_l[index, dim_index] = torch.exp(MGP_d(K_L1_12[index].view(1, self.M), K_L1_22, K_L1_11_diag[index].view(1,1),
        #                                 self.mu_U[dim_index, j, :], self.Sigma_U[dim_index, j, :, :]))

        # compute summary statistics for the latent functions at training inputs
        K_G_11_diag = torch.ones(batch_size).type(TensorType).to(device) * self.sigma2_g
        K_G_12 = create_Gibbs(inputs, self.Z, sampled_ell_X, sampled_ell_Z, scale2=self.sigma2_g)
        K_G_22 = create_Gibbs(self.Z, self.Z, sampled_ell_Z, sampled_ell_Z, scale2=self.sigma2_g)
        # compute the marginal mean and variance for g with dimension D x batch_size
        mu_g, sigma2_g = MGP_mu_sigma2(K_G_12, K_G_22, K_G_11_diag, self.mu_W, self.Sigma_W)

        sampled_F = torch.sum(torch.mul(sampled_l, mu_g.t()), axis=1).view(-1, 1)
        # import pdb; pdb.set_trace()
        SELBO_R = Normal_logprob(sampled_F, torch.sqrt(self.sigma2_err), outputs)
        SELBO_R -= 0.5 / self.sigma2_err * (sampled_l ** 2 * sigma2_g.t()).sum()
        if verbose:
            print("computing the reconstruction term costs {}s".format(time.time() - t1))

        if verbose:
            t1 = time.time()
        # KL divergence
        # ts = time.time()
        KL_W = KL_Gaussian(self.mu_W, self.Sigma_W, torch.zeros(self.M).type(TensorType).to(device), K_G_22).sum()
        # print(time.time() - ts)
        # ts = time.time()
        KL_v = KL_Gaussian(self.mu_v, self.Sigma_v, torch.zeros(self.M).type(TensorType).to(device), K_tildeell_22)
        # print(time.time() - ts)
        # ts = time.time()
        KL_U = 0

        # for i in range(self.D):
        #     for j in range(i + 1):
        #         if i == j:
        #             KL_U += KL_Gaussian(self.mu_U[i, j, :], self.Sigma_U[i, j, :, :], torch.zeros(self.M).type(TensorType).to(device), K_L1_22)
        #         else:
        #             KL_U += KL_Gaussian(self.mu_U[i, j, :], self.Sigma_U[i, j, :, :], torch.zeros(self.M).type(TensorType).to(device), K_L0_22)

        mu_U_1_list = list()
        Sigma_U_1_list = list()
        mu_U_0_list = list()
        Sigma_U_0_list = list()
        for i in range(self.D):
            mu_U_1_list.append(self.mu_U[i, i, :])
            Sigma_U_1_list.append(self.Sigma_U[i, i, :, :])
            if i > 0:
               mu_U_0_list.append(self.mu_U[i, :i, :].view(i, self.M))
               Sigma_U_0_list.append(self.Sigma_U[i, :i, :, :].view(i, self.M, self.M))

        KL_U += KL_Gaussian(torch.stack(mu_U_1_list), torch.stack(Sigma_U_1_list),
                            torch.zeros(self.M).type(TensorType).to(device), K_L1_22).sum()
        KL_U += KL_Gaussian(torch.cat(mu_U_0_list), torch.cat(Sigma_U_0_list),
                            torch.zeros(self.M).type(TensorType).to(device), K_L0_22).sum()
        if verbose:
            print("computing the regularization term costs {}s".format(time.time() - t1))

        # Compute SELBO
        SELBO = self.N / batch_size * SELBO_R - KL_W - KL_v - KL_U
        return -SELBO

    def compute_ELBO(self, inputs_list, outputs_list, index=None, n_sample=1000, verbose=False):
        if index is not None:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, index)])
        else:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, range(self.D))])
        output_index = torch.from_numpy(np.stack([np.arange(I.shape[0]), I], axis=1)).type(torch.long)
        inputs = torch.cat(inputs_list).view(-1, 1)
        outputs = torch.cat(outputs_list).view(-1, 1)

        # reparameterization
        l_sqrt_W = mat2ltri(self.sqrt_W.cpu().detach())
        self.Sigma_W = torch.matmul(l_sqrt_W, l_sqrt_W.permute(0, 2, 1))
        l_sqrt_v = mat2ltri(self.sqrt_v.cpu().detach())
        self.Sigma_v = torch.matmul(l_sqrt_v, l_sqrt_v.permute(1, 0))
        l_sqrt_U = mat2ltri(self.sqrt_U.cpu().detach())
        self.Sigma_U = torch.matmul(l_sqrt_U, l_sqrt_U.permute(0, 1, 3, 2))

        self.sigma2_tildeell = torch.exp(self.sigma2_tildeell_log.cpu().detach())
        self.length_scales_tildeell = torch.exp(self.length_scales_tildeell_log.cpu().detach())
        self.sigma2_L0 = torch.exp(self.sigma2_L0_log.cpu().detach())
        self.length_scales_L0 = torch.exp(self.length_scales_L0_log.cpu().detach())
        self.sigma2_L1 = torch.exp(self.sigma2_L1_log.cpu().detach())
        self.length_scales_L1 = torch.exp(self.length_scales_L1_log.cpu().detach())
        self.sigma2_err = torch.exp(self.sigma2_err_log.cpu().detach())

        sampled_log_likelihoods = list()
        input_size = inputs.shape[0]
        for index in range(n_sample):
            if verbose:
                print("Monte Carlo index:", index)
            # sample tilde_ell_star
            K_tildeell_11_diag = torch.ones(self.N).type(TensorType) * self.sigma2_tildeell
            K_tildeell_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_tildeell.cpu().detach(),
                                       length_scales=self.length_scales_tildeell.cpu().detach())
            K_tildeell_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_tildeell.cpu().detach(), length_scales=self.length_scales_tildeell.cpu().detach())
            sampled_v_tilde_ell = JGP_S(K_tildeell_11_diag, K_tildeell_12, K_tildeell_22, self.mu_v.cpu().detach(), self.Sigma_v.cpu().detach())
            sampled_tilde_ell = sampled_v_tilde_ell[:input_size]
            sampled_v = sampled_v_tilde_ell[input_size:]
            sampled_ell_Z = torch.exp(sampled_v)
            sampled_ell_X = torch.exp(sampled_tilde_ell)
            # sample L
            K_L0_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_L0.cpu().detach()
            K_L0_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L0.cpu().detach(), length_scales=self.length_scales_L0.cpu().detach())
            K_L0_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L0.cpu().detach(), length_scales=self.length_scales_L0.cpu().detach())
            K_L1_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_L1.cpu().detach()
            K_L1_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L1, length_scales=self.length_scales_L1.cpu().detach())
            K_L1_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L1.cpu().detach(), length_scales=self.length_scales_L1.cpu().detach())
            sampled_L = torch.zeros(self.D, self.D, input_size).type(TensorType)
            for i in range(self.D):
                for j in range(i + 1):
                    if i == j:
                        sampled_logLii = MGP_d(K_L1_12, K_L1_22, K_L1_11_diag, self.mu_U.cpu().detach()[i, j, :],
                                               self.Sigma_U.cpu().detach()[i, j, :, :])
                        sampled_L[i, j, :] = torch.exp(sampled_logLii)
                    else:
                        sampled_Lij = MGP_d(K_L0_12, K_L0_22, K_L0_11_diag, self.mu_U.cpu().detach()[i, j, :],
                                            self.Sigma_U.cpu().detach()[i, j, :, :])
                        sampled_L[i, j, :] = sampled_Lij
            sampled_l = sampled_L.permute(2,1,0)[output_index[:,0], output_index[:,1]]

            # sample G_star
            K_G_12 = create_Gibbs(inputs, self.Z.cpu().detach(), sampled_ell_X, sampled_ell_Z)
            K_G_22 = create_Gibbs(self.Z.cpu().detach(), self.Z.cpu().detach(), sampled_ell_Z, sampled_ell_Z)
            K_G_11_diag = torch.ones(input_size).type(TensorType)
            # compute the marginal mean and variance for g with dimension D x batch_size
            mu_g, sigma2_g = MGP_mu_sigma2(K_G_12, K_G_22, K_G_11_diag, self.mu_W, self.Sigma_W)
            # Alternative
            # sampled_G = MGP_d(K_G_12, K_G_22, K_G_11_diag, self.mu_W.cpu().detach(), self.Sigma_W.cpu().detach())

            sampled_F = torch.sum(torch.mul(sampled_l, mu_g.t()), axis=1).view(-1, 1)
            # import pdb; pdb.set_trace()
            SELBO_R = Normal_logprob(sampled_F, torch.sqrt(self.sigma2_err), outputs)
            SELBO_R -= 0.5 / self.sigma2_err * (sampled_l ** 2 * sigma2_g.t()).sum()
            sampled_log_likelihoods.append(SELBO_R)
            # Alternative
            # sampled_F =torch.sum(torch.mul(sampled_l, sampled_G.t()), axis=1).view(-1,1)
            # sampled_log_likelihoods.append(Normal_logprob(sampled_F, torch.sqrt(self.sigma2_err.cpu().detach()), outputs))
            # import pdb; pdb.set_trace()
        sampled_log_likelihoods = torch.stack(sampled_log_likelihoods)

        # KL divergence
        # import pdb; pdb.set_trace()
        KL_W = KL_Gaussian(self.mu_W.cpu().detach(), self.Sigma_W.cpu().detach(), torch.zeros(self.M).type(TensorType), K_G_22, device0='cpu').sum()
        KL_v = KL_Gaussian(self.mu_v.cpu().detach(), self.Sigma_v.cpu().detach(), torch.zeros(self.M).type(TensorType), K_tildeell_22, device0='cpu')
        KL_U = 0
        mu_U_1_list = list()
        Sigma_U_1_list = list()
        mu_U_0_list = list()
        Sigma_U_0_list = list()
        for i in range(self.D):
            mu_U_1_list.append(self.mu_U.cpu().detach()[i, i, :])
            Sigma_U_1_list.append(self.Sigma_U.cpu().detach()[i, i, :, :])
            if i > 0:
                mu_U_0_list.append(self.mu_U[i, :i, :].cpu().detach().view(i, self.M))
                Sigma_U_0_list.append(self.Sigma_U[i, :i, :, :].cpu().detach().view(i, self.M, self.M))
        # import pdb; pdb.set_trace()
        KL_U += KL_Gaussian(torch.stack(mu_U_1_list), torch.stack(Sigma_U_1_list),
                            torch.zeros(self.M).type(TensorType), K_L1_22, device0='cpu').sum()
        KL_U += KL_Gaussian(torch.cat(mu_U_0_list), torch.cat(Sigma_U_0_list),
                            torch.zeros(self.M).type(TensorType), K_L0_22, device0='cpu').sum()

        return torch.mean(sampled_log_likelihoods) - KL_W - KL_v - KL_U

    def sample_Y(self, inputs_list, index=None, n_sample=1000):
        if index is not None:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, index)])
        else:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, range(self.D))])
        output_index = torch.from_numpy(np.stack([np.arange(I.shape[0]), I], axis=1)).type(torch.long)
        inputs = torch.cat(inputs_list).view(-1, 1)

        # reparameterization
        l_sqrt_W = mat2ltri(self.sqrt_W.cpu().detach())
        self.Sigma_W = torch.matmul(l_sqrt_W, l_sqrt_W.permute(0, 2, 1))
        l_sqrt_v = mat2ltri(self.cpu().sqrt_v.detach())
        self.Sigma_v = torch.matmul(l_sqrt_v, l_sqrt_v.permute(1, 0))
        l_sqrt_U = mat2ltri(self.cpu().sqrt_U.detach())
        self.Sigma_U = torch.matmul(l_sqrt_U, l_sqrt_U.permute(0, 1, 3, 2))

        self.sigma2_tildeell = torch.exp(self.sigma2_tildeell_log.cpu().detach())
        self.length_scales_tildeell = torch.exp(self.length_scales_tildeell_log.cpu().detach())
        self.sigma2_L0 = torch.exp(self.sigma2_L0_log.cpu().detach())
        self.length_scales_L0 = torch.exp(self.length_scales_L0_log.cpu().detach())
        self.sigma2_L1 = torch.exp(self.sigma2_L1_log.cpu().detach())
        self.length_scales_L1 = torch.exp(self.length_scales_L1_log.cpu().detach())
        self.sigma2_err = torch.exp(self.sigma2_err_log.cpu().detach())

        sampled_Ys = list()
        sampled_Ls = list()
        sampled_Gs = list()
        sampled_tilde_ells = list()
        input_size = inputs.shape[0]
        for i in range(n_sample):
            # print("{}th sampling completed.".format(i))
            # sample tilde_ell_star
            K_tildeell_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_tildeell.cpu().detach()
            K_tildeell_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_tildeell.cpu().detach(),
                                       length_scales=self.length_scales_tildeell.cpu().detach())
            K_tildeell_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_tildeell.cpu().detach(), length_scales=self.length_scales_tildeell.cpu().detach())
            sampled_v_tilde_ell = JGP_S(K_tildeell_11_diag, K_tildeell_12, K_tildeell_22, self.mu_v.cpu().detach(), self.Sigma_v.cpu().detach())
            sampled_tilde_ell_star = sampled_v_tilde_ell[:input_size]
            sampled_v = sampled_v_tilde_ell[input_size:]
            sampled_ell_Z = torch.exp(sampled_v)
            sampled_ell_X = torch.exp(sampled_tilde_ell_star)

            K_L0_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L0.cpu().detach(), length_scales=self.length_scales_L0.cpu().detach())
            K_L0_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L0.cpu().detach(), length_scales=self.length_scales_L0.cpu().detach())
            K_L0_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_L0.cpu().detach()
            K_L1_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L1.cpu().detach(), length_scales=self.length_scales_L1.cpu().detach())
            K_L1_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L1.cpu().detach(), length_scales=self.length_scales_L1.cpu().detach())
            K_L1_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_L1.cpu().detach()

            sampled_L_star = torch.zeros(self.D, self.D, input_size).type(TensorType)
            for i in range(self.D):
                for j in range(i + 1):
                    if i == j:
                        sampled_logLii_star = MGP_d(K_L1_12, K_L1_22, K_L1_11_diag, self.mu_U.cpu().detach()[i, j, :],
                                               self.Sigma_U.cpu().detach()[i, j, :, :])
                        sampled_L_star[i, j, :] = torch.exp(sampled_logLii_star)
                    else:
                        sampled_Lij_star = MGP_d(K_L0_12, K_L0_22, K_L0_11_diag, self.mu_U.cpu().detach()[i, j, :],
                                            self.Sigma_U.cpu().detach()[i, j, :, :])
                        sampled_L_star[i, j, :] = sampled_Lij_star
            sampled_l_star = sampled_L_star.permute(2, 0, 1)[output_index[:, 0], output_index[:, 1]]

            # sample G_star
            K_G_12 = create_Gibbs(inputs, self.Z.cpu().detach(), sampled_ell_X, sampled_ell_Z)
            K_G_22 = create_Gibbs(self.Z.cpu().detach(), self.Z.cpu().detach(), sampled_ell_Z, sampled_ell_Z)
            K_G_11_diag = torch.ones(input_size).type(TensorType)
            sampled_G_star = MGP_d(K_G_12, K_G_22, K_G_11_diag, self.mu_W.cpu().detach(), self.Sigma_W.cpu().detach())

            # sample Y_star
            sampled_F_star = torch.sum(torch.mul(sampled_l_star, sampled_G_star.t()), axis=1)
            # sampled_F_star = torch.matmul(sampled_L_star.permute(2, 0, 1), sampled_G_star.permute(1, 0).unsqueeze(2))[:,
            #                  :, 0]

            z_F_star = torch.randn(sampled_F_star.size()).type(TensorType)
            sampled_Y_star = reparameterize(sampled_F_star, torch.ones_like(sampled_F_star) * self.sigma2_err.cpu().detach(), z_F_star,
                                            full_cov=False)

            sampled_Ys.append(sampled_Y_star)
            sampled_Ls.append(sampled_l_star)
            sampled_Gs.append(sampled_G_star)
            sampled_tilde_ells.append(sampled_tilde_ell_star)
        sampled_Ys = torch.stack(sampled_Ys)
        sampled_Ls = torch.stack(sampled_Ls)
        sampled_Gs = torch.stack(sampled_Gs)
        sampled_tilde_ells = torch.stack(sampled_tilde_ells)
        return sampled_Ys, sampled_Ls, sampled_Gs, sampled_tilde_ells

    def sample_FY(self, inputs, n_sample=1000):
        inputs = inputs.view(-1, 1)
        # reparameterization
        l_sqrt_W = mat2ltri(self.sqrt_W.cpu().detach())
        self.Sigma_W = torch.matmul(l_sqrt_W, l_sqrt_W.permute(0, 2, 1))
        l_sqrt_v = mat2ltri(self.sqrt_v.cpu().detach())
        self.Sigma_v = torch.matmul(l_sqrt_v, l_sqrt_v.permute(1, 0))
        l_sqrt_U = mat2ltri(self.sqrt_U.cpu().detach())
        self.Sigma_U = torch.matmul(l_sqrt_U, l_sqrt_U.permute(0, 1, 3, 2))

        self.sigma2_tildeell = torch.exp(self.sigma2_tildeell_log.cpu().detach())
        self.length_scales_tildeell = torch.exp(self.length_scales_tildeell_log.cpu().detach())
        self.sigma2_L0 = torch.exp(self.sigma2_L0_log.cpu().detach())
        self.length_scales_L0 = torch.exp(self.length_scales_L0_log.cpu().detach())
        self.sigma2_L1 = torch.exp(self.sigma2_L1_log.cpu().detach())
        self.length_scales_L1 = torch.exp(self.length_scales_L1_log.cpu().detach())
        self.sigma2_err = torch.exp(self.sigma2_err_log.cpu().detach())

        sampled_tilde_ells = list()
        sampled_Ys = list()
        sampled_corrs = list()

        input_size = inputs.shape[0]
        for i in range(n_sample):
            if i % 100 == 99:
                print("{}th sampling completed.".format(i+1))
            # sample tilde_ell_star
            K_tildeell_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_tildeell.cpu().detach()
            K_tildeell_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_tildeell.cpu().detach(),
                                       length_scales=self.length_scales_tildeell.cpu().detach())
            K_tildeell_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_tildeell.cpu().detach(),
                                       length_scales=self.length_scales_tildeell.cpu().detach())
            sampled_v_tilde_ell = JGP_S(K_tildeell_11_diag, K_tildeell_12, K_tildeell_22, self.mu_v.cpu().detach(), self.Sigma_v.cpu().detach())
            sampled_tilde_ell_star = sampled_v_tilde_ell[:input_size]
            sampled_v = sampled_v_tilde_ell[input_size:]
            sampled_ell_Z = torch.exp(sampled_v)
            sampled_ell_X = torch.exp(sampled_tilde_ell_star)

            K_L0_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L0.cpu().detach(), length_scales=self.length_scales_L0.cpu().detach())
            K_L0_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L0.cpu().detach(), length_scales=self.length_scales_L0.cpu().detach())
            K_L0_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_L0.cpu().detach()
            K_L1_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L1.cpu().detach(), length_scales=self.length_scales_L1.cpu().detach())
            K_L1_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L1.cpu().detach(), length_scales=self.length_scales_L1.cpu().detach())
            K_L1_11_diag = torch.ones(input_size).type(TensorType) * self.sigma2_L1.cpu().detach()

            sampled_L_star = torch.zeros(self.D, self.D, input_size).type(TensorType)
            for i in range(self.D):
                for j in range(i + 1):
                    if i == j:
                        sampled_logLii_star = MGP_d(K_L1_12, K_L1_22, K_L1_11_diag, self.mu_U.cpu().detach()[i, j, :],
                                               self.Sigma_U.cpu().detach()[i, j, :, :])
                        sampled_L_star[i, j, :] = torch.exp(sampled_logLii_star)
                    else:
                        sampled_Lij_star = MGP_d(K_L0_12, K_L0_22, K_L0_11_diag, self.mu_U.cpu().detach()[i, j, :],
                                            self.Sigma_U.cpu().detach()[i, j, :, :])
                        sampled_L_star[i, j, :] = sampled_Lij_star

            # sample G_star
            K_G_12 = create_Gibbs(inputs, self.Z.cpu().detach(), sampled_ell_X, sampled_ell_Z)
            K_G_22 = create_Gibbs(self.Z.cpu().detach(), self.Z.cpu().detach(), sampled_ell_Z,
                                  sampled_ell_Z)
            K_G_11_diag = torch.ones(input_size).type(TensorType)
            sampled_G_star = MGP_d(K_G_12, K_G_22, K_G_11_diag, self.mu_W.cpu().detach(),
                                   self.Sigma_W.cpu().detach())

            # sample Y_star
            sampled_F_star = torch.matmul(sampled_L_star.permute(2, 0, 1), sampled_G_star.permute(1, 0).unsqueeze(2))[:,
                             :, 0]
            z_F_star = torch.randn(sampled_F_star.size()).type(TensorType)
            sampled_Y_star = reparameterize(sampled_F_star,
                                            torch.ones_like(sampled_F_star) * self.sigma2_err.cpu().detach(), z_F_star,
                                            full_cov=False)

            # sample correlation matrices
            sampled_cov_star = torch.matmul(sampled_L_star.permute(2, 0, 1), sampled_L_star.permute(2, 1, 0))
            sampled_invstd_star = torch.sqrt(torch.diag_embed(1./torch.diagonal(sampled_cov_star, dim1=-2, dim2=-1)))
            sampled_corr_star = torch.matmul(torch.matmul(sampled_invstd_star, sampled_cov_star), sampled_invstd_star)
            # import pdb; pdb.set_trace()

            # saving sample results
            sampled_tilde_ells.append(sampled_tilde_ell_star)
            sampled_Ys.append(sampled_Y_star)
            sampled_corrs.append(sampled_corr_star)

        sampled_tilde_ells = torch.stack(sampled_tilde_ells)
        sampled_Ys = torch.stack(sampled_Ys)
        sampled_corrs = torch.stack(sampled_corrs)
        return sampled_tilde_ells, sampled_Ys, sampled_corrs

    def sample_Y_gpu(self, inputs_list, index=None, n_sample=1000):
        if index is not None:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, index)])
        else:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, range(self.D))])
        output_index = torch.from_numpy(np.stack([np.arange(I.shape[0]), I], axis=1)).type(torch.long)
        inputs = torch.cat(inputs_list).view(-1, 1)

        # reparameterization
        l_sqrt_W = mat2ltri(self.sqrt_W.detach())
        self.Sigma_W = torch.matmul(l_sqrt_W, l_sqrt_W.permute(0, 2, 1))
        l_sqrt_v = mat2ltri(self.sqrt_v.detach())
        self.Sigma_v = torch.matmul(l_sqrt_v, l_sqrt_v.permute(1, 0))
        l_sqrt_U = mat2ltri(self.sqrt_U.detach())
        self.Sigma_U = torch.matmul(l_sqrt_U, l_sqrt_U.permute(0, 1, 3, 2))

        self.sigma2_tildeell = torch.exp(self.sigma2_tildeell_log.detach())
        self.length_scales_tildeell = torch.exp(self.length_scales_tildeell_log.detach())
        self.sigma2_L0 = torch.exp(self.sigma2_L0_log.detach())
        self.length_scales_L0 = torch.exp(self.length_scales_L0_log.detach())
        self.sigma2_L1 = torch.exp(self.sigma2_L1_log.detach())
        self.length_scales_L1 = torch.exp(self.length_scales_L1_log.detach())
        self.sigma2_err = torch.exp(self.sigma2_err_log.detach())

        sampled_Ys = list()
        sampled_Ls = list()
        sampled_Gs = list()
        sampled_tilde_ells = list()
        input_size = inputs.shape[0]
        for i in range(n_sample):
            # print("{}th sampling completed.".format(i))
            # sample tilde_ell_star
            K_tildeell_11_diag = torch.ones(input_size).type(TensorType).to(device) * self.sigma2_tildeell.detach()
            K_tildeell_12 = create_RBF(inputs, self.Z.detach(), scale2=self.sigma2_tildeell.detach(),
                                       length_scales=self.length_scales_tildeell.detach())
            K_tildeell_22 = create_RBF(self.Z.detach(), scale2=self.sigma2_tildeell.detach(), length_scales=self.length_scales_tildeell.detach())
            sampled_v_tilde_ell = JGP_S(K_tildeell_11_diag, K_tildeell_12, K_tildeell_22, self.mu_v.detach(), self.Sigma_v.detach())
            sampled_tilde_ell_star = sampled_v_tilde_ell[:input_size]
            sampled_v = sampled_v_tilde_ell[input_size:]
            sampled_ell_Z = torch.exp(sampled_v)
            sampled_ell_X = torch.exp(sampled_tilde_ell_star)

            K_L0_12 = create_RBF(inputs, self.Z.detach(), scale2=self.sigma2_L0.detach(), length_scales=self.length_scales_L0.detach())
            K_L0_22 = create_RBF(self.Z.detach(), scale2=self.sigma2_L0.detach(), length_scales=self.length_scales_L0.detach())
            K_L0_11_diag = torch.ones(input_size).type(TensorType).to(device) * self.sigma2_L0.detach()
            K_L1_12 = create_RBF(inputs, self.Z.detach(), scale2=self.sigma2_L1.detach(), length_scales=self.length_scales_L1.detach())
            K_L1_22 = create_RBF(self.Z.detach(), scale2=self.sigma2_L1.detach(), length_scales=self.length_scales_L1.detach())
            K_L1_11_diag = torch.ones(input_size).type(TensorType).to(device) * self.sigma2_L1.detach()

            sampled_L_star = torch.zeros(self.D, self.D, input_size).type(TensorType).to(device)
            for i in range(self.D):
                for j in range(i + 1):
                    if i == j:
                        sampled_logLii_star = MGP_d(K_L1_12, K_L1_22, K_L1_11_diag, self.mu_U.detach()[i, j, :],
                                               self.Sigma_U.detach()[i, j, :, :])
                        sampled_L_star[i, j, :] = torch.exp(sampled_logLii_star)
                    else:
                        sampled_Lij_star = MGP_d(K_L0_12, K_L0_22, K_L0_11_diag, self.mu_U.detach()[i, j, :],
                                            self.Sigma_U.detach()[i, j, :, :])
                        sampled_L_star[i, j, :] = sampled_Lij_star

            # sample G_star
            K_G_12 = create_Gibbs(inputs, self.Z.detach(), sampled_ell_X, sampled_ell_Z)
            K_G_22 = create_Gibbs(self.Z.detach(), self.Z.detach(), sampled_ell_Z, sampled_ell_Z)
            K_G_11_diag = torch.ones(input_size).type(TensorType).to(device)
            sampled_G_star = MGP_d(K_G_12, K_G_22, K_G_11_diag, self.mu_W.detach(), self.Sigma_W.detach())

            # sample Y_star
            sampled_F_star = torch.matmul(sampled_L_star.permute(2, 0, 1), sampled_G_star.permute(1, 0).unsqueeze(2))[:,
                             :, 0]

            z_F_star = torch.randn(sampled_F_star.size()).type(TensorType).to(device)
            sampled_Y_star = reparameterize(sampled_F_star, torch.ones_like(sampled_F_star) * self.sigma2_err.detach(), z_F_star,
                                            full_cov=False)
            sampled_Ys.append(sampled_Y_star[output_index[:,0], output_index[:,1]])
            sampled_Ls.append(sampled_L_star)
            sampled_Gs.append(sampled_G_star)
            sampled_tilde_ells.append(sampled_tilde_ell_star)
        sampled_Ys = torch.stack(sampled_Ys)
        sampled_Ls = torch.stack(sampled_Ls)
        sampled_Gs = torch.stack(sampled_Gs)
        sampled_tilde_ells = torch.stack(sampled_tilde_ells)
        return sampled_Ys, sampled_Ls, sampled_Gs, sampled_tilde_ells

    def predict_Y(self, inputs_list, index=None):
        if index is not None:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, index)])
        else:
            I = np.hstack([np.repeat(j, _x.size(0)) for _x, j in zip(inputs_list, range(self.D))])
        output_index = torch.from_numpy(np.stack([np.arange(I.shape[0]), I], axis=1)).type(torch.long)
        inputs = torch.cat(inputs_list).view(-1, 1)

        # reparameterization
        l_sqrt_W = mat2ltri(self.sqrt_W.cpu().detach())
        self.Sigma_W = torch.matmul(l_sqrt_W, l_sqrt_W.permute(0, 2, 1))
        l_sqrt_v = mat2ltri(self.sqrt_v.cpu().detach())
        self.Sigma_v = torch.matmul(l_sqrt_v, l_sqrt_v.permute(1, 0))
        l_sqrt_U = mat2ltri(self.sqrt_U.cpu().detach())
        self.Sigma_U = torch.matmul(l_sqrt_U, l_sqrt_U.permute(0, 1, 3, 2))

        self.sigma2_tildeell = torch.exp(self.sigma2_tildeell_log.cpu().detach())
        self.length_scales_tildeell = torch.exp(self.length_scales_tildeell_log.cpu().detach())
        self.sigma2_L0 = torch.exp(self.sigma2_L0_log.cpu().detach())
        self.length_scales_L0 = torch.exp(self.length_scales_L0_log.cpu().detach())
        self.sigma2_L1 = torch.exp(self.sigma2_L1_log.cpu().detach())
        self.length_scales_L1 = torch.exp(self.length_scales_L1_log.cpu().detach())
        self.sigma2_err = torch.exp(self.sigma2_err_log.cpu().detach())

        input_size = inputs.shape[0]
        # estimate ell
        # import pdb;pdb.set_trace()
        K_tildeell_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_tildeell,
                                   length_scales=self.length_scales_tildeell)
        K_tildeell_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_tildeell,
                                   length_scales=self.length_scales_tildeell)
        est_v = self.mu_v.cpu().detach()
        # import pdb; pdb.set_trace()
        est_tilde_ell_star = MGP_mu(K_tildeell_12, K_tildeell_22, est_v, device0='cpu')
        est_ell_Z = torch.exp(est_v)
        est_ell_X = torch.exp(est_tilde_ell_star)
        # estimate L
        K_L0_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L0, length_scales=self.length_scales_L0)
        K_L0_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L0, length_scales=self.length_scales_L0)
        K_L1_12 = create_RBF(inputs, self.Z.cpu().detach(), scale2=self.sigma2_L1, length_scales=self.length_scales_L1)
        K_L1_22 = create_RBF(self.Z.cpu().detach(), scale2=self.sigma2_L1, length_scales=self.length_scales_L1)
        est_L_star = torch.zeros(self.D, self.D, input_size).type(TensorType)
        for i in range(self.D):
            for j in range(i + 1):
                if i == j:
                    est_logLii = MGP_mu(K_L1_12, K_L1_22, self.mu_U.cpu().detach()[i, j, :], device0='cpu')
                    est_L_star[i, j, :] = torch.exp(est_logLii)
                else:
                    est_Lij = MGP_mu(K_L0_12, K_L0_22, self.mu_U.cpu().detach()[i, j, :], device0='cpu')
                    est_L_star[i, j, :] = est_Lij
        # estimate G_star
        K_K_12 = create_Gibbs(inputs, self.Z.cpu().detach(), est_ell_X, est_ell_Z)
        K_K_22 = create_Gibbs(self.Z.cpu().detach(), self.Z.cpu().detach(), est_ell_Z, est_ell_Z)
        est_G_star = MGP_mu(K_K_12, K_K_22, self.mu_W.cpu().detach(), device0='cpu')
        # est Y_star
        est_Y_star = torch.matmul(est_L_star.permute(2, 0, 1), est_G_star.permute(1, 0).unsqueeze(2))[:, :, 0]
        return est_Y_star[output_index[:,0], output_index[:,1]]


def plot_samples(grids, S, true_X, true_Y):
    samples_quantiles = np.percentile(S, q=np.array([2.5, 97.5]), axis=0)
    samples_mean = np.mean(S, axis=0)
    n_sample, n_grid, n_dim = S.shape
    for d in range(n_dim):
        fig = plt.figure()
        plt.plot(grids, samples_mean[:, d], color='b')
        plt.plot(grids, samples_quantiles[:, :, d].T, color='r', linestyle='dashed')
        plt.scatter(true_X.reshape(-1), true_Y[:, d], color="black")
        fig.savefig("Dim{}.pdf".format(d))


def pre_intialization(M, D, factor=1e-2):
    mu_W = np.zeros([D, M])
    sqrt_v = np.eye(M) * factor
    sqrt_W = np.stack([np.eye(M) for _ in range(D)]) * factor
    sqrt_U = np.stack([np.stack([np.eye(M) for _ in range(D)]) for _ in range(D)]) * factor
    return mu_W, sqrt_v, sqrt_W, sqrt_U


def vec2list(X, Y, I, dim, device=None):
    X_list = list()
    Y_list = list()
    for m in range(dim):
        if device is None:
            X_list.append(X[I == m])
            Y_list.append(Y[I == m])
        else:
            X_list.append(X[I == m].to(device))
            Y_list.append(Y[I == m].to(device))
    return X_list, Y_list


def inference(X_train_list, Y_train_list, z, batch_size, dim_outputs, hyperpars=None, fix_hyperpars=True, mu_v=None,
              mu_W=None, mu_U=None, sqrt_v=None, sqrt_W=None, sqrt_U=None, lr=0.01, itnum=1000,
              do_stop_criterion=False, seed=22, verbose=False, PATH="model.pt", continuous_training=False,
              show_ELBO=True, save_model=False, X_test_list=None, Y_test_list=None):
    X_train_vec = np.concatenate(X_train_list)
    Y_train_vec = np.concatenate(Y_train_list)
    train_index = np.concatenate([np.ones_like(Y_train_list[i]) * i for i in range(dim_outputs)]).astype(int)

    X = torch.from_numpy(X_train_vec).type(TensorType)
    Y = torch.from_numpy(Y_train_vec).type(TensorType)
    I = torch.from_numpy(train_index).type(TensorType)
    Z = torch.from_numpy(z).type(TensorType).unsqueeze(1).to(device)
    X_list, Y_list = vec2list(X, Y, I, dim=dim_outputs)

    NMGP_model = NMGP(number_observations=Y_train_vec.shape[0], dim_outputs=dim_outputs, Z=Z, minibatch_size=batch_size,
                      mu_v=mu_v, mu_W=mu_W, mu_U=mu_U, sqrt_v=sqrt_v, sqrt_W=sqrt_W, sqrt_U=sqrt_U, seed=seed)

    # optimizer = torch.optim.Adam([{'params': NMGP_model.parameters()}, {'params': NMGP_model.sqrt_U, 'lr': 1e-4}], lr=1e-2)
    NMGP_model.to(device)
    optimizer = torch.optim.Adam(NMGP_model.parameters(), lr=lr)

    if hyperpars is not None:
        if "sigma2_tildeell_log" in hyperpars:
            NMGP_model.sigma2_tildeell_log.data.fill_(hyperpars['sigma2_tildeell_log'])
        if "sigma2_L0_log" in hyperpars:
            NMGP_model.sigma2_L0_log.data.fill_(hyperpars['sigma2_L0_log'])
        if "sigma2_L1_log" in hyperpars:
            NMGP_model.sigma2_L0_log.data.fill_(hyperpars['sigma2_L1_log'])
        if "sigma2_err_log" in hyperpars:
            NMGP_model.sigma2_err_log.data.fill_(hyperpars['sigma2_err_log'])

    if continuous_training:
        checkpoint = torch.load(PATH)
        NMGP_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # frozen hyper-parameters
        if fix_hyperpars:
            NMGP_model.length_scales_tildeell_log.requires_grad = False
            NMGP_model.length_scales_L0_log.requires_grad = False
            NMGP_model.length_scales_L1_log.requires_grad = False
            if "length_scales_tildeell_log" in hyperpars:
                NMGP_model.length_scales_tildeell_log.data.fill_(hyperpars['length_scales_tildeell_log'])
            if "length_scales_L0_log" in hyperpars:
                NMGP_model.length_scales_L0_log.data.fill_(hyperpars['length_scales_L0_log'])
            if "length_scales_L1_log" in hyperpars:
                NMGP_model.length_scales_L1_log.data.fill_(hyperpars['length_scales_L1_log'])
    elif fix_hyperpars:
        NMGP_model.length_scales_tildeell_log.requires_grad = False
        NMGP_model.length_scales_L0_log.requires_grad = False
        NMGP_model.length_scales_L1_log.requires_grad = False
        if "length_scales_tildeell_log" in hyperpars:
            NMGP_model.length_scales_tildeell_log.data.fill_(hyperpars['length_scales_tildeell_log'])
        if "length_scales_L0_log" in hyperpars:
            NMGP_model.length_scales_L0_log.data.fill_(hyperpars['length_scales_L0_log'])
        if "length_scales_L1_log" in hyperpars:
            NMGP_model.length_scales_L1_log.data.fill_(hyperpars['length_scales_L1_log'])

    train_data = trainData(X, Y, I)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # training
    loss_list = []
    time_list = []
    if X_test_list is not None:
        rmse_test_list = []
        Y_test_vec = np.concatenate(Y_test_list)

    ts = time.time()
    for epoch in range(itnum):
        batch = 0
        for X_batch, Y_batch, I_batch in train_loader:
            batch = batch + 1
            # print(X_batch.shape, Y_batch.shape)
            optimizer.zero_grad()
            # if torch.cuda.is_available():
            #     print("1. cuda memory:", torch.cuda.memory_allocated())
            # else:
            #     print("1", print_mem(epoch))
            X_batch_list, Y_batch_list = vec2list(X_batch, Y_batch, I_batch, dim=dim_outputs, device=device)
            # import pdb; pdb.set_trace()
            loss = NMGP_model(X_batch_list, Y_batch_list, verbose=verbose)
            # import pdb; pdb.set_trace()
            # if torch.cuda.is_available():
            #     print("2. cuda memory:", torch.cuda.memory_allocated())
            # else:
            #     print("2", print_mem(epoch))
            if verbose:
                t1 = time.time()
            loss.backward(retain_graph=True)
            if verbose:
                print("backward takes {}s".format(time.time()-t1))
            # if torch.cuda.is_available():
            #     print("3. cuda memory:", torch.cuda.memory_allocated())
            # else:
            #     print("3", print_mem(epoch))
            optimizer.step()
            # if torch.cuda.is_available():
            #     print("4. cuda memory:", torch.cuda.memory_allocated())
            # else:
            #     print("4", print_mem(epoch))
            loss_value = loss.detach().data.cpu().numpy()
            # del loss
            loss_list.append(loss_value)
            # record time
            time_list.append(time.time()-ts)

            if X_test_list is not None:
                est_Y_test = predict_Y(NMGP_model, X_test_list)
                rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - Y_test_vec) ** 2))
                rmse_test_list.append(rmse_test)

            if verbose:
                print("epoch: {}/{}, batch: {}/{}, loss: {}".format(epoch, itnum, batch,
                                        X_train_vec.shape[0] / batch_size, loss_value))

            # print(loss.detach())

        if do_stop_criterion:
            if epoch % 5 == 4 and epoch > 5:
                loss_array = np.array(loss_list)
                curr_loss = np.sum(loss_array[-batch:])
                prev_loss = np.sum(loss_array[-batch*6:-batch*5])
                if curr_loss > prev_loss:
                    print("Stop criteria is satisfied.")
                    break

        if epoch % 100 == 99 and show_ELBO:
            elbo = NMGP_model.compute_ELBO(X_list, Y_list)
            print("epoch: {}, ELBO: {}".format(epoch + 1, elbo.detach()))
            print(print_mem(epoch + 1))
            # import pdb; pdb.set_trace()
    print("training takes {}s".format(time.time() - ts))

    # import pdb; pdb.set_trace()
    if save_model:
        torch.save({
            'epoch': epoch,
            'model_state_dict': NMGP_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)

    if show_ELBO:
        elbo = NMGP_model.compute_ELBO(X_list, Y_list)
        print("epoch: {}, ELBO: {}".format(epoch + 1, elbo.detach()))
        print(print_mem(epoch + 1))

    if X_test_list is not None:
        return NMGP_model, loss_list, rmse_test_list, time_list
    else:
        return NMGP_model, loss_list, time_list


def sample_Y(model, X_list, n_sample=1000):
    # convert numpy to tensor
    X_list = [torch.from_numpy(x).type(TensorType) for x in X_list]
    sampled_Ys, sampled_Ls, sampled_Gs, sampled_tilde_ells = model.sample_Y(X_list, n_sample=n_sample)
    # X_list = [torch.from_numpy(x).type(TensorType).to(device) for x in X_list]
    # sampled_Ys, sampled_Ls, sampled_Gs, sampled_tilde_ells = model.sample_Y_gpu(X_list)
    return sampled_Ys.data.numpy(), sampled_Ls.data.numpy(), sampled_Gs.data.numpy(), sampled_tilde_ells.data.numpy()


def sample_FY(model, x, n_sample=1000):
    x = torch.from_numpy(x).type(TensorType)
    sampled_Ys, sampled_tilde_ells, sampled_corrs = model.sample_FY(x, n_sample=n_sample)
    return sampled_Ys.data.numpy(), sampled_tilde_ells.data.numpy(), sampled_corrs.data.numpy()


def predict_Y(model, X_list):
    X_list = [torch.from_numpy(x).type(TensorType) for x in X_list]
    predicted_Ys = model.predict_Y(X_list)
    return predicted_Ys.data.cpu().numpy()


def plot(dim_outputs, X_train_list, Y_train_list, X_test_list, Y_test_list, test_index, est_Y_test, grids, gridy_Y, dir_name=None, name=None):
    number_grids = grids.shape[0]
    for m in range(dim_outputs):
        fig = plt.figure()
        plt.scatter(X_train_list[m], Y_train_list[m])
        plt.plot(grids, gridy_Y[1, m * number_grids:(m + 1) * number_grids], color='b')
        plt.plot(grids, gridy_Y[[0, 2], m * number_grids:(m + 1) * number_grids].T, color="r", linestyle="dashed")
        plt.xlabel("x", fontsize=22)
        plt.ylabel("y{}".format(m + 1), rotation=0, fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.scatter(X_test_list[m], Y_test_list[m], color='k', label="test")
        plt.scatter(X_test_list[m], est_Y_test[test_index == m], color="r", label='prediction')
        plt.tight_layout()
        plt.show()
        if dir_name is not None:
            fig.savefig(dir_name + "Pos_analysis_{}_".format(m) + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", help="number of inducing points", type=int, default=20)
    parser.add_argument("--itnum", help="number of iterations", type=int, default=1000)
    parser.add_argument("--batchsize", help="minibatch size", type=int, default=0)
    parser.add_argument("--data", help="data name", type=str, default='sim_illustration_low_freq')
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    args = parser.parse_args()

    do_inference = True
    do_vi_res_analysis = True
    do_plot_post_process = True

    if not os.path.exists("../res/sim_VI/{}".format(args.data)):
        os.mkdir("../res/sim_VI/{}".format(args.data))

    # Upload Data
    with open("../data/simulation/" + args.data + ".pickle", "rb") as res:
        X_list, Y_list, Xt_list, Yt_list = pickle.load(res)
    z = np.linspace(0, 1, num=args.M)
    dim_outputs = len(X_list)

    # convert data format
    X_train_list = X_list
    X_test_list = Xt_list
    X_train_vec = np.concatenate(X_train_list)
    X_test_vec = np.concatenate(X_test_list)
    Y_train_list = Y_list
    Y_test_list = Yt_list
    Y_train_vec = np.concatenate(Y_train_list)
    Y_test_vec = np.concatenate(Y_test_list)
    train_index = np.concatenate([np.ones_like(Y_train_list[i]) * i for i in range(dim_outputs)]).astype(int)
    test_index = np.concatenate([np.ones_like(Y_test_list[i]) * i for i in range(dim_outputs)]).astype(int)

    if args.batchsize == 0:
        batch_size = X_train_vec.shape[0]
    else:
        batch_size = args.batchsize

    # # Upload Data
    # with open("../data/simulation/" + args.data + ".pickle", "rb") as res:
    #     x, l, L_vecs, sigma2_err, Y = pickle.load(res)
    # M = args.M
    # z = np.linspace(np.min(x), np.max(x), num=M)
    # dim_outputs = Y.shape[1]
    #
    # # split training and testing data
    # x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.33, random_state=42)
    # X_train_list = [x_train, x_train]
    # X_test_list = [x_test, x_test]
    # Y_train_list = [Y_train[:, i] for i in range(dim_outputs)]
    # Y_test_list = [Y_test[:, i] for i in range(dim_outputs)]
    # number_train = int(np.array([v.shape[0] for v in X_train_list]).sum())
    # number_test = int(np.array([v.shape[0] for v in X_test_list]).sum())

    # pre-estimation
    # v_array, U_array, _ = pre_estimation(x_train, Y_train, z, P=10)
    # mu_W, sqrt_v, sqrt_W, sqrt_U = pre_intialization(M, Y_train.shape[1])
    # v_array, U_array, mu_W, sqrt_v, sqrt_W, sqrt_U = None, None, None, None, None, None

    # if args.batchsize == 0:
    #     batch_size = number_train
    # else:
    #     batch_size = args.batchsize

    number_grids = 200

    if do_inference:
        NMGP_model = inference(X_train_list, Y_train_list, z, batch_size, dim_outputs)

        # prediction
        grids = np.linspace(0, 1, number_grids)
        predicted_Ys, _, _, _ = sample_Y(NMGP_model, [grids, grids])
        # reconstruction
        # ts = time.time()
        predicted_Y_train, _, _, _ = sample_Y(NMGP_model, X_train_list)
        est_Y_train = np.mean(predicted_Y_train, axis=0)
        # print(time.time() - ts)

        # alternative way
        # ts = time.time()
        # est_Y_train = predict_Y(NMGP_model, X_train_list)
        # print(time.time() - ts)

        # testing
        predicted_Y_test, _, _, _ = sample_Y(NMGP_model, X_test_list)
        est_Y_test = np.mean(predicted_Y_test, axis=0)
        # alterantive way
        est_Y_test = predict_Y(NMGP_model, X_test_list)

        # save results
        with open("../res/sim_VI/{}/prediction_res_M{}_B{}.pickle".format(args.data, args.M, args.batchsize), "wb") as res:
            pickle.dump([predicted_Ys, predicted_Y_train, est_Y_train, predicted_Y_test, est_Y_test], res)
    else:
        grids = torch.from_numpy(np.linspace(0, 1, 200)[:, None]).type(TensorType).to(device)
        with open("../res/sim_VI/{}/prediction_res_M{}_B{}.pickle".format(args.data, args.M, args.batchsize), "rb") as res:
            predicted_Ys, predicted_Y_train, est_Y_train, predicted_Y_test, est_Y_test = pickle.load(res)

    if do_vi_res_analysis:
        gridy_mean = np.mean(predicted_Ys, axis=0)
        gridy_quantiles = np.percentile(predicted_Ys, q=(2.5, 97.5), axis=0)
        gridy_Y = np.stack([gridy_quantiles[0, :], gridy_mean, gridy_quantiles[1, :]])
        rmse_train = np.sqrt(np.mean((est_Y_train[:, None] - np.vstack(Y_train_list)) ** 2))
        predy_quantiles = np.percentile(predicted_Y_test, q=(2.5, 97.5), axis=0)
        length_test = np.mean(predy_quantiles[1] - predy_quantiles[0])
        rmse_test = np.sqrt(np.mean((est_Y_test[:, None] - np.vstack(Y_test_list)) ** 2))
        # print("rmse_reconstruction: {}".format(rmse_train))
        # print("rmse_predtiction: {}".format(rmse_test))
        # print("average length of CI: {}".format(length_test))
        # compute coverage rate
        CN = np.zeros(dim_outputs)
        TT = np.zeros(dim_outputs)
        for i in range(Y_test_vec.shape[0]):
            if Y_test_vec[i] > predy_quantiles[0, i] and Y_test_vec[i] < predy_quantiles[1, i]:
                CN[test_index[i]] += 1
            TT[test_index[i]] += 1
        CR = CN / TT
        # print("coverage rate", CR)

        if do_plot_post_process:
            for m in range(dim_outputs):
                fig = plt.figure()
                plt.scatter(X_train_list[m], Y_train_list[m])
                plt.plot(grids, gridy_Y[1, m * number_grids:(m + 1) * number_grids], color='b')
                plt.plot(grids, gridy_Y[[0, 2], m * number_grids:(m + 1) * number_grids].T, color="r",
                         linestyle="dashed")
                plt.xlabel("x", fontsize=22)
                plt.ylabel("y{}".format(m + 1), rotation=0, fontsize=22)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.scatter(X_test_list[m], Y_test_list[m], color='k', label="test")
                plt.scatter(X_test_list[m], est_Y_test[:, None][test_index == m], color="r", label='prediction')
                plt.tight_layout()
                plt.legend()
                fig.savefig("../res/sim_VI/{}/Pos_analysis_{}_{}{}_B{}.png".format(args.data, m, 'VI', args.M, args.batchsize))
                plt.show()

        # gridy_mean = np.mean(predicted_Ys, axis=0)
        # gridy_quantiles = np.percentile(predicted_Ys, q=(2.5, 97.5), axis=0)
        # gridy_Y = np.stack([gridy_quantiles[0, :], gridy_mean, gridy_quantiles[1, :]])
        # rmse_train = np.sqrt(np.mean((est_Y_train - np.concatenate(Y_train_list)) ** 2))
        # predy_quantiles = np.percentile(predicted_Y_test, q=(2.5, 97.5), axis=0)
        # length_test = np.mean(predy_quantiles[1] - predy_quantiles[0])
        # rmse_test = np.sqrt(np.mean((est_Y_test - np.concatenate(Y_test_list)) ** 2))
        # print("rmse_reconstruction: {}".format(rmse_train))
        # print("rmse_predtiction: {}".format(rmse_test))
        # print("average length of CI: {}".format(length_test))
        # # import pdb; pdb.set_trace()
        # # compute coverage rate
        # CN = np.zeros(dim_outputs)
        # TT = np.zeros(dim_outputs)
        # Y_test_list = [Y_test[:, i] for i in range(Y_test.shape[1])]
        # Y_test_vec = np.concatenate(Y_test_list)
        # test_index = np.concatenate([np.ones_like(Y_test_list[i]) * i for i in range(dim_outputs)]).astype(int)
        # for i in range(number_test):
        #     if Y_test_vec[i] > predy_quantiles[0, i] and Y_test_vec[i] < predy_quantiles[1, i]:
        #         CN[test_index[i]] += 1
        #     TT[test_index[i]] += 1
        # CR = CN/TT
        # print("coverage rate", CR)
        # # import pdb; pdb.set_trace()
        #
        # dir_name = "../res/sim_VI/{}/".format(args.data)
        # name = "{}{}_B{}.png".format('VI', M, batch_size)
        #
        # plot(dim_outputs, X_train_list, Y_train_list, X_test_list, Y_test_list, test_index, est_Y_test, grids, gridy_Y, dir_name, name)
        # # for m in range(dim_outputs):
        # #     fig = plt.figure()
        # #     plt.scatter(X_train_list[m], Y_train_list[m])
        # #     plt.plot(grids, gridy_Y[1, m*number_grids:(m+1)*number_grids], color='b')
        # #     plt.plot(grids, gridy_Y[[0, 2], m*number_grids:(m+1)*number_grids].T, color="r", linestyle="dashed")
        # #     plt.xlabel("x", fontsize=22)
        # #     plt.ylabel("y{}".format(m + 1), rotation=0, fontsize=22)
        # #     plt.xticks(fontsize=18)
        # #     plt.yticks(fontsize=18)
        # #     plt.scatter(X_test_list[m], Y_test_list[m], color='k', label="test")
        # #     plt.scatter(x_test, est_Y_test[test_index==m], color="r", label='prediction')
        # #     plt.tight_layout()
        # #     fig.savefig("../res/sim_VI/{}/Pos_analysis_{}_{}{}_B{}.png".format(args.data, m, 'VI', M, args.batchsize))

    import pdb; pdb.set_trace()
