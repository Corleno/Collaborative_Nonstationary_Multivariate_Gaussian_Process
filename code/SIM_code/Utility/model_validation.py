"""
Model validation criteria, including AIC, BIC and DIC
deviance_funct should include observation Y which is 2d observations with dim N by M.
"""

import numpy as np
import torch

def get_AIC(pars, deviance_func, *args, **kwargs):
    """
    Compute AIC value according to the parameter estimates pars
    :param pars: parameter estimates with length N_p
    :param deviance_func: deviance function
    :param args: argumentes of the deviance function
    :param kwargs: keyword arguments of the deviance function
    :return: scalar
    """
    N_p = pars.size(0)
    return deviance_func(pars, *args, **kwargs) + 2*N_p

def get_BIC(pars, deviance_func, *args, **kwargs):
    """
    Compute BIC value according to the parameter estimates pars
    :param pars: parameter estimates with length N_p
    :param deviance_func: deviance function
    :param args: argumentes of the deviance function
    :param kwargs: keyword arguments of the deviance function
    :return: scalar
    """
    N_p = pars.size(0)
    Y = kwargs['Y']
    N, M = Y.size()
    return deviance_func(pars, *args, **kwargs) + np.log(N)*N_p

def get_DIC(pars_hist, deviance_func, *args, **kwargs):
    """
    Compute DIC value according to the parameter estimates pars
    :param pars_hist: posterior samples of parameters with dim N_hist by N_p
    :param deviance_func: deviance function
    :param args: argumentes of the deviance function
    :param kwargs: keyword arguments of the deviance function
    :return: scalar
    """
    N_hist, N_p = pars_hist.size()
    pos_mean_pars = torch.mean(pars_hist, dim=0)
    bar_D = 0
    for i in range(N_hist):
        bar_D += deviance_func(pars_hist[i,:], *args, **kwargs)
    bar_D /= N_hist
    p_D = bar_D - deviance_func(pos_mean_pars, *args, **kwargs)
    return bar_D + p_D


def test_function(pars, Y):
    return pars.sum()+Y.sum()


if __name__ == "__main__":
    x0 = torch.randn(2)
    x0_hist = torch.randn([5,2])
    x1 = torch.ones([3,1])
    print(test_function(x0, x1))

    print(get_AIC(x0, test_function, Y=x1))
    print(get_BIC(x0, test_function, Y=x1))
    print(get_DIC(x0_hist, test_function, Y=x1))

