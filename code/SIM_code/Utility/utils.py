import torch
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import norm
from copy import deepcopy

from . import settings


def uLvec2Lvec(uL_vec, M):
    T = int(M*(M+1)/2)
    on_diagonal_indices = list(np.cumsum(np.arange(1 ,M+1))-1)
    off_diagonal_indices = [ind for ind in range(T) if ind not in on_diagonal_indices]
    try: 
        L_vec = torch.zeros_like(uL_vec)
        L_vec[on_diagonal_indices] = torch.exp(uL_vec[on_diagonal_indices])
        L_vec[off_diagonal_indices] = uL_vec[off_diagonal_indices]
    except:
        L_vec = np.zeros_like(uL_vec)
        L_vec[on_diagonal_indices] = np.exp(uL_vec[on_diagonal_indices])
        L_vec[off_diagonal_indices] = uL_vec[off_diagonal_indices]
    return L_vec

def Lvec2uLvec(L_vec, M):
    T = int(M*(M+1)/2)
    on_diagonal_indices = list(np.cumsum(np.arange(1 ,M+1))-1)
    off_diagonal_indices = [ind for ind in range(T) if ind not in on_diagonal_indices]
    try: 
        uL_vec = torch.zeros_like(L_vec)
        uL_vec[on_diagonal_indices] = torch.log(L_vec[on_diagonal_indices])
        uL_vec[off_diagonal_indices] = L_vec[off_diagonal_indices]
    except:
        uL_vec = np.zeros_like(L_vec)
        uL_vec[on_diagonal_indices] = np.log(L_vec[on_diagonal_indices])
        uL_vec[off_diagonal_indices] = L_vec[off_diagonal_indices]
    return uL_vec

def uLvecs2Lvecs(uL_vecs, N, M):
    T = int(M*(M+1)/2)
    # import pdb
    # pdb.set_trace()
    try:
        L_vecs = torch.cat([uLvec2Lvec(uL_vecs[n*T: (n+1)*T], M) for n in range(N)])
    except:
        L_vecs = np.concatenate([uLvec2Lvec(uL_vecs[n*T: (n+1)*T], M) for n in range(N)])
    return L_vecs

def Lvecs2uLvecs(L_vecs, N, M):
    T = int(M*(M+1)/2)
    try:
        uL_vecs = torch.cat([Lvec2uLvec(L_vecs[n*T: (n+1)*T], M) for n in range(N)])
    except:
        uL_vecs = np.concatenate([Lvec2uLvec(L_vecs[n*T: (n+1)*T], M) for n in range(N)])
    return uL_vecs

def vec2lowtriangle(x, N = None):
    """
    Convert vector to low-triangle matrix
    :param x: vector with length N(N+1)/2
    :return: 2d tensor with dim N by N
    """
    try:
        if N*(N+1)/2 != x.size(0):
            raise ValueError("check the dimension size!")
        mat = torch.zeros([N, N]).type(settings.torchType)
        tril_indices = torch.tril_indices(N, N)
        mat[tril_indices[0], tril_indices[1]] = x
    except:
        if N*(N+1)/2 != x.shape[0]:
            raise ValueError("check the dimension size!")
        mat = np.zeros([N, N])
        tril_indices = np.tril_indices(N)
        mat[tril_indices[0], tril_indices[1]] = x
    return mat


def lowtriangle2vec(L, N=None):
    """
    Convert  low-triangle matrix to vector
    :param L: low-triangle matrix L
    :return: 1d tensor with size N(N+1)/2
    """
    try:
        tril_indices = torch.tril_indices(N, N)
        return L[tril_indices[0], tril_indices[1]]
    except:
        tril_indices = np.tril_indices(N)
        return L[tril_indices[0], tril_indices[1]]


def data_split_non(x, indx, y, test_size=0.25, random_state=22, shuffle=True):
    """
    Randomly split data into training data and testing data
    :param x: 1d array with length N
    :param indx: 1d array with length N
    :param y: 1d array with length N
    :param test_size: the percentage of testing data
    :param random_state: random seed
    :return: x_train,  x_test, indx_train, indx_test, y_train and y_test
    """
    x_train, x_test, indx_train, indx_test, y_train, y_test = train_test_split(x, indx, y, test_size=test_size,
                                                                               random_state=random_state, shuffle=shuffle)
    return x_train, x_test, indx_train, indx_test, y_train, y_test


def data_split_non_chunk(x, indx, y, chunk_size=0.2, random_state=22, fix=False):
    M = len(np.unique(indx))
    x_train_list = []
    x_test_list = []
    indx_train_list = []
    indx_test_list = []
    y_train_list = []
    y_test_list = []
    np.random.seed(random_state)
    for m in range(M):
        x_m = x[indx == m]
        y_m = y[indx == m]
        n_m = x_m.shape[0]
        n_m_test = int(chunk_size * n_m)
        n_m_train = n_m - n_m_test
        if fix:
            s_m_test = int(np.floor(m*n_m_train/(M-1)))
        else:
            s_m_test = np.random.choice(n_m_train)
        indx_m_train = np.concatenate([np.arange(0, s_m_test), np.arange(s_m_test+n_m_test, n_m)])
        indx_m_test = np.arange(s_m_test, s_m_test + n_m_test)
        x_train_list.append(x_m[indx_m_train])
        x_test_list.append(x_m[indx_m_test])
        indx_train_list.append(m*np.ones(n_m_train))
        indx_test_list.append(m*np.ones(n_m_test))
        y_train_list.append(y_m[indx_m_train])
        y_test_list.append(y_m[indx_m_test])
    return np.concatenate(x_train_list), np.concatenate(x_test_list), np.concatenate(indx_train_list), \
           np.concatenate(indx_test_list), np.concatenate(y_train_list), np.concatenate(y_test_list)


def data_split(x, Y, test_size=0.25, random_state=22, shuffle=True):
    """
    Randomly split data into training data and testing data
    :param x: 1d array with length N
    :param Y: 2d array with length N, M
    :param test_size: the percentage of testing data
    :param random_state: random seed
    :return: x_train, x_test, Y_train and Y_test
    """
    x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    # reorder training data and testing data
    train_indx = np.argsort(x_train)
    test_indx = np.argsort(x_test)
    x_train = x_train[train_indx]
    x_test = x_test[test_indx]
    Y_train = Y_train[train_indx]
    Y_test = Y_test[test_indx]
    return x_train, x_test, Y_train, Y_test


def data_split_extrapolation(x, Y, size=5):
    x_train = x[:-size]
    x_test = x[-size:]
    Y_train = Y[:-size]
    Y_test = Y[-size:]
    return x_train, x_test, Y_train, Y_test


def MSE(x ,y, axis=None):
    """
    Compute mean square errors between x and y
    :param x: array
    :param y: array
    :return: scale
    """
    return np.mean((x - y)**2, axis=axis)


def RMSE(x ,y, axis=None):
    """
    Compute mean square errors between x and y
    :param x: array
    :param y: array
    :return: scale
    """
    return np.sqrt(np.mean((x - y)**2, axis=axis))


def LPD(mean_array, std_array, y_array):
    """
    Compute log predictive density
    :param mean_array: array
    :param std_array: array
    :param y_array: array
    :return: scale
    """
    mean_vec = mean_array.reshape(-1)
    std_vec = std_array.reshape(-1)
    y_vec = y_array.reshape(-1)
    res = np.stack([norm.logpdf(y, loc=mean, scale=std) for mean, std, y in zip(mean_vec, std_vec, y_vec)]).mean()
    return res


if __name__ == "__main__":
    # vec = torch.ones(6)
    # print(vec2lowtriangle(vec, 3))
    # from torch.autograd import Variable
    # vec0 = Variable(vec, requires_grad = True)
    # L = vec2lowtriangle(vec0, N = 3)
    # print(L)
    # L.backward(torch.eye(3))
    # print(vec0.grad)

    x = np.sort(np.random.randn(10))
    Y = np.random.randn(10, 5)
    print(data_split(x, Y))
