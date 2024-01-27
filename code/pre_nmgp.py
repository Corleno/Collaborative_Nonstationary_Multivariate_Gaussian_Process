import pickle
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import pickle

tridiagonal_jitter = 1e-6

def search_nearest_neighhood(x, Y, z_m, P = 10):
    dist = np.abs(x - z_m)
    indices = np.argsort(dist)[:10]
    return x[indices], Y[indices]

def squared_distance(X, X2):
    if X2 is not None:
        difference = np.expand_dims(X, 1) - np.expand_dims(X2, 0)
    else:
        difference = np.expand_dims(X, 1) - np.expand_dims(X2, 0)
    squared_distance = np.sum(difference * difference, -1)
    return squared_distance

def squared_dist(X, X2, length_scales):
    if X2 is None:
        return squared_distance(X / length_scales, X / length_scales)
    else:
        return squared_distance(X / length_scales, X2 / length_scales)

def create_RBF(X, X2=None, scale2=1., length_scales=1.):
    # compute covariance matrix induced by RBF
    if length_scales < 1e-8:
        length_scales = 1e-8
    r2 = squared_dist(X, X2, length_scales)
    return scale2*np.exp(-0.5*r2)

def compute_loglik(pars, x, Y):
    N, D = Y.shape
    log_sigma2_err = pars[0]
    log_ell = pars[1]
    L_vec = pars[2:]
    L = np.zeros([D, D])
    L[np.tril_indices(D)] = L_vec
    B = L.dot(L.T)
    K = create_RBF(x.reshape([-1, 1]), length_scales=np.exp(log_ell))
    C = np.kron(K, B) + np.eye(N*D)*np.exp(log_sigma2_err)
    y_vec = Y.reshape(-1)
    return multivariate_normal.logpdf(y_vec, cov=C)

def compute_loglik_part(pars, x, Y, L):
    N, D = Y.shape
    log_sigma2_err = pars[0]
    log_ell = pars[1]
    B = L.dot(L.T)
    K = create_RBF(x.reshape([-1, 1]), length_scales=np.exp(log_ell))
    C = np.kron(K, B) + np.eye(N*D)*np.exp(log_sigma2_err)
    y_vec = Y.reshape(-1)
    return multivariate_normal.logpdf(y_vec, cov=C)

def objective(pars, x, Y):
    return -compute_loglik(pars, x, Y)

def objective_part(pars, x, Y, L):
    return -compute_loglik_part(pars, x, Y, L)

def pre_estimation_all(x, Y, z, P=10):
    N, D = Y.shape
    sigma2_err_log_list = []
    L_list = []
    ell_list = []
    for z_local in z:
        x_local, Y_local = search_nearest_neighhood(x, Y, z_local, P=P)
        est_L = np.linalg.cholesky(Y_local.T.dot(Y_local)/(P-1))
        initial_pars = np.random.randn(int(D*(1+D)/2+2))
        initial_pars[0] = -6
        initial_pars[1] = -6
        initial_pars[2:] = est_L[np.tril_indices(D)]
        # import pdb
        # pdb.set_trace()
        loglik = compute_loglik(initial_pars, x_local, Y_local)
        print(loglik)
        res = minimize(objective, initial_pars, args=(x_local, Y_local))
        print(-res.fun)
        print("sigma2_err: {}".format(np.exp(res.x[0])))
        print("ell: {}".format(np.exp(res.x[1])))
        L_vec = res.x[2:]
        L = np.zeros([D, D])
        L[np.tril_indices(D)] = L_vec
        print("init_B {}".format((est_L.dot(est_L.T))))
        print("B: {}".format(L.dot(L.T)))
        import pdb
        pdb.set_trace()
        ell_list.append(np.exp(res.x[1]))
        L_vec = res.x[2:]
        L = np.zeros([D, D])
        L[np.tril_indices(D)] = L_vec
        L_list.append(np.linalg.cholesky(L.dot(L.T) + np.eye(D)*tridiagonal_jitter))
        sigma2_err_log_list.append(res.x[0])
    v_array = np.log(np.array(ell_list))
    U_array = np.stack(L_list, axis=-1)
    sigma2_err_log_array = np.array(sigma2_err_log_list)
    return v_array, U_array, sigma2_err_log_array

def pre_estimation_partial(x, Y, z, P=10):
    N, D = Y.shape
    L_tensor = np.stack([np.linalg.cholesky(Y.T.dot(Y)/(N-1)) for i in range(z.shape[0])], axis=-1)
    sigma2_err_log_list = []
    ell_list = []
    for index, z_local in enumerate(z):
        x_local, Y_local = search_nearest_neighhood(x, Y, z_local, P=P)
        est_L = L_tensor[:,:,index]
        initial_pars = np.array([-6,-6])
        # import pdb
        # pdb.set_trace()
        # loglik = compute_loglik_part(initial_pars, x_local, Y_local, est_L)
        # print(loglik)
        res = minimize(objective_part, initial_pars, args=(x_local, Y_local, est_L))
        # print(-res.fun)
        # print("sigma2_err: {}".format(np.exp(res.x[0])))
        # print("ell: {}".format(np.exp(res.x[1])))
        # import pdb
        # pdb.set_trace()
        sigma2_err_log_list.append(res.x[0])
        ell_list.append(np.exp(res.x[1]))
    v_array = np.log(np.array(ell_list))
    sigma2_err_log_array = np.array(sigma2_err_log_list)
    return v_array, L_tensor, sigma2_err_log_array


if __name__ == "__main__":
    # Upload Data
    with open("../data/simulation/sim_MNTS.pickle", "rb") as res:
        x, l, L_vecs, sigma2_err, Y = pickle.load(res)
    P = 10
    z = np.linspace(np.min(x), np.max(x), num=P)

    # estimate the constant U_array
    N, D = Y.shape

    v_array, U_array, sigma2_err_log_array = pre_estimation_partial(x, Y, z, P=P)
    # v_array, U_array, sigma2_err_log_array = pre_estimation_all(x, Y, z, P=P)

    with open("../res/sim_VI/pre_estimation.pickle", "wb") as res:
        pickle.dump([v_array, U_array, sigma2_err_log_array], res)
