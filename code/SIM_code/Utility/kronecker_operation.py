import torch
import time


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )
    return expanded_t1 * tiled_t2


def kronecker_product_diag(d1, d2):
    """
    Compute the Kronecker production between diagonal matrix D1 and diagonal matrix D2
    :param d1: diagonal array of D1 --- 1d tensor with length N1
    :param d2: diagonal array of D2 --- 1d tensor with length N2
    :return: 1d tensor with length N1N2.
    """
    res = kronecker_product(d1.view([-1, 1]), d2.view([-1, 1]))
    return res.view(-1)


def kron_inv(sigma2, B, K):
    """
    Compute structured matrix inversion (sigma2 I + B otimes K)
    :param sigma2: scalar tensor with dim 1
    :param B: 2D tensor with dim M by M
    :param K: 2D tensor with dim N by N
    :return: 2D tensor MN by MN
    """
    # eigendecomposition
    w_B, v_B = torch.symeig(B, eigenvectors=True)
    # print(torch.svd(K)
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    # print("w_B = {}, v_B={}".format(w_B, v_B))
    # print("w_K = {}, v_K={}".format(w_K, v_K))
    U = kronecker_product(v_B, v_K)
    t = kronecker_product_diag(w_B, w_K)
    A = torch.diag(1./(t + sigma2))
    res = torch.mm(torch.mm(U, A), U.t())
    return res


def kron_logdet(sigma2, B, K):
    """
    Compute structured matrix log determinant (sigma2 I + B otimes K)
    :param sigma2: scalar tensor
    :param B: 2D tensor with dim M by M
    :param K: 2D tensor with dime N by N
    :return: scalar tensor
    """
    # eigendecomposition
    w_B, v_B = torch.symeig(B, eigenvectors=True)
    w_K, v_K = torch.symeig(K, eigenvectors=True)
    T = kronecker_product(torch.diag(w_B), torch.diag(w_K))
    return torch.log(torch.diagonal(T) + sigma2).sum()


def kron_mv(B, K, y):
    """
    Compute structured matrix multiplication (B \otimes K y)
    :param B: 2D tensor with dim M1 by M2
    :param K: 2D tensor with dime N1 by N2
    :param y: 1D tensor with length M2N2
    :return: 1D tensor with length NM
    """
    M = B.size(1)
    N = K.size(1)
    Y = y.view([M, N]).t()
    A = torch.mm(torch.mm(K, Y), B.t())
    a = A.t().contiguous().view(-1)
    return a


if __name__ == "__main__":
    # I = torch.eye(2)
    # R = torch.rand([2,2])
    # print(kronecker_product(I, R))

    # sigma2 = torch.tensor(1.)
    # L_B = torch.randn([2,2])
    # B = torch.mm(L_B, L_B.t())
    # L_K =torch.randn([2,2])
    # K = torch.mm(L_K, L_K.t())
    # print("B, ", B)
    # print("K, ", K)
    # print("log determinant: ", kron_logdet(sigma2, B, K))
    # print("inverse: ", kron_inv(sigma2, B, K))

    L_B = torch.randn([2, 2])
    B = torch.mm(L_B, L_B.t())
    L_K =torch.randn([2, 2])
    K = torch.mm(L_K, L_K.t())
    y = torch.randn(4)
    print("B, ", B)
    print("K, ", K)
    print("y: ", y)
    ts = time.time()
    print(kron_mv(B, K, y))
    print(time.time() - ts)
    ts = time.time()
    print(torch.mv(kronecker_product(B, K), y))
    print(time.time() - ts)