import numpy as np


def eculid_affinity(x, y):
    return np.linalg.norm(x - y)


def gaussian_affinity(x, y, sig1=1, sig2=8):
    norm = np.linalg.norm(x - y)
    return np.exp(-norm**2 / (2 * sig1 * sig2))


def cosine_affinity(x, y):
    num = sum(map(float, x * y))
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return round(num / float(denom), 3)


methods = {'gaussian': gaussian_affinity,
           'eculid': eculid_affinity, 'cosine': cosine_affinity}


def affinity_matrix(data, method='gaussian', sig1=1, sig2=8):
    N = data.shape[0]
    ans = np.zeros((N, N))
    kernel = methods[method]

    for i in range(N):
        for j in range(N):
            if method == 'gaussian':
                ans[i][j] = kernel(data[i], data[j], sig1, sig2)
            else:
                ans[i][j] = kernel(data[i], data[j])

    return ans


def laplacian(W, std=False):
    w = np.sum(W, axis=0)
    D = np.diag(w)
    L = D - W
    sqrtD = np.diag(w**(-0.5))

    # D^(-0.5) L D^(-0.5)
    if std:
        return sqrtD.dot(L).dot(sqrtD)
    return L


def spectral_data(L, n_clusters, min=True):
    eigval, eigvec = np.linalg.eig(L)
    eigval = eigval.real
    eigvec = eigvec.real

    # minimal vigvector?
    if not min:
        eigval = -eigval

    X = [eigvec[:, i] for i in np.argsort(eigval)[0:n_clusters]]
    X = np.vstack(X).T

    # normlize eigvector
    for i in range(X.shape[0]):
        X[i] /= np.linalg.norm(X[i])

    return X
