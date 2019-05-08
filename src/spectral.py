import numpy as np

def affinity_matrix(data, sig1=1,sig2=0.8):
    N=data.shape[0]
    ans=np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            ans[i][j]=gaussian_affinity(data[i],data[j], sig1, sig2)
    
    return ans

def gaussian_affinity(x,y,sig1=1, sig2=0.8):
    norm=np.linalg.norm(x-y)
    return np.exp(-norm**2/(2*sig1*sig2))

def laplacian(W):
    w = np.sum(W, axis=0)
    D = np.diag(w)
    L = D - W
    sqrtD = np.diag(w**(-0.5))

    #D^(-0.5) W D^(-0.5)
    return sqrtD.dot(W).dot(sqrtD)

def spectral_data(L, n_clusters):
    eigval,eigvec=np.linalg.eig(L)
    eigval=eigval.real; eigvec=eigvec.real

    X=[eigvec[:,i] for i in np.argsort(-eigval)[0:n_clusters]]
    X=np.vstack(X).T

    for i in range(X.shape[0]):
        X[i]/=np.linalg.norm(X[i])

    return X