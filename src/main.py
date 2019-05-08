import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from kmeans import kmeans
from util import load_data,accuracy
from spectral import affinity_matrix,laplacian,spectral_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
 

if __name__ == "__main__":
    plt.figure('iris clustering')
    X=load_data('iris.data')
    iris=PCA(n_components=2).fit_transform(X)

    y_pred=kmeans(X,3)
    acc=accuracy(y_pred)
    plt.subplot(121)
    plt.scatter(iris[:,0],iris[:,1],c=y_pred)
    plt.title('kmeans(accuracy:'+str(acc)[:5]+')')

    A=affinity_matrix(X)
    L=laplacian(A)
    spectral_ft=spectral_data(L,3)
    y_pred=kmeans(spectral_ft,3)
    acc=accuracy(y_pred)

    plt.subplot(122)
    plt.scatter(iris[:,0],iris[:,1],c=y_pred)
    plt.title('specttral clustering(accuracy:'+str(acc)[:5]+')')

    plt.savefig('img/result.png')
    plt.show()