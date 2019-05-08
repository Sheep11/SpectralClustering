import numpy as np
from functools import reduce
import random

def euclidean_distance(x,y):
    return np.linalg.norm(x-y)

def empty_list(N):
    array=[]
    for i in range(N):
        array.append([])
    return array


def kmeans(X, n_clusters, max_iter=300):
    N,M=X.shape
    dis=np.zeros((N,n_clusters))
    begin=int(random.random()*(N-n_clusters))
    centers= [x for x in X[begin:begin+n_clusters]]

    y_pred=[0]*N

    flag=True; niter=0
    while(flag):
        if niter==max_iter:
            break;

        clusters=empty_list(n_clusters)
        flag=False; niter+=1
        for i in range(N):
            for j in range(n_clusters):
                dis[i][j]=euclidean_distance(X[i],centers[j])
            
            cluster_index=np.argsort(dis[i])[0]
            clusters[cluster_index].append(X[i])

            y_pred[i]=cluster_index
            
        for icluster in range(n_clusters):
            if len(clusters[icluster])==0:
                break
            newcenter=reduce(lambda x,y:x+y, clusters[icluster])
            newcenter/=len(clusters[icluster])

            if (centers[icluster]!=newcenter).all() :
                centers[icluster]=newcenter
                flag=True

    return y_pred