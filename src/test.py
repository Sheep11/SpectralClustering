import numpy as np
import matplotlib.pyplot as plt
from src.kmeans import kmeans
from src.util import load_data, accuracy
from src.model import affinity_matrix, laplacian, spectral_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys


def test_gaussian():
    plt.figure('iris clustering')
    X = load_data('data/iris.data')
    iris = PCA(n_components=2).fit_transform(X)

    acc1 = []
    acc2 = []
    sig2 = 0.5
    for i in range(39):
        A = affinity_matrix(X, 'gaussian', 1, sig2)
        L = laplacian(A, std=True)
        spectral_ft = spectral_data(L, 3)
        #y_pred = kmeans(spectral_ft, 3)
        y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
        acc = accuracy(y_pred)
        acc1.append(acc)

        L = laplacian(A, std=False)
        spectral_ft = spectral_data(L, 3)
        #y_pred = kmeans(spectral_ft, 3)
        y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
        acc = accuracy(y_pred)
        acc2.append(acc)

        sig2 += 0.5

    plt.subplot(2, 1, 1)
    plt.plot([x / 10 for x in range(5, 200, 5)], acc1)
    plt.xlabel('sqrt(sig)')
    plt.ylabel('accuracy rate')
    plt.title('Gaussian with std Lapalacian')

    plt.subplot(2, 1, 2)
    plt.plot([x / 10 for x in range(5, 200, 5)], acc2)
    plt.xlabel('sqrt(sig)')
    plt.ylabel('accuracy rate')
    plt.title('Gaussian with Lapalacian')
    plt.tight_layout()
    plt.show()


def test_methods():
    plt.figure('iris clustering')
    X = load_data('data/iris.data')
    iris = PCA(n_components=2).fit_transform(X)

    A = affinity_matrix(X, 'gaussian', 1, 8)
    L = laplacian(A, std=False)
    spectral_ft = spectral_data(L, 3)
    #y_pred = kmeans(spectral_ft, 3)
    y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
    acc = accuracy(y_pred)
    plt.subplot(3, 2, 1)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Gaussian with Lapalacian:' + str(acc)[:5])

    A = affinity_matrix(X, 'gaussian', 1, 0.5)
    L = laplacian(A, std=True)
    spectral_ft = spectral_data(L, 3)
    #y_pred = kmeans(spectral_ft, 3)
    y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
    acc = accuracy(y_pred)
    plt.subplot(3, 2, 2)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Gaussian with std Lapalacian' + str(acc)[:5])

    A = affinity_matrix(X, 'eculid')
    L = laplacian(A, std=False)
    spectral_ft = spectral_data(L, 3, min=False)
    #y_pred = kmeans(spectral_ft, 3)
    y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
    acc = accuracy(y_pred)
    plt.subplot(3, 2, 3)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Eculid with Lapalacian' + str(acc)[:5])

    A = affinity_matrix(X, 'eculid')
    L = laplacian(A, std=True)
    spectral_ft = spectral_data(L, 3, min=False)
    #y_pred = kmeans(spectral_ft, 3)
    y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
    acc = accuracy(y_pred)
    plt.subplot(3, 2, 4)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Eculid with std Lapalacian' + str(acc)[:5])

    A = affinity_matrix(X, 'cosine')
    L = laplacian(A, std=False)
    spectral_ft = spectral_data(L, 3)
    #y_pred = kmeans(spectral_ft, 3)
    y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
    acc = accuracy(y_pred)
    plt.subplot(3, 2, 5)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Cosine with Lapalacian' + str(acc)[:5])

    A = affinity_matrix(X, 'cosine')
    L = laplacian(A, std=True)
    spectral_ft = spectral_data(L, 3)
    y_pred = KMeans(n_clusters=3).fit_predict(spectral_ft)
    acc = accuracy(y_pred)
    plt.subplot(3, 2, 6)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Cosine with std Lapalacian' + str(acc)[:5])

    plt.tight_layout()
    plt.show()


def test_kmeans():
    plt.figure('iris clustering')
    X = load_data('data/iris.data')
    iris = PCA(n_components=2).fit_transform(X)

    y_pred = kmeans(X, 3)
    acc = accuracy(y_pred)
    plt.subplot(2, 1, 1)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('My Kmeans:' + str(acc)[:5])

    y_pred = KMeans(n_clusters=3).fit_predict(X)
    acc = accuracy(y_pred)
    plt.subplot(2, 1, 2)
    plt.scatter(iris[:, 0], iris[:, 1], c=y_pred)
    plt.title('Sklearn Kmeans:' + str(acc)[:5])

    plt.tight_layout()
    plt.show()
