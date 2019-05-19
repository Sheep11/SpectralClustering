import numpy as np
from sklearn import datasets
import math


def load_data(path):
    data = []
    with open(path) as f:
        for line in f.readlines():
            if line == "\n":
                break

            feature = line.split(',')
            feature.pop()
            data.append(list(map(float, feature)))

    return np.array(data)


# return full permutation of ele_list
def perm(ele_list):
    if len(ele_list) <= 1:
        return [ele_list]
    rank_array = []
    for i in range(len(ele_list)):
        s = ele_list[:i] + ele_list[i + 1:]
        p = perm(s)
        for x in p:
            rank_array.append(ele_list[i:i + 1] + x)
    return rank_array


def accuracy(res):
    rank_array = perm([0, 1, 2])
    rank_array = np.array(rank_array)

    best_acc = 0
    for i in range(rank_array.shape[0]):
        acc = 0
        for x in range(150):
            if res[x] == rank_array[i][math.floor(x / 50)]:
                acc = acc + 1

        if acc >= best_acc:
            best_acc = acc

    return best_acc / 150
