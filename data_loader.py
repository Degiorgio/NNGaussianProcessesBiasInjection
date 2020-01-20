# Author: K.Degiorgio
#
# utilities to load data

import ipdb
import io
import random
import numpy as np
import scipy.io
import requests
import sklearn
from config import device

random.seed(30)


class DataSet:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def T(self):
        self.X = X.T
        self.Y = Y.T

    def shape(self):
        print(self.X.shape)
        print(self.Y.shape)

    def normalize_independent(self):
        self.X = sklearn.preprocessing.normalize(self.X)

    def normalize_dependent(self):
        self.Y = sklearn.preprocessing.normalize(self.Y)

    def create_tensor(self):
        import torch

        tensor_x = torch.Tensor(self.X).to(device)
        tensor_y = torch.Tensor(self.Y).to(device)
        return tensor_x, tensor_y

    def trim(self, n):
        x = np.delete(self.X, np.arange(0, self.X.size, n)).reshape(-1, 1)
        y = np.delete(self.Y, np.arange(0, self.Y.size, n)).reshape(-1, 1)
        return DataSet(X=x, Y=y)

    def merge(self, other):
        x = np.vstack([self.X, other.X])
        y = np.vstack([self.Y, other.Y])
        return DataSet(X=x, Y=y)

    def order(self):
        p = self.X.squeeze().argsort()
        self.X = self.X[p]
        self.Y = self.Y[p]
        return

    def shuffle(self):
        p = np.random.permutation(len(self.X))
        self.X = self.X[p]
        self.Y = self.Y[p]


def _load_data(shuffle=True):
    def shuffle_data(X, Y):
        index_shuf = list(range(len(X)))
        random.shuffle(index_shuf)
        return X[index_shuf], Y[index_shuf]

    r = requests.get("http://mlg.eng.cam.ac.uk/teaching/4f13/1920/cw/cw1a.mat")
    xs = None
    ys = None
    with io.BytesIO(r.content) as f:
        data = scipy.io.loadmat(f)
        xs, ys = data["x"], data["y"]
    if shuffle:
        xs, ys = shuffle_data(xs, ys)
    return xs, ys


def simple(percent=10):
    X, Y = _load_data()
    num = X.shape[0]
    i = int(num / 100 * percent)
    return DataSet(X=X[i:], Y=Y[i:]), DataSet(X=X[:i], Y=Y[:i])


def make_dataset(X, Y):
    return DataSet(X=X, Y=Y)
