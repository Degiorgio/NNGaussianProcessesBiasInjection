# Author: K.Degiorgio
import numpy as np
import scipy.io                         # some useful data input-output routines
import requests                         # for retrieving data over the web
import io
import random
from sklearn.model_selection import LeaveOneOut
import sklearn
import ipdb

random.seed(30)

def _load_data(shuffle=True):
    def shuffle_data(X, Y):
      index_shuf = list(range(len(X)))
      random.shuffle(index_shuf)
      return X[index_shuf], Y[index_shuf]
    r = requests.get('http://mlg.eng.cam.ac.uk/teaching/4f13/1920/cw/cw1a.mat')
    xs = None
    ys = None
    with io.BytesIO(r.content) as f:
        data = scipy.io.loadmat(f)
        xs,ys = data['x'], data['y']
    if shuffle:
        xs, ys = shuffle_data(xs, ys)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(xs)
    # xs = scaler.transform(xs)
    return xs, ys


def simple(percent=10):
    from collections import namedtuple
    DataSet = namedtuple('data_set', 'X Y')
    X, Y = _load_data()
    num = X.shape[0]
    i = int(num/100*percent)
    return  DataSet(X=X[i:], Y=Y[i:]), DataSet(X=X[:i], Y=Y[:i])

def make_dataset(X,Y):
    from collections import namedtuple
    DataSet = namedtuple('data_set', 'X Y')
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    return DataSet(X=X, Y=Y)

def leave_one_out():
    from collections import namedtuple
    DataSet = namedtuple('data_set', 'X Y')
    X, Y = _load_data()
    loo = leaveoneout()
    for train_index, test_index in loo.split(X):
        yield DataSet(X=X[train_index], Y=Y[train_index]), DataSet(X=X[test_index], Y=Y[text_index])
