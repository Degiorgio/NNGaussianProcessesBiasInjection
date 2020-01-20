# Author: K.Degiorgio
#
# data generation

import ipdb

import numpy as np
import random
from enum import Enum
from GP import gaussian_process
from kernels import *
import data_loader
import math


class training_testing_split(Enum):
    SEPERATE = 1
    INTERSPREAD = 2
    RANDOM = 3
    MIXED = 4
    SEPERATE_LONG = 5


def __seperate(n_training_points, n_testing_points, low, high):
    mid = (low + high) / 2
    x_training = np.linspace(low, mid, num=n_training_points).reshape(-1, 1)
    x_testing = np.linspace(mid, high, num=n_testing_points).reshape(-1, 1)
    return x_training, x_testing


def __seperate_long(n_training_points, n_testing_points, low, high):
    x_training = np.linspace(low, high, num=n_training_points).reshape(-1, 1)
    x_testing = np.linspace(high, high * 10, num=n_testing_points).reshape(-1, 1)
    return x_training, x_testing


def __interspread(n_training_points, n_testing_points, low, high):
    temp = list(np.linspace(low, high, num=(n_training_points + n_testing_points)))
    x_training = random.sample(temp, n_training_points)
    x_testing = list(set(temp) - set(x_training))
    x_testing = np.array(x_testing).reshape(-1, 1)
    x_training = np.array(x_training).reshape(-1, 1)
    return x_training, x_testing


def __random(n_training_points, n_testing_points, low, high):
    mid = (low + high) / 2
    scale = (math.fabs(low) + math.fabs(high)) / 2
    x_training = np.random.normal(mid, scale, n_training_points).reshape(-1, 1)
    x_testing = np.random.normal(mid, scale, n_testing_points).reshape(-1, 1)
    return x_training, x_testing


def create_simple_data_set(
    n_training_points,
    n_testing_points,
    low=0,
    high=3,
    mode=training_testing_split.SEPERATE,
    kernel=kernel_matern,
    shuffle=True,
):
    """
    This function uses GP to generate data
    """
    gp = gaussian_process(kernel=kernel, verbose=True)

    mid = (low + high) / 2

    if mode == training_testing_split.SEPERATE_LONG:
        x_training, x_testing = __seperate_long(
            n_training_points, n_testing_points, low, high
        )
    elif mode == training_testing_split.SEPERATE:
        x_training, x_testing = __seperate(
            n_training_points, n_testing_points, low, high
        )
    elif mode == training_testing_split.INTERSPREAD:
        x_training, x_testing = __interspread(
            n_training_points, n_testing_points, low, high
        )
    elif mode == training_testing_split.RANDOM:
        x_training, x_testing = __random(n_training_points, n_testing_points, low, high)
    elif mode == training_testing_split.MIXED:

        def r(z):
            dist = np.random.randint(low=1, high=100, size=4)
            λ = lambda x: x / dist.sum()
            vfunc = np.vectorize(λ)
            dist = vfunc(dist)
            return (z * dist).round().astype(int)

        training_dist = r(n_training_points)
        testing_dist = r(n_testing_points)
        x1, x2 = __random(training_dist[0], testing_dist[0], low, high)
        x11, x22 = __interspread(training_dist[1], testing_dist[1], low, high)
        x111, x222 = __interspread(training_dist[2], testing_dist[2], low, high)
        x1111, x2222 = __seperate(training_dist[3], testing_dist[3], low, high)
        x_training = np.vstack([x1, x11, x111, x1111])
        x_testing = np.vstack([x2, x22, x222, x222])

    y_samples = gp.sample(np.vstack([x_training, x_testing]), 1).squeeze()
    y_training = y_samples[: len(x_training)].reshape(-1, 1)
    y_testing = y_samples[len(x_training) :].reshape(-1, 1)
    training_data_set = data_loader.DataSet(X=x_training, Y=y_training)
    testing_data_set = data_loader.DataSet(X=x_testing, Y=y_testing)

    if shuffle:
        training_data_set.shuffle()
        testing_data_set.shuffle()

    return training_data_set, testing_data_set
