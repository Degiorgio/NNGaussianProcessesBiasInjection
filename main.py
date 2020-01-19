import random
random.seed(30)

import ipdb
import numpy as np
import pandas
import pprint
import config

from sklearn.metrics import mean_squared_error
from GP import gaussian_process, kernel_Sigmoid_withNoise
import data_loader
import vis

pp = pprint.PrettyPrinter(indent=4)
logger = config.getlogger("main")



def NN_fit(train_set, val_set, nn=5, epochs=10, width=10,layers=2):
    from NN import Model
    last_error=100
    last_predicated = None
    fnn = None
    for x in range(nn):
        nn = Model()
        nn.train(train_set, val_set,
                 epochs=epochs,
                 width=width,
                 layers=layers,
                 batch_size=20,
                 learning_rate=0.1)
        predicted_y, _ = nn.predict(train_set.X)
        logger.info(f"NN train MSE {mean_squared_error(train_set.Y, predicted_y)}")
        predicted_y, _ = nn.predict(val_set.X)
        error = mean_squared_error(val_set.Y, predicted_y)
        logger.info(f"NN dev MSE {error}")
        if error < last_error:
            last_error = error
            fnn = nn
    return fnn

def GP_fit(train_set, val_set, plot=False):
    gp = gaussian_process(kernel=kernel_Sigmoid_withNoise, verbose=True)
    gp.fit_parameters(val_set.X, val_set.Y)
    gp.fit(train_set.X, train_set.Y)

    predicted_y, _ = gp.predict(train_set.X)
    logger.info(f"GP train MSE {mean_squared_error(train_set.Y, predicted_y)}")
    predicted_y, _ = gp.predict(val_set.X)
    logger.info(f"GP dev MSE {mean_squared_error(val_set.Y, predicted_y)}")

    if plot:
        # uniformly sample a bunch of points 
        μ , σ, newx= gp.predict_uniform(200, low=-3, high=3)
        vis.plot_gp(predict_set.X,
                    predict_set.Y,
                    train_set.X,
                    train_set.Y,
                    newx, μ, σ, gp.model,
                    axs=None)
    return gp


def experiment1(train_set, val_set, predict_set, nn=5, epochs=10, width=10,layers=2):
    return


def compute_MSE(model, predict_set):
    predicted_y, _ = model.predict(predict_set.X)
    error = mean_squared_error(predict_set.Y, predicted_y)
    return error

def experiment1():
    train_set, test_set = data_loader.simple(percent=10)
    logger.info(f"----> Fitting GP ...")
    gp = GP_fit(train_set, train_set, plot=False)
    logger.info(f"----> Fitting vanilla NN ...")
    nn = NN_fit(train_set, train_set, nn=5, epochs=5, width=5,layers=2)
    nn_error = compute_MSE(nn, test_set)
    gp_error = compute_MSE(gp, test_set)
    logger.info(f"NN PREDICT MSE: {nn_error}")
    logger.info(f"GP PREDICT MSE: {gp_error}")
    vis.gp_nn_joint_plot(gp, nn, test_set, nn_error, gp_error, train_set,df=True)

def experiment2():
    train_set, test_set = data_loader.simple()
    logger.info(f"----> Fitting GP ...")
    gp = GP_fit(train_set, train_set, plot=False)
    # create data
    n_samples=40000
    y_samples, _ ,x_samples = gp.predict_random(n_samples, low=0, high=1, noise=False)
    new_train_set = data_loader.make_dataset(x_samples.reshape(-1,1), y_samples)
    y_samples, _ ,x_samples = gp.predict_random(100, low=0, high=1, noise=True)
    new_test_set = data_loader.make_dataset(x_samples.reshape(-1,1), y_samples)
    #vis.quick_scatter(x_samples, y_samples)

    logger.info(f"----> NN fitting wih more Data...")
    nn = NN_fit(new_train_set, new_train_set,
                   nn=5, epochs=5, width=10,layers=2)
    logger.info(f"----> Computing results...")
    nn_error = compute_MSE(nn, new_test_set)
    gp_error = compute_MSE(gp, new_test_set)
    logger.info(f"NN PREDICT MSE: {nn_error}")
    logger.info(f"GP PREDICT MSE: {gp_error}")
    vis.gp_nn_joint_plot(gp, nn, new_test_set, nn_error, gp_error, new_train_set)


def experiment3():
    # experiment 3
    return

experiment2()
