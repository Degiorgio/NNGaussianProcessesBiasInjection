# Author: K.Degiorgio

import numpy as np
import pandas
import ipdb
import pprint
import config
from joblib import dump, load
import sklearn.gaussian_process as gp

def kernel_RBF_withNoise(n=10):
    λ=0.2; v=1; σ=9.0
    K = v**2 * gp.kernels.RBF(length_scale=λ)
    K += kgp.kernels.WhiteKernel(noise_level=σ)
    parameters = {
        'kernel__k1__k2__length_scale':np.linspace(0.1, 10, num=n),
        'kernel__k2__noise_level':np.linspace(0.1, 10, num=n),
        'kernel__k1__k1__constant_value':np.linspace(0.1, 10, num=n)
    }
    return K, parameters


def kernel_Sigmoid_withNoise(n=10):
    σ = 8; v = 0.1; λ = 0.1; T = 0.1
    K = v*gp.kernels.ExpSineSquared(length_scale=λ, periodicity=T)
    K = K+gp.kernels.WhiteKernel(noise_level=σ)
    parameters = {
        'kernel__k1__k2__length_scale':np.linspace(0.1, 10, num=n),
        'kernel__k2__noise_level':np.linspace(0.1, 10, num=n),
        'kernel__k1__k1__constant_value':np.linspace(0.1, 10, num=n)
    }
    return K, parameters


class gaussian_process:
    def __init__(self, kernel, verbose=False):
        self.K, self.parameters_search_space = kernel()
        self.model = gp.GaussianProcessRegressor(kernel=self.K)
        self.verbose = verbose
        self.logger = config.getlogger("GP")
        self.best_param = None
        self.trained = False

    def fit(self, x, y):
        self.logger.info(f"Training ...")
        if self.best_param is not None:
            self.logger.info("Using pre-set parameters")
            self.model.set_params(**self.best_param)
        else:
            self.logger.info(f"No set hyperparameters using defaults")
        self.model.fit(x, y)
        self.trained = True


    def fit_parameters(self, x, y, code = "code"):
        self.logger.info(f"Parameter search ...")
        import os
        from sklearn.model_selection import GridSearchCV
        path = os.path.join(config.out_dir, str(code) + ".params")
        if os.path.exists(path + ".npy"):
            self.logger.info(f"loading optimized param from: {path}.npy")
            self.best_param  = np.load(path + ".npy", allow_pickle='TRUE').item()
            return

        # Grid serach using log maginal liklihood
        def score_func(clf, x, y_true):
            assert(clf.kernel_ != None)
            return clf.log_marginal_likelihood()

        # Grid search, using a small hack to disable multiple folds
        ss = GridSearchCV(self.model,
                          self.parameters_search_space,
                          verbose=1,
                          n_jobs=-1,
                          scoring=score_func,
                          cv=[(slice(None), slice(None))])
        ss.fit(x, y)
        best_param = ss.best_params_
        best_score = ss.best_score_
        self.best_param = best_param
        np.save(path, self.best_param)

        if self.verbose:
            self.logger.info(f"optimized_param: {best_param}")
            self.logger.info(f"best score: {best_score}")

    # sample from PRIOR
    def sample(self, xs):
        # xs = (n_samples, n_features)
        ys = self.model.sample_y(xs)
        return ys

    # sample from PRIOR
    def sample_uniform(self, number, low=-3, high=3):
        # xs = (n_samples, n_features)
        newx= np.linspace(low, high, number)
        ys = self.model.sample_y(newx[..., np.newaxis])
        return ys, newx

    def predict_uniform(self, number, low=-3, high=3):
        assert self.trained
        newx= np.linspace(low, high, number)
        μ, σ  = self.model.predict(newx[..., np.newaxis], return_std=True)
        return μ, σ , newx

    def predict_random(self, number, low=-3, high=3, noise=False):
        assert self.trained
        import random
        newx = [low + random.random() * (high - low) for _ in range(number)]
        newx = np.array(newx)
        μ, σ  = self.model.predict(newx[...,np.newaxis], return_std=True)
        if noise:
            noise = np.random.normal(0,σ,number).reshape(-1,1)
            μ += noise
        return μ, σ , newx

    def predict(self, x):
        #xs = (n_samples, n_features) 
        assert self.trained
        return self.model.predict(x, return_std=True)
