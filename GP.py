# Author: K.Degiorgio
#
# GaussianProcess wrapper

import os
import random
import numpy as np
from sklearn.model_selection import GridSearchCV
import sklearn.gaussian_process as gp

import config


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

    def fit_parameters(self, x, y, code="code"):
        self.logger.info(f"Parameter search ...")

        path = os.path.join(config.out_dir, str(code) + ".params")
        if os.path.exists(path + ".npy"):
            self.logger.info(f"loading optimized param from: {path}.npy")
            self.best_param = np.load(path + ".npy", allow_pickle="TRUE").item()
            return

        # Grid serach using log maginal liklihood
        def score_func(clf, x, y_true):
            assert clf.kernel_ is not None
            return clf.log_marginal_likelihood()

        # Grid search, using a small hack to disable multiple folds
        ss = GridSearchCV(
            self.model,
            self.parameters_search_space,
            verbose=1,
            n_jobs=-1,
            scoring=score_func,
            cv=[(slice(None), slice(None))],
        )
        ss.fit(x, y)
        best_param = ss.best_params_
        best_score = ss.best_score_
        self.best_param = best_param
        # np.save(path, self.best_param)

        if self.verbose:
            self.logger.info(f"optimized_param: {best_param}")
            self.logger.info(f"best score: {best_score}")

    # sample from PRIOR
    def sample(self, xs, n_samples=1):
        # xs = (query points, n_features)
        y_mean, y_std = self.model.predict(xs, return_std=True)
        ys = self.model.sample_y(xs, n_samples=n_samples)
        return ys

    def predict_uniform(self, number, low=-3, high=3):
        assert self.trained
        newx = np.linspace(low, high, number)
        μ, σ = self.model.predict(newx[..., np.newaxis], return_std=True)
        return μ, σ, newx

    def predict_random(self, number, low=-3, high=3, noise=False):
        assert self.trained

        newx = [low + random.random() * (high - low) for _ in range(number)]
        newx = np.array(newx)
        μ, σ = self.model.predict(newx[..., np.newaxis], return_std=True)
        if noise:
            noise = np.random.normal(0, σ, number).reshape(-1, 1)
            μ += noise
        return μ, σ, newx

    def predict(self, x):
        # xs = (n_samples, n_features)
        assert self.trained
        return self.model.predict(x, return_std=True)
