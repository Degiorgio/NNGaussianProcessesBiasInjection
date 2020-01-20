# K.Degiorgio
#
# Main Entry point


def warn(*args, **kwargs):
    pass


import os
import warnings

warnings.warn = warn

import random
from enum import Enum

random.seed(20)

import ipdb
import numpy as np
import pandas as pd
import pprint
import config

from stats import permutation_test
from sklearn.metrics import mean_squared_error
from GP import gaussian_process
from kernels import *

import data_generator
import data_loader
import vis

pp = pprint.PrettyPrinter(indent=4)
logger = config.getlogger("main")

flatten = lambda l: [item for sublist in l for item in sublist]


def NN_fit(train_set, val_set, nn=5, epochs=10, width=10, layers=2):
    from NN import NeuralNetwork

    last_error = 100
    last_predicated = None
    fnn = None
    for x in range(nn):
        nn = NeuralNetwork()
        nn.train(
            train_set,
            val_set,
            epochs=epochs,
            width=width,
            layers=layers,
            batch_size=20,
            learning_rate=0.001,
        )
        predicted_y, _ = nn.predict(train_set.X)

        logger.info(f"NN train MSE {mean_squared_error(train_set.Y, predicted_y)}")
        predicted_y, _ = nn.predict(val_set.X)
        error = mean_squared_error(val_set.Y, predicted_y)
        logger.info(f"NN dev MSE {error}")

        if error < last_error:
            last_error = error
            fnn = nn

    return fnn


def GP_fit(
    train_set,
    val_set,
    k=kernel_sigmoid_with_noise,
    plot=False,
    skip_parameter_search=True,
):
    gp = gaussian_process(kernel=k, verbose=True)
    if not skip_parameter_search:
        gp.fit_parameters(val_set.X, val_set.Y)
    gp.fit(train_set.X, train_set.Y)
    predicted_y, _ = gp.predict(train_set.X)
    logger.info(f"GP train MSE {mean_squared_error(train_set.Y, predicted_y)}")
    predicted_y, _ = gp.predict(val_set.X)
    logger.info(f"GP dev MSE {mean_squared_error(val_set.Y, predicted_y)}")
    if plot:
        # uniformly sample a bunch of points
        μ, σ, newx = gp.predict_uniform(200, low=-3, high=3)
        vis.plot_gp(
            predict_set.X,
            predict_set.Y,
            train_set.X,
            train_set.Y,
            newx,
            μ,
            σ,
            gp.model,
            axs=None,
        )
    return gp


def compute_MSE(model, predict_set):
    predicted_y, _ = model.predict(predict_set.X)
    error = mean_squared_error(predict_set.Y, predicted_y)
    return error, predicted_y


def experiment1(train_set, test_set, prior=kernel_sigmoid_with_noise):
    logger.info(f"----> Fitting GP ...")
    gp = GP_fit(train_set, train_set, plot=False, k=prior, skip_parameter_search=False)
    logger.info(f"----> Fitting vanilla NN ...")
    nn = NN_fit(train_set, train_set, nn=5, epochs=10, width=10, layers=1)
    nn_error, _ = compute_MSE(nn, test_set)
    gp_error, _ = compute_MSE(gp, test_set)
    logger.info(f"NN PREDICT MSE: {nn_error}")
    logger.info(f"GP PREDICT MSE: {gp_error}")
    vis.gp_nn_joint_plot(gp, nn, test_set, nn_error, gp_error, train_set, df=True)


def experiment2(train_set, test_set, prior=kernel_sigmoid_with_noise):
    logger.info(f"----> Fitting GP ...")
    gp = GP_fit(train_set, train_set, plot=False, k=prior, skip_parameter_search=False)
    # create data
    n_samples = 50000
    y_samples, _, x_samples = gp.predict_random(n_samples, low=0, high=1, noise=False)
    new_train_set = data_loader.make_dataset(x_samples.reshape(-1, 1), y_samples)
    y_samples, _, x_samples = gp.predict_random(100, low=0, high=1, noise=True)
    new_test_set = data_loader.make_dataset(x_samples.reshape(-1, 1), y_samples)
    # del gp
    # # refit GP to be fair ...
    # # (we are using less data cuz we run into memory issues if we don't
    # trimmed = new_train_set.trim(2).trim(2).trim(2).trim(2).trim(2).trim(2).trim(2)
    # gp = GP_fit(trimmed, trimmed, k=prior)

    logger.info(f"----> NN fitting wih more Data...")
    nn = NN_fit(new_train_set, new_train_set, nn=5, epochs=5, width=30, layers=2)
    logger.info(f"----> Computing results...")
    nn_error, _ = compute_MSE(nn, new_test_set)
    gp_error, _ = compute_MSE(gp, new_test_set)
    logger.info(f"NN PREDICT MSE: {nn_error}")
    logger.info(f"GP PREDICT MSE: {gp_error}")
    vis.gp_nn_joint_plot(gp, nn, new_test_set, nn_error, gp_error, new_train_set)


def generate_data(
    kernel=kernel_sigmoid_no_noise,
    mode=data_generator.training_testing_split.SEPERATE,
    shuffle=True,
    show_data=False,
    save=None,
    drange=[0, 1],
    n_train=80,
    n_test=20,
):
    logger.info(f"Data synthesis mode: {str(mode)}")
    k, _ = kernel()
    logger.info(f"kernel: {str(k)}")
    train_set, test_set = data_generator.create_simple_data_set(
        n_train,
        n_test,
        mode=mode,
        low=drange[0],
        high=drange[1],
        kernel=kernel,
        shuffle=shuffle,
    )
    vis.plot_data(train_set, test_set, save=save, show_data=show_data, k=k)
    return train_set, test_set


def get_data_sets_config():
    datasets = [
        dict(
            kernel=kernel_sigmoid_no_noise,
            mode=data_generator.training_testing_split.SEPERATE,
            n_train=80,
            n_test=20,
            drange=[0, 1],
        ),
        dict(
            kernel=kernel_sigmoid_no_noise3,
            mode=data_generator.training_testing_split.INTERSPREAD,
            n_train=80,
            n_test=20,
            drange=[0, 1],
        ),
        dict(
            kernel=kernel_sigmoid_no_noise4,
            mode=data_generator.training_testing_split.SEPERATE,
            n_train=80,
            n_test=20,
        ),
        dict(
            kernel=kernel_sigmoid_no_noise4,
            mode=data_generator.training_testing_split.INTERSPREAD,
            n_train=80,
            n_test=20,
        ),
        dict(
            kernel=kernel_rbf_with_no_noise,
            mode=data_generator.training_testing_split.RANDOM,
            n_train=80,
            n_test=20,
        ),
        dict(
            kernel=kernel_matern,
            mode=data_generator.training_testing_split.INTERSPREAD,
            n_train=80,
            n_test=20,
        ),
        dict(
            kernel=kernel_matern,
            mode=data_generator.training_testing_split.MIXED,
            n_train=80,
            n_test=20,
        ),
    ]
    return datasets


def get_models(n_samples=50000):
    from NN import NeuralNetwork
    from KADNN import KernelAwareNN, KernelAwareNNSeq

    models = [
        dict(
            model=KernelAwareNN(NeuralNetwork, stacked=False),
            mid="KANN",
            n_samples=n_samples,
        ),
        # dict(model=KernelAwareNN(NeuralNetwork, stacked=True), mid="KANN_ext", n_samples=n_samples),
        # dict(model=KernelAwareNNSeq(NeuralNetwork), mid="KANN_seq", n_samples=n_samples),
    ]
    return models


def compare_results(
    train_set, test_set, gp_predicted_y, nn_predicated_y, bid, show_graph=False
):
    from stats import permutation_test

    assert gp_predicted_y.shape == nn_predicated_y.shape

    err_diff = mean_squared_error(gp_predicted_y, nn_predicated_y)
    err_system_A = mean_squared_error(gp_predicted_y, test_set.Y)
    err_system_B = mean_squared_error(nn_predicated_y, test_set.Y)
    p_value = permutation_test(
        err_system_A,
        err_system_B,
        gp_predicted_y.copy(),
        nn_predicated_y.copy(),
        test_set.Y,
    )
    logger.info(f"----> Results for this benchmark")
    logger.info("MSE GP: {:.10f}".format(err_system_A))
    logger.info("MSE NN: {:.10f}".format(err_system_B))
    logger.info("MSE diff: {:.10f}".format(err_diff))
    logger.info("p-value: {:.10f}".format(p_value))

    vis.plot_gg(
        train_set,
        test_set,
        gp_predicted_y,
        "GP",
        nn_predicated_y,
        bid,
        err_diff,
        err_system_A,
        err_system_B,
        p_value,
        show=show_graph,
        save=config.get_save_file(bid + ".svg"),
    )

    return err_diff, err_system_B


def fit_gp(train_set, test_set, prior):
    logger.info(f"----> Fitting GP ...")
    gp = GP_fit(train_set, train_set, k=prior, skip_parameter_search=True)
    error, gp_predicted_y = compute_MSE(gp, test_set)
    del gp
    return error, gp_predicted_y


def fit_nn(model, n_samples, prior, train_set, test_set, gp_predicted_y, bid):
    logger.info(f"----> Fitting Kernel Aware NN...")
    logger.info(f"Training NN with {n_samples}")
    model.train(n_samples, prior, train_set)
    nn_predicated_y, train_loss = model.predict(test_set.X)
    return (
        train_loss,
        compare_results(train_set, test_set, gp_predicted_y, nn_predicated_y, bid),
    )


def print_data_frame(df):
    df = df.round(4)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    logger.info("\n" + df.to_string().replace("\n", "\n\t") + "\n")


def fit_gp_and_models(models, train_set, test_set, i, results, prior):
    gp_MSE, gp_predicted_y = fit_gp(train_set, test_set, prior)
    results.at[i, "GP"] = gp_MSE
    for model in models:
        train_loss, (err_diff, err_system_nn) = fit_nn(
            model["model"],
            model["n_samples"],
            prior,
            train_set,
            test_set,
            gp_predicted_y,
            bid="Model: " + model["mid"],
        )
        results.at[i, model["mid"]] = err_system_nn
        results.at[i, model["mid"] + " MSE diff"] = err_diff
        results.at[i, model["mid"] + " train loss"] = train_loss


def main(models, datasets, fake_pior=None):
    model_ids = flatten(
        [[x["mid"], x["mid"] + " train loss", x["mid"] + " MSE diff"] for x in models]
    )
    index = list(range(len(datasets)))
    results = pd.DataFrame(columns=["GP"] + model_ids, index=index)
    for i, dataset in enumerate(datasets):
        if fake_pior is not None:
            prior = fake_pior
        else:
            prior = dataset["kernel"]
        logger.info(f"===============")
        kernel, _ = prior()
        logger.info(
            f"training on benchmark : {str(i)},  kernel: {str(kernel)},  genertation"
            f" mode: {dataset['mode']}"
        )
        train_set, test_set = generate_data(
            **dataset, save=config.get_save_file(str(i) + "_data.svg"), show_data=False
        )
        fit_gp_and_models(models, train_set, test_set, i, results, prior)
    results.loc["Average"] = results.mean()
    results.loc["Variance"] = results.var()
    return results


def go_abalation():
    models = get_models(100)
    model_ids = [x["mid"] for x in models]

    logger.info("-----------------------> Ablation 1: Small Number of Samples")
    ab1 = go_cross_validation(models, save=False)

    logger.info("-----------------------> Ablation 2: Wrong Prior")
    models = get_models()
    ab2 = go_cross_validation(
        models, kernel=kernel_matern, fake_pior=kernel_dot_product, save=False
    )
    logger.info("-----------------------> Ablation 3: Noisy Kernel")
    ab3 = go_cross_validation(models, kernel=kernel_rbf_with_noise, save=False)

    ab1.to_pickle(os.path.join(config.out_dir, "abs_small_number_of_samples.pkl"))
    ab2.to_pickle(os.path.join(config.out_dir, "abs_wrong_prior.pkl"))
    ab3.to_pickle(os.path.join(config.out_dir, "abs_noisy_kernel.pkl"))


def go_normal():
    models = get_models()
    datasets = get_data_sets_config()
    results = main(models, datasets)
    logger.info(f"----> All results")
    print_data_frame(results.T)
    logger.info(f"----> Datasets:")
    for i, dataset in enumerate(datasets):
        prior = dataset["kernel"]
        kernel, _ = prior()
        logger.info(f"({str(i)} --> kernel: {str(kernel)}, mode: {dataset['mode']}")

    results.to_pickle(os.path.join(config.out_dir, "normal.pkl"))


def go_cross_validation(
    models, kernel=kernel_sigmoid_no_noise3, fake_pior=None, save=True
):
    def get_crossvalidation():
        train_set, test_set = data_generator.create_simple_data_set(
            50,
            50,
            mode=data_generator.training_testing_split.INTERSPREAD,
            low=0,
            high=1,
            kernel=kernel,
            shuffle=True,
        )
        combined_data_set = train_set.merge(test_set)
        return combined_data_set.X, combined_data_set.Y

    X, Y = get_crossvalidation()
    from sklearn.model_selection import KFold

    lpo = KFold(n_splits=100 // 20)
    index = list(range(lpo.get_n_splits(X, Y)))
    model_ids = flatten(
        [[x["mid"], x["mid"] + " train loss", x["mid"] + " MSE diff"] for x in models]
    )
    results = pd.DataFrame(columns=["GP"] + model_ids, index=index)
    for i, (train_index, test_index) in enumerate(lpo.split(X)):
        logger.info(f"------->Fold {i}/5")
        logger.info(
            f"Training set size: {len(train_index)} Testing set size: {len(test_index)}"
        )
        train_set = data_loader.make_dataset(X[train_index], Y[train_index])
        test_set = data_loader.make_dataset(X[test_index], Y[test_index])
        if fake_pior is not None:
            prior = fake_pior
        else:
            prior = kernel
        fit_gp_and_models(models, train_set, test_set, i, results, prior)
    results.loc["Average"] = results.mean()
    results.loc["Variance"] = results.var()
    results = results.round(5)
    print_data_frame(results)
    if save:
        results.to_pickle(os.path.join(config.out_dir, "cross_validation_results.pkl"))
    return results


go_cross_validation(get_models())
print(config.run_id)
