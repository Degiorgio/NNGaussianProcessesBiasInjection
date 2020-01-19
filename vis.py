import ipdb
import numpy as np                      # matlab-style matrix and vector manipulation
import matplotlib.pyplot as plt         # matlab-style plotting
import matplotlib                       # for more control over plotting
import math

def plot_gp(p_x, p_y, xs, ys, newx, μ, σ, model, axs=None):
    show = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show = True
    title_str =f"log_marginal_likelihood: {model.log_marginal_likelihood()}"
    axs.fill_between(newx, μ.squeeze()-2*σ, μ.squeeze()+2*σ, alpha=.2)
    axs.plot(newx, μ.squeeze())
    axs.scatter(xs, ys, color='black', label=str("observed values"))
    axs.scatter(p_x, p_y, color='red', label=str("new values"))
    axs.set_xlabel("t")
    axs.set_ylabel("y")
    axs.legend()
    axs.set_title(title_str)
    if show:
        plt.show(block=True)

def gp_nn_joint_plot(gp, nn, testing_set, nn_error, gp_error,
                     training_set=None, axs=None, df=False):
    show = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show = True
    if not df:
        μ, σ, newx= gp.predict_uniform(200,
                                       low=math.floor(testing_set.X.min())-1,
                                       high=math.ceil(testing_set.X.max())+1)
    else:
        μ, σ, newx= gp.predict_uniform(200,-3, 3)

    axs.fill_between(newx, μ.squeeze()-2*σ, μ.squeeze()+2*σ, alpha=.2)
    axs.plot(newx, μ.squeeze(),
             label=f"Gaussian Process, MSE: {round(gp_error,5)}")
    axs.set_xlabel("t")
    axs.set_ylabel("y")
    axs.scatter(testing_set.X.squeeze(), testing_set.Y.squeeze(),
                color='blue', label=str("un-observed values"))

    if training_set is not None:
        axs.scatter(training_set.X.squeeze(), training_set.Y.squeeze(),
                    color='black', label=str("training values"))

    predicted_y, _ = nn.predict(newx[...,np.newaxis])
    axs.plot(newx, predicted_y.squeeze(),
             label=f"Neural Network, MSE: {round(nn_error, 5)}", c='r')
    axs.legend()
    if show:
        plt.show(block=True)


def plot_nn(nn, train_set, axs=None):
    show = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show = True

    axs.set_xlabel("t")
    axs.set_ylabel("y")
    axs.scatter(train_set.X.squeeze(),
                train_set.Y.squeeze(),
                color='black', label=str("observed values"))
    axs.legend()

    y = nn.predict(train_set.X)
    axs.scatter(train_set.X.squeeze(), y.squeeze(), label="NN", c='r')

    if show:
        plt.show(block=True)


def quick_scatter(x, y):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x.squeeze(), y.squeeze())
    plt.show(block=True)
