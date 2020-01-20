# Author: K.Degiorgio
#
# Displaying graphs, this file is a mess, proceed with caution.

import ipdb
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style

style.use("seaborn-paper")  # sets the size of the charts
# style.use('ggplot')


def plot_gp(p_x, p_y, xs, ys, newx, μ, σ, model, axs=None):
    show = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show = True
    title_str = f"log_marginal_likelihood: {model.log_marginal_likelihood()}"
    axs.fill_between(newx, μ.squeeze() - 2 * σ, μ.squeeze() + 2 * σ, alpha=0.2)
    axs.plot(newx, μ.squeeze())
    axs.scatter(xs, ys, color="black", label=str("observed values"))
    axs.scatter(p_x, p_y, color="red", label=str("new values"))
    axs.set_xlabel("t")
    axs.set_ylabel("y")
    axs.legend()
    axs.set_title(title_str)
    if show:
        plt.show(block=True)


def gp_nn_joint_plot(
    gp, nn, testing_set, nn_error, gp_error, training_set=None, axs=None, df=False
):
    show = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show = True
    if not df:
        μ, σ, newx = gp.predict_uniform(
            200,
            low=math.floor(testing_set.X.min()) - 1,
            high=math.ceil(testing_set.X.max()) + 1,
        )
    else:
        μ, σ, newx = gp.predict_uniform(200, -3, 3)

    axs.fill_between(newx, μ.squeeze() - 2 * σ, μ.squeeze() + 2 * σ, alpha=0.2)
    axs.plot(newx, μ.squeeze(), label=f"Gaussian Process, MSE: {round(gp_error,5)}")
    axs.scatter(
        testing_set.X.squeeze(),
        testing_set.Y.squeeze(),
        color="blue",
        label=str("un-observed values"),
    )

    if training_set is not None:
        axs.scatter(
            training_set.X.squeeze(),
            training_set.Y.squeeze(),
            color="black",
            label=str("training values"),
        )

    predicted_y, _ = nn.predict(newx[..., np.newaxis])
    axs.plot(
        newx,
        predicted_y.squeeze(),
        label=f"Neural Network, MSE: {round(nn_error, 5)}",
        c="r",
    )
    axs.legend()

    fig.tight_layout()
    plt.savefig("expr1.svg", format="svg", dpi=1200)
    if show:
        plt.show(block=True)


def plot_nn(nn, train_set, axs=None):
    show = False
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        show = True

    axs.set_xlabel("t")
    axs.set_ylabel("y")
    axs.scatter(
        train_set.X.squeeze(),
        train_set.Y.squeeze(),
        color="black",
        label=str("observed values"),
    )
    axs.legend()

    y = nn.predict(train_set.X)
    axs.scatter(train_set.X.squeeze(), y.squeeze(), label="NN", c="r")

    if show:
        plt.show(block=True)


def quick_scatter(x, y):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x.squeeze(), y.squeeze())
    plt.show(block=True)


def quick_line_plot(X, Y, y_low=-3, y_high=3, x_low=0, x_high=5):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.plot(X, Y, lw=1)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.show(block=True)


def plot_gg(
    training_set,
    testing_set,
    system_A,
    system_A_name,
    system_B,
    system_B_name,
    err_diff,
    err_system_A,
    err_system_B,
    p_value,
    show,
    save,
):

    high = (
        testing_set.Y.max()
        if testing_set.Y.max() > training_set.Y.max()
        else training_set.Y.max()
    )
    high += 1
    low = (
        testing_set.Y.min()
        if testing_set.Y.min() < training_set.Y.min()
        else training_set.Y.min()
    )
    low -= 1

    dataset = training_set.merge(testing_set)
    dataset.order()
    # -----
    fig2, axs = plt.subplots(1, 1)
    # spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig2)

    # axs = fig2.add_subplot(spec2[0, 0])
    # axs.set_title(system_A_name + " (MSE: " + str(round(err_system_A, 5)) + ")",loc='left')
    # axs.scatter(training_set.X, training_set.Y,
    #             color="black",
    #             label="observed values")
    # axs.scatter(testing_set.X, system_A,
    #             color="blue",
    #             label=system_A_name)
    # axs.set_ylim(low, high)
    # axs.plot(dataset.X.squeeze(), dataset.Y.squeeze(), color="red", label="ground truth")
    # -----
    axs.set_title(system_B_name, loc="left")
    axs.scatter(training_set.X, training_set.Y, color="black", label="Observed values")
    axs.scatter(testing_set.X, system_B, color="blue", label="Predicted values")
    axs.plot(
        dataset.X.squeeze(),
        dataset.Y.squeeze(),
        color="red",
        label="Interpolated Ground-truth",
    )
    axs.set_ylim(low, high)
    # -----
    # axs = fig2.add_subplot(spec2[2, 0])
    # axs.set_title("Ground Truth",loc='left')
    # axs.scatter(training_set.X.squeeze(),
    #             training_set.Y.squeeze(),
    #             color="black",
    #             label="observed values")
    # axs.scatter(testing_set.X.squeeze(),
    #             testing_set.Y.squeeze(),
    #             color="blue",
    #             label="predicted values")
    # axs.set_ylim(low, high)
    # dataset = training_set.merge(testing_set)
    # dataset.order()
    # axs.plot(dataset.X.squeeze(), dataset.Y.squeeze(), color="red")

    import itertools

    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    # handles, labels = axs.get_legend_handles_labels()
    # fig2.legend(flip(handles, 3), flip(labels,3), loc='lower center', ncol=3)

    axs.legend()

    if save != None:
        fig2.tight_layout()
        plt.savefig(save, format="svg", dpi=1200)
    if show:
        plt.show(block=True)


def plot_data(training_set, testing_set, save, show_data, k):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(
        training_set.X.squeeze(),
        training_set.Y.squeeze(),
        color="black",
        label="training set",
    )
    axs.scatter(
        testing_set.X.squeeze(),
        testing_set.Y.squeeze(),
        color="blue",
        label="testing set",
    )
    dataset = training_set.merge(testing_set)
    dataset.order()
    axs.plot(dataset.X.squeeze(), dataset.Y.squeeze(), color="red")
    axs.legend()
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_title(str(k))
    if save != None:
        fig.tight_layout()
        plt.savefig(save, format="svg", dpi=1200)
    if show_data:
        plt.show(block=True)
