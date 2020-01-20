# Author: K.Degiorgio
#
# KernelAwareNN

import ipdb
import numpy as np

from GP import gaussian_process
import data_loader


def _synthesise_samples_joint(n_samples, gp, x_obs, x_unk):
    """
    :param x_unk: Query points for which we have no labels.
    :param x_obs: Training data.

    """
    number_of_unkowns = x_unk.shape[0]

    # each sample represents a function
    # Note: syn = synthesized

    if x_obs is not None:
        temp = np.vstack([x_obs, x_unk])
    else:
        temp = x_unk
    # (y_samples=(function_points, functions)
    y_samples = gp.sample(temp, n_samples).squeeze()

    # split and return
    if n_samples != 1:
        number_of_observations = x_obs.shape[0]
        y_syn_obs = y_samples[:number_of_observations, :]
        y_syn_unk = y_samples[number_of_observations:, :]
        return x_obs, y_syn_obs.T, y_syn_unk.T
    else:
        return y_samples.T


class KernelAwareNN:
    def __init__(self, neural_netwok, stacked):
        """
        :param prior:  kernel prior
        :param neural_netwok: underlying NN model to use
        :param stacked: use query points during prediction
        """
        self.neural_netwok = neural_netwok
        self.stack = stacked
        return

    def train(self, n_samples, prior, train_set):
        """
        :param number_of_samples:  number of samples to synthezie from kernel
        :param prior: prior to use for GP
        :param train_set:  used during predication
        """
        self.n_samples = n_samples
        self.gp = gaussian_process(kernel=prior, verbose=False)
        self.train_set = train_set

    def predict(self, x_unk):
        """
        f :  y_syn_obs --> y_syn_ukn

        This model needs to know in advance predication points, i.e:
        training and predication are not seperable steps.

        :param x_unk:     Query points for which we have no labels.
        """

        x_obs = self.train_set.X  # x_obs, query points for which we have labels

        # step 1: synthezie labels from unfitted GP
        #
        # y_syn_*=(functions, function_points)
        x_obs, y_syn_obs, y_syn_unk = _synthesise_samples_joint(
            self.n_samples, self.gp, x_obs, x_unk
        )

        if self.stack:
            stack = np.hstack([np.tile(x_obs, self.n_samples).T, y_syn_obs])
        else:
            stack = y_syn_obs
        syntheized_data_set = data_loader.make_dataset(stack, y_syn_unk)
        # step 2
        #
        # train NN to transform a synthesied fucntion (y_synth_obs) for a set of fixed quey
        # points (x_obs), into a set of unkown points (y_synth_unk)
        #
        # In doing so we hope that the neural network learns a latent
        # representation for the kernel
        nn = self.neural_netwok()
        loss = nn.train(
            syntheized_data_set,
            None,
            epochs=4,
            width=100,
            layers=4,
            batch_size=20,
            learning_rate=0.001,
            initw=True,
        )
        # step 3. Predict
        y_obs = self.train_set.Y.T

        if self.stack:
            stack = np.hstack([x_obs.T, y_obs])
        else:
            stack = y_obs
        y_unk, _ = nn.predict(stack)
        return y_unk.T, loss


class KernelAwareNNSeq:
    def __init__(self, neural_netwok):
        """
        :param neural_netwok: underlying NN model to use
        """
        self.neural_netwok = neural_netwok
        return

    def train(self, n_samples, prior, train_set):
        self.n_samples = n_samples
        self.gp = gaussian_process(kernel=prior, verbose=False)
        self.train_set = train_set

        chunks = train_set.X.shape[0] / 10

        nn = self.neural_netwok()
        try:
            X = np.split(train_set.X, 2)
            Y = np.split(train_set.Y, 2)
        except:
            ipdb.set_trace()

        use_old = False

        datasets = []
        for x, y in zip(X, Y):
            x_1, y_1 = x[20:], y[20:]
            x_2, y_2 = x[:20], y[:20]
            _, y_syn_1, y_syn_2 = _synthesise_samples_joint(
                self.n_samples, self.gp, x_1, x_2
            )
            syntheized_data_set_1 = data_loader.make_dataset(
                y_syn_2, np.tile(y_2, n_samples).T
            )
            syntheized_data_set_2 = data_loader.make_dataset(
                y_syn_1, np.tile(y_1, n_samples).T
            )
            merged = syntheized_data_set_1.merge(syntheized_data_set_2)
            datasets.append(merged)
        dt = datasets[0].merge(datasets[1])
        loss = nn.train(
            dt,
            None,
            epochs=4,
            width=100,
            layers=4,
            batch_size=20,
            learning_rate=0.001,
            initw=True,
            use_old=use_old,
        )
        self.nn = nn

    def predict(self, x_unk):
        x_obs = self.train_set.X  # x_obs, query points for which we have labels
        y_syn_unk = _synthesise_samples_joint(1, self.gp, None, x_unk)
        # (1,20)
        y_syn_unk = y_syn_unk.reshape(1, -1)
        y_unk, _ = self.nn.predict(y_syn_unk)
        # (20,1)
        return y_unk.T, 0
