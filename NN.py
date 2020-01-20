# Author: K.Degiorgio

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import config
from config import device
import data_loader


class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=50, output_size=1):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        # input
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )
        self.apply(init_weights)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class _Network(nn.Module):
    def __init__(self, dim_in, dim_out, layers, width, initw, activation=nn.Sigmoid):
        super().__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        modules = []
        # input
        modules.append(nn.Linear(dim_in, width))
        # hidden
        for x in range(layers):
            modules.append(nn.Linear(width, width))
            modules.append(activation())
        # output
        modules.append(nn.Linear(width, dim_out))

        self.net = nn.Sequential(*modules)
        if initw:
            self.net.apply(init_weights)

    def forward(self, x):
        return self.net.forward(x)


class NeuralNetwork:
    def __init__(self, *args, **kwargs):
        self.logger = config.getlogger("NN")
        self.loss_func = torch.nn.MSELoss()
        self.model = None
        return super().__init__(*args, **kwargs)

    def evaluate(self, testing_set, model):
        nexamples = testing_set.X.shape[0]
        tensor_x, tensor_y = testing_set.create_tensor()
        my_dataset = data.TensorDataset(tensor_x, tensor_y)
        my_dataloader = data.DataLoader(my_dataset, batch_size=nexamples)
        losses = 0.0
        it = 0
        for i, (x, y) in enumerate(iter(my_dataloader)):
            prediction = model(x)
            loss = self.loss_func(prediction, y)
            losses += loss.item()
            it += 1
        return losses / it

    def train(
        self,
        training_set,
        validation_set,
        epochs=10,
        width=10,
        batch_size=20,
        layers=2,
        learning_rate=0.1,
        initw=False,
        LSTM=False,
        use_old=False,
    ):
        # training_set.X = (number_of_samplesxdim_in)
        # training_set.Y = (number_of_samplesxdim_out)
        nexamples = training_set.X.shape[0]
        dim_in = training_set.X.shape[1]
        dim_out = training_set.Y.shape[1]

        if LSTM:
            model = LSTM(dim_in, 50, dim_out)
        else:
            if use_old and self.model != None:
                self.logger.info(f"Using old....")
                model = self.model
            else:
                model = _Network(dim_in, dim_out, layers, width, initw=initw)

        model.to(device)

        # create model
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.train()

        self.logger.info(f"dimensionality of each input sample: {dim_in}")
        self.logger.info(f"dimensionality of each ouput sample: {dim_out}")
        self.logger.info(f"number of training samples: {nexamples}")
        self.logger.info(f"Layers: {layers}")
        self.logger.info(f"Width: {width}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Learning Rate: {learning_rate}")
        self.logger.info(f"number of trainable parameters: {total_parameters}")
        self.logger.info(f"init weighs: {initw}")

        # batchify data
        tensor_x, tensor_y = training_set.create_tensor()
        my_dataset = data.TensorDataset(tensor_x, tensor_y)
        my_dataloader = data.DataLoader(my_dataset, batch_size=batch_size)

        # train using ADAM
        optimizer = optim.Adam(parameters, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5000, gamma=0.1
        )
        last_running_loss = None
        pat, pat2 = 0, 0
        epoch_loss = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (x, y) in enumerate(iter(my_dataloader)):
                optimizer.zero_grad()
                # forward pass
                prediction = model(x)
                # compute loss
                loss = self.loss_func(prediction, y)
                running_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters, 3.0)
                # backprop
                loss.backward()
                optimizer.step()
                # update learning rate
                scheduler.step()

            # compute progress
            epoch_loss = running_loss / batch_size
            self.logger.info(f"epoch {epoch} train loss {epoch_loss}")
            last_running_loss = running_loss / batch_size

            if round(last_running_loss, 5) == round(running_loss, 5):
                pat += 1
            elif last_running_loss > running_loss:
                pat2 += 1

            if pat >= 3:
                self.logger.info(f"losing ground, early stopping")
                break
            elif pat2 >= 2:
                self.logger.info(f"stuck, early stopping")
                break

            # check dev loss
            if validation_set is not None:
                dev_loss = self.evaluate(validation_set, model)
                self.logger.info(f"\tdev loss {dev_loss}")

        self.model = model
        return epoch_loss

    def predict(self, x):
        # x = (n_samples, n_features)
        assert self.model is not None
        tensor_x = torch.Tensor(x).to(device)
        prediction = self.model(tensor_x)
        return prediction.cpu().detach().numpy(), None
