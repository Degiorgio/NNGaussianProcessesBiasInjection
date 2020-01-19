import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from torch.utils import data
import data_loader

def standardize(x):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x

def normalize(x):
    # from sklearn.preprocessing import normalize
    # x = normalize(x)
    return x

class Network(nn.Module):
    def __init__(self, size, layers, width,
                 activation=nn.Sigmoid, initw=False):
        super().__init__()
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
        modules = []

        # project
        modules.append(nn.Linear(size, width))
        # hidden
        for x in range(layers):
            modules.append(nn.Linear(width, width))
            modules.append(activation())
        # output
        modules.append(nn.Linear(width, size))

        self.net = nn.Sequential(*modules)
        if initw:
            self.net.apply(init_weights)

    def forward(self, x):
        return self.net.forward(x)


class Model:
    def __init__(self, *args, **kwargs):
        self.logger = config.getlogger("NN")
        self.loss_func = torch.nn.MSELoss()
        self.model = None
        return super().__init__(*args, **kwargs)

    def train(self, training_set, validation_set,
              epochs=10, width=10, batch_size=20, layers=2, learning_rate=0.1):

        training_set = data_loader.make_dataset(normalize(training_set.X), training_set.Y)

        def evaluate(testing_set):
            nexamples = testing_set.X.shape[0]
            tensor_x = torch.Tensor(testing_set.X)
            tensor_y = torch.Tensor(testing_set.Y)
            my_dataset = data.TensorDataset(tensor_x, tensor_y)
            my_dataloader = data.DataLoader(my_dataset, batch_size=nexamples)
            losses = 0.0; it = 0
            for i, (x,y) in enumerate(iter(my_dataloader)):
                prediction = model(x)
                loss = self.loss_func(prediction, y)
                losses += loss.item()
                it += 1
            return losses/it

        dim = training_set.X.shape[1]
        nexamples = training_set.X.shape[0]
        model = Network(dim, layers, width)

        self.logger.debug(f'dimensionality of each sample: {dim}')
        self.logger.debug(f'number of training samples: {nexamples}')
        self.logger.debug(f'Layers: {layers}')
        self.logger.debug(f'Width: {width}')
        self.logger.debug(f'Epochs: {epochs}')
        self.logger.debug(f'Learning Rate: {learning_rate}')

        model.train()
        tensor_x = torch.Tensor(training_set.X)
        tensor_y = torch.Tensor(training_set.Y)
        my_dataset = data.TensorDataset(tensor_x, tensor_y)
        my_dataloader = data.DataLoader(my_dataset, batch_size=batch_size)

        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        for epoch in range(epochs):
            running_loss = 0.0
            self.logger.debug(f'epoch: {epoch}')
            for i, (x,y) in enumerate(iter(my_dataloader)):
                optimizer.zero_grad()
                # forward pass
                prediction = model(x)
                # compute loss
                loss = self.loss_func(prediction, y)
                running_loss += loss.item()
                # backprop
                loss.backward()
                optimizer.step()
            scheduler.step()


            epoch_loss = running_loss / batch_size
            dev_loss = evaluate(validation_set)
            self.logger.debug(f'\ttrain loss {epoch_loss}')
            self.logger.debug(f'\tdev loss {dev_loss}')
        self.model  = model

    def predict(self, x):
        # x = (n_samples, n_features)
        assert self.model is not None
        x =  normalize(x)
        tensor_x = torch.Tensor(x)
        prediction = self.model(tensor_x)
        return prediction.detach().numpy(), None
