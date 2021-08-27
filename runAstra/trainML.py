import sys

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset, SubsetRandomSampler
from sklearn.tree import DecisionTreeRegressor
import pickle
import matplotlib.pyplot as plt

import numpy as np
import io


def load_dataset(fname):
    x = []
    y = []
    names = []
    local_x = []
    local_y = []
    local_names = []
    isInput = False
    isOutput = False
    f = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for i, line in enumerate(f):
        tokens = line.strip().split(' ')
        if tokens[0] == "input:":
            isInput = True
            isOutput = False
            local_x = []
            if local_y != []:
                y.append(local_y)
                names.append(local_names)
            continue
        if tokens[0] == "output:":
            isOutput = True
            isInput = False
            local_y = []
            local_names = []
            if local_x != []:
                x.append(local_x)
            continue
        if isInput:
            local_x.append(float(tokens[1]))
        if isOutput:
            local_y.append(float(tokens[1]))
            local_names.append(tokens[0])
    if local_y != []:
        y.append(local_y)
        names.append(local_names)
    f.close()
    x = np.array(x)
    y = np.array(y)
    f = open("weights.txt", "w")
    info = ""
    weights_x = x.shape[0] / np.sum(np.abs(x), axis=0)
    weights_y = y.shape[0] / np.sum(np.abs(y), axis=0)
    for i in range(weights_x.shape[0]):
        info += str(weights_x[i]) + "\n"
    for i in range(weights_y.shape[0]):
        info += str(weights_y[i]) + "\n"
    f.write(info)
    f.close()
    return (x.shape[0] * x / np.sum(np.abs(x), axis=0)).astype(np.float32), (
                y.shape[0] * y / np.sum(np.abs(y), axis=0)).astype(np.float32), names


class DESY_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.datasetX, self.datasetY, self.names = load_dataset(path)

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        if self.transform != None:
            self.datasetX[index] = self.transform(self.datasetX[index])
        ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index, ind])


def train_model(model, train_loader, loss, optimizer, num_epochs, scheduler=None):
    print("start model nn train")
    loss_history = []
    train_history = []
    for epoch in range(num_epochs):
        model.train()  # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            # x = x.to(device)
            # y = y.to(device)
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            indices = prediction
            correct_samples += torch.sum(torch.abs(indices - y) < y * 0.001) / (y.shape[1] if len(y.shape) > 1 else 1)
            total_samples += y.shape[0]

            loss_accum += loss_value

        if scheduler != None:
            scheduler.step()

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)

        print("Average loss: %f, Train accuracy: %f, epoch: %f" % (ave_loss, train_accuracy, epoch))

    return loss_history, train_history


def train_NN():
    # 2d:
    print("start nn training")
    train_dataset = DESY_dataset("informationCorrected.txt")
    print("dataset created")
    batch_size = 4

    data_size = len(train_dataset)
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices = indices

    train_sampler = SubsetRandomSampler(train_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler)

    nn_model = nn.Sequential(
        nn.Linear(train_dataset[0][0].shape[0], 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, train_dataset[0][1].shape[0]),
    )
    nn_model.type(torch.FloatTensor)
    # nn_model.to(device)

    loss = nn.MSELoss(reduction='mean')

    optimizer = optim.Adam(nn_model.parameters(), lr=2e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.7)

    print("prepared to train nn")
    train_model(nn_model, train_loader, loss, optimizer, 4000, scheduler)
    print("trained nn")
    torch.save(nn_model.state_dict(), 'nnModel.pt')
    X, y, names = load_dataset("informationCorrected.txt")
    from operator import add

    if np.array(train_dataset[:][0][0, :]).shape[0] == 2:
        X = np.array(train_dataset[:][0][:, 0])
        Y = np.array(train_dataset[:][0][:, 1])
        Z = np.array(train_dataset[:][1][:, 1])
        x_surf, y_surf = np.meshgrid(
            np.linspace(X.min() - (X.max() - X.min()) / 10, X.max() + (X.max() - X.min()) / 10, 100),
            np.linspace(Y.min() - (Y.max() - Y.min()) / 10, Y.max() + (Y.max() - Y.min()) / 10, 100))

        model_viz = np.expand_dims(np.array([x_surf.flatten(), y_surf.flatten()]).T,0)
        print(torch.tensor(model_viz).shape)
        z_surf = np.array(nn_model(torch.FloatTensor(model_viz))[0,:, 1].tolist())

        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_surf, y_surf, z_surf.reshape(x_surf.shape), alpha=0.4, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)
        ax.set_xlabel('MAXB')
        ax.set_ylabel('QBUNCH')
        ax.set_zlabel('alphaX')
        plt.show()


def train_tree():
    print("start train tree")
    X, y, names = load_dataset("informationCorrected.txt")
    X_train = X[:]
    y_train = y[:]
    regr = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)

    # save the model to disk
    filename = 'treeModel.sav'
    pickle.dump(regr, open(filename, 'wb'))
    print("end train tree")

    train_dataset = DESY_dataset("informationCorrected.txt")
    if np.array(train_dataset[:][0][0, :]).shape[0] == 2:
        X = np.array(train_dataset[:][0][:, 0])
        Y = np.array(train_dataset[:][0][:, 1])
        Z = np.array(train_dataset[:][1][:, 1])
        x_surf, y_surf = np.meshgrid(
            np.linspace(X.min() - (X.max() - X.min()) / 10, X.max() + (X.max() - X.min()) / 10, 100),
            np.linspace(Y.min() - (Y.max() - Y.min()) / 10, Y.max() + (Y.max() - Y.min()) / 10, 100))
        model_viz = np.array([x_surf.flatten(), y_surf.flatten()]).T
        z_surf = regr.predict(model_viz)[:, 1]

        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_surf, y_surf, z_surf.reshape(x_surf.shape), alpha=0.4, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)
        ax.set_xlabel('MAXB')
        ax.set_ylabel('QBUNCH')
        ax.set_zlabel('alphaX')
        plt.show()


def train_tree_boost():
    print("start train tree")
    X, y, names = load_dataset("informationCorrected.txt")
    X_train = X[:]
    y_train = y[:]
    est = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=10, random_state=0, loss='ls')).fit(
        X_train, y_train)

    # save the model to disk
    filename = 'treeBoostModel.sav'
    pickle.dump(est, open(filename, 'wb'))
    print("end train tree")

    train_dataset = DESY_dataset("informationCorrected.txt")
    if np.array(train_dataset[:][0][0, :]).shape[0] == 2:
        X = np.array(train_dataset[:][0][:, 0])
        Y = np.array(train_dataset[:][0][:, 1])
        Z = np.array(train_dataset[:][1][:, 1])
        x_surf, y_surf = np.meshgrid(
            np.linspace(X.min() - (X.max() - X.min()) / 10, X.max() + (X.max() - X.min()) / 10, 100),
            np.linspace(Y.min() - (Y.max() - Y.min()) / 10, Y.max() + (Y.max() - Y.min()) / 10, 100))
        model_viz = np.array([x_surf.flatten(), y_surf.flatten()]).T
        z_surf = est.predict(model_viz)[:, 1]

        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_surf, y_surf, z_surf.reshape(x_surf.shape), alpha=0.4, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)
        ax.set_xlabel('First parameter')
        ax.set_ylabel('Second parameter')
        ax.set_zlabel('alphaX')
        plt.show()


def train(model):
    if model == "NN":
        train_NN()
    elif model == "Tree":
        train_tree()
    elif model == "Tree Boost":
        train_tree_boost()
    else:
        train_NN()
        train_tree_boost()


print("start_train_python")
train(sys.argv[1])
