import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset, SubsetRandomSampler
from sklearn.tree import DecisionTreeRegressor
import pickle

import numpy as np
import loadDataset


def train_model(model, train_loader, loss, optimizer, num_epochs, scheduler=None):
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

        if scheduler is not None:
            scheduler.step()

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)

        print("Average loss: %f, Train accuracy: %f, epoch: %f" % (ave_loss, train_accuracy, epoch))

    return loss_history, train_history


def train_NN():
    train_dataset = loadDataset.DESY_dataset("informationCorrected.txt")
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

    train_model(nn_model, train_loader, loss, optimizer, 4000, scheduler)
    torch.save(nn_model.state_dict(), 'nnModel.pt')


def train_tree():
    X, y, names, namesX = loadDataset.load_dataset("informationCorrected.txt")
    X_train = X[:]
    y_train = y[:]
    regr = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
    # save the model to disk
    filename = 'treeModel.sav'
    pickle.dump(regr, open(filename, 'wb'))


def train_tree_boost():
    X, y, names, namesX = loadDataset.load_dataset("informationCorrected.txt")
    X_train = X[:]
    y_train = y[:]
    est = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=10, random_state=0, loss='ls')).fit(
        X_train, y_train)

    # save the model to disk
    filename = 'treeBoostModel.sav'
    pickle.dump(est, open(filename, 'wb'))


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
