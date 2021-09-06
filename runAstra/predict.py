import sys

import torch
import torch.nn as nn
from matplotlib import cm
import pickle
import matplotlib.pyplot as plt

import numpy as np
import io
import loadDataset

x_coord = 0
y_coord = 1
z_coord = 0
initial_static = 0
size_x = 100
size_y = 100


def get_x_test_and_y_weights():
    f = io.open("inputParametersValues.txt", 'r', encoding='utf-8', newline='\n', errors='ignore')
    x_test = []
    for line in f:
        x_test.append(float(line[:-1]))

    f = io.open("weights.txt", 'r', encoding='utf-8', newline='\n', errors='ignore')
    x_weights = []
    y_weights = []
    for i, line in enumerate(f):
        if i < len(x_test):
            x_weights.append(float(line[:-1]))
        elif line != "":
            y_weights.append(float(line[:-1]))

    return list(np.array(x_test) * np.array(x_weights)), y_weights


def predict_nn():
    x_test, y_weights = get_x_test_and_y_weights()

    nn_model = nn.Sequential(
        nn.Linear(len(x_test), 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, len(y_weights)),
    )
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load("nnModel.pt"))

    pred = nn_model(torch.FloatTensor([x_test]))
    y = []
    for j in range(len(pred[0])):
        y.append(pred[0, j].item())
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def draw_NN():
    x_test, y_weights = get_x_test_and_y_weights()

    nn_model = nn.Sequential(
        nn.Linear(len(x_test), 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, len(y_weights)),
    )
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load("nnModel.pt"))
    draw_train_results("NN", nn_model)


def predict_tree():
    x_test, y_weights = get_x_test_and_y_weights()
    filename = 'treeModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    pred = regr.predict([list(x_test)])
    y = []
    for j in range(len(pred[0])):
        y.append(pred[0, j].item())
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def draw_tree():
    x_test, y_weights = get_x_test_and_y_weights()

    filename = 'treeModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    draw_train_results("tree", regr)


def predict_tree_boost():
    x_test, y_weights = get_x_test_and_y_weights()

    filename = 'treeBoostModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    pred = regr.predict([list(x_test)])
    y = []
    for j in range(len(pred[0])):
        y.append(pred[0, j].item())
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def draw_tree_boost():
    x_test, y_weights = get_x_test_and_y_weights()

    filename = 'treeBoostModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    draw_train_results("tree_boost", regr)


def predict_nn_boost():
    device = torch.device("cuda:0")
    x_test, y_weights = get_x_test_and_y_weights()

    nn_model = nn.Sequential(
        nn.Linear(len(x_test), 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, len(y_weights)),
    )
    nn_model.type(torch.cuda.FloatTensor)
    nn_model.to(device)
    nn_model.load_state_dict(torch.load("nnModel.pt"))

    filename = 'treeBoostModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    pred = regr.predict([list(x_test)])

    predNN = nn_model(torch.tensor([x_test]).to(device))
    y = []
    for j in range(len(pred[0])):
        y.append((pred[0, j].item() + predNN[0, j].item()) / 2)
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def draw_NN_boost():
    x_test, y_weights = get_x_test_and_y_weights()

    nn_model = nn.Sequential(
        nn.Linear(len(x_test), 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, len(y_weights)),
    )
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load("nnModel.pt"))

    filename = 'treeBoostModel.sav'
    regr = pickle.load(open(filename, 'rb'))

    draw_train_results("NN+boost", nn_model, aux_model=regr)


def draw_train_results(model_name, model, aux_model=None):
    train_dataset = loadDataset.DESY_dataset("informationCorrected.txt")
    X = []
    Y = []
    Z = []
    if np.array(train_dataset[:][0][0, :]).shape[0] > 1:
        init_params = np.array(train_dataset[initial_static][0])
        for i in range(len(train_dataset)):
            count_as_dataset = True
            for j, val in enumerate(init_params):
                if j != x_coord and j != y_coord:
                    if val != train_dataset[i][0][j]:
                        count_as_dataset = False
            if count_as_dataset:
                X.append(train_dataset[i][0][x_coord].item())
                Y.append(train_dataset[i][0][y_coord].item())
                Z.append(train_dataset[i][1][z_coord].item())
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        x_surf, y_surf = np.meshgrid(
            np.linspace(X.min() - (X.max() - X.min()) / 10, X.max() + (X.max() - X.min()) / 10, size_x),
            np.linspace(Y.min() - (Y.max() - Y.min()) / 10, Y.max() + (Y.max() - Y.min()) / 10, size_y))

        list_to_model = []
        for i in range(np.array(train_dataset[:][0][0, :]).shape[0]):
            if i == x_coord:
                list_to_model.append(x_surf.flatten())
            elif i == y_coord:
                list_to_model.append(y_surf.flatten())
            else:
                list_to_model.append(train_dataset[initial_static][0][i].item() * np.ones(size_x * size_y))
        model_viz = np.expand_dims(np.array(list_to_model).T, 0)
        model_viz1 = np.array(list_to_model).T
        if model_name == "NN":
            z_surf = np.array(model(torch.FloatTensor(model_viz))[0, :, z_coord].tolist())
        elif model_name == "tree":
            z_surf = model.predict(model_viz1)[:, z_coord]
        elif model_name == "tree_boost":
            z_surf = model.predict(model_viz1)[:, z_coord]
        elif aux_model is not None:
            z_surf = (np.array(model(torch.FloatTensor(model_viz))[0, :, z_coord].tolist()) + aux_model.predict(
                model_viz1)[:, z_coord]) / 2

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_surf, y_surf, z_surf.reshape(x_surf.shape), alpha=0.4, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)
        _, _, names, namesX = loadDataset.load_dataset("informationCorrected.txt")
        ax.set_xlabel(namesX[0][x_coord])
        ax.set_ylabel(namesX[0][y_coord])
        ax.set_zlabel(names[0][z_coord])
        plt.show()


def predict(model):
    if model == "NN":
        predict_nn()
    elif model == "Tree":
        predict_tree()
    elif model == "Tree_Boost":
        predict_tree_boost()
    elif model == "NN+Boost":
        predict_nn_boost()
    if model == "NN_draw":
        draw_NN()
    elif model == "Tree_draw":
        draw_tree()
    elif model == "Tree_Boost_draw":
        draw_tree_boost()
    elif model == "NN+Boost_draw":
        draw_NN_boost()


print("start python predict")
if len(sys.argv)>2:
    x_coord = int(sys.argv[2])
    y_coord = int(sys.argv[3])
    z_coord = int(sys.argv[4])
    initial_static = int(sys.argv[5])
predict(sys.argv[1])
