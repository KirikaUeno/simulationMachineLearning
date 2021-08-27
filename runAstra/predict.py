import io
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn


def get_x_test_and_y_weights():
    f = io.open("inputParametersValues.txt", 'r', encoding='utf-8', newline='\n', errors='ignore')
    x_test = []
    for line in f:
        x_test.append(float(line[:-1]))

    print(x_test)
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
    device = torch.device("cuda:0")
    x_test, y_weights = get_x_test_and_y_weights()

    nn_model = nn.Sequential(
        nn.Linear(len(x_test), 256,bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, len(y_weights)),
    )
    nn_model.type(torch.cuda.FloatTensor)
    nn_model.to(device)
    nn_model.load_state_dict(torch.load("nnModel.pt"))

    pred = nn_model(torch.tensor([x_test]).to(device))
    y = []
    for j in range(len(pred[0])):
        y.append(pred[0, j].item())
    print(y)
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def predict_tree():
    x_test, y_weights = get_x_test_and_y_weights()

    filename = 'treeModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    pred = regr.predict([list(x_test)])
    y = []
    for j in range(len(pred[0])):
        y.append(pred[0, j].item())
    print(y)
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def predict_tree_boost():
    x_test, y_weights = get_x_test_and_y_weights()

    filename = 'treeBoostModel.sav'
    regr = pickle.load(open(filename, 'rb'))
    pred = regr.predict([list(x_test)])
    y = []
    for j in range(len(pred[0])):
        y.append(pred[0, j].item())
    print(y)
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def predict_nn_boost():
    device = torch.device("cuda:0")
    x_test, y_weights = get_x_test_and_y_weights()

    nn_model = nn.Sequential(
        nn.Linear(len(x_test), 256,bias=True),
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
        y.append((pred[0, j].item()+predNN[0, j].item())/2)
    print(y)
    y = list(np.array(y) / np.array(y_weights))
    info = ""
    f = open("predictOutput.txt", "w")

    for i in range(len(pred[0])):
        info += str(y[i]) + "\n"
    f.write(info)
    f.close()


def train(model):
    if model == "NN":
        predict_nn()
    elif model == "Tree":
        predict_tree()
    elif model == "Tree Boost":
        predict_tree_boost()
    else:
        predict_nn_boost()


print("start python predict")
train(sys.argv[1])
