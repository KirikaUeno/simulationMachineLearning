import io
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset(fname):
    x = []
    y = []
    names = []
    namesX = []
    local_x = []
    local_y = []
    local_names = []
    local_namesX = []
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
                namesX.append(local_namesX)
            continue
        if tokens[0] == "output:":
            isOutput = True
            isInput = False
            local_y = []
            local_names = []
            local_namesX = []
            if local_x != []:
                x.append(local_x)
            continue
        if isInput:
            local_x.append(float(tokens[1]))
            local_namesX.append(tokens[0])
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
            y.shape[0] * y / np.sum(np.abs(y), axis=0)).astype(np.float32), names, namesX


class DESY_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.datasetX, self.datasetY, self.names, self.namesX = load_dataset(path)

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        if self.transform != None:
            self.datasetX[index] = self.transform(self.datasetX[index])
        ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index, ind])