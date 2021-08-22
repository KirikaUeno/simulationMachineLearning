import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler
from sklearn.tree import DecisionTreeRegressor
import pickle
from torchvision import models

import numpy as np
import io

device = torch.device("cuda:0")


def load_dataset(fname):
    x = []
    y = []
    names = []
    local_x=[]
    local_y=[]
    local_names=[]
    isInput=False
    isOutput=False
    f = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for i, line in enumerate(f):
        tokens = line.strip().split(' ')
        if tokens[0]=="input:":
            isInput=True
            isOutput=False
            local_x=[]
            if local_y!=[]:
                y.append(local_y)
                names.append(local_names)
            continue
        if tokens[0]=="output:":
            isOutput=True
            isInput=False
            local_y=[]
            local_names=[]
            if local_x!=[]:
                x.append(local_x)
            continue
        if isInput:
            local_x.append(float(tokens[1]))
        if isOutput:
            local_y.append(float(tokens[1]))
            local_names.append(tokens[0])
    if local_y!=[]:
        y.append(local_y)
        names.append(local_names)
    f.close()
    x=np.array(x)
    y=np.array(y)
    f = open("weights.txt", "w")
    info = ""
    weights_x = x.shape[0]/np.sum(np.abs(x),axis=0)
    weights_y = y.shape[0]/np.sum(np.abs(y),axis=0)
    for i in range(weights_x.shape[0]):
        info+=str(weights_x[i])+"\n"
    for i in range(weights_y.shape[0]):
        info+=str(weights_y[i])+"\n"
    f.write(info)
    f.close()
    return (x.shape[0]*x/np.sum(np.abs(x),axis=0)).astype(np.float32), (y.shape[0]*y/np.sum(np.abs(y),axis=0)).astype(np.float32), names


#NN here
class DESY_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.datasetX, self.datasetY, self.names = load_dataset(path)

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        if self.transform != None:
            self.datasetX[index] = self.transform(self.datasetX[index])
        ind = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index,ind])


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, scheduler=None):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train() # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            indices = prediction
            correct_samples += torch.sum(torch.abs(indices-y) < y*0.001)/(y.shape[1] if len(y.shape)>1 else 1)
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

def compute_accuracy(model, loader):
    model.eval()
    correct_samples = 0
    total_samples = 0

    for i_step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        prediction = model(x)
        correct_samples += torch.sum(torch.abs(prediction-y) < y*0.001)/(y.shape[1] if len(y.shape)>1 else 1)
        total_samples += y.shape[0]

    accuracy = float(correct_samples) / total_samples
    return accuracy


def train_NN():
    #2d:
    train_dataset = DESY_dataset("data/information2d.txt")
    batch_size = 4

    data_size = len(train_dataset)
    validation_split = 0.0
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         sampler=val_sampler)

    nn_model = nn.Sequential(
        nn.Linear(2, 10,bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(10, 24),
        nn.ReLU(inplace=True),
        nn.Linear(24, 48),
        nn.ReLU(inplace=True),
        nn.Linear(48, 84),
        nn.ReLU(inplace=True),
        nn.Linear(84, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 90),
        nn.ReLU(inplace=True),
        nn.Linear(90, 52),
        nn.ReLU(inplace=True),
        nn.Linear(52, 18),
        nn.ReLU(inplace=True),
        nn.Linear(18, train_dataset[0][1].shape[0]),
    )
    nn_model.type(torch.cuda.FloatTensor)
    nn_model.to(device)

    loss = nn.MSELoss(reduction='mean')

    optimizer = optim.Adam(nn_model.parameters(), lr=5e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

    loss_history, train_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 500, scheduler)

    optimizer = optim.Adam(nn_model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)

    loss_history, train_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 500, scheduler)


def train_tree():
    print("start train tree")
    X, y = load_dataset("informationCorrected.txt")

    data_size = len(X)
    validation_split = 0.1
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]

    # Fit regression model
    regr = DecisionTreeRegressor(max_depth=8)
    regr.fit(X_train, y_train)

    # save the model to disk
    filename = 'treeModel.sav'
    pickle.dump(regr, open(filename, 'wb'))
    print("end train tree")


print("start_train_python")
train_tree()
