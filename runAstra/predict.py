import io
import pickle
import numpy as np

print("start python predict")

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
    elif line!="":
        y_weights.append(float(line[:-1]))

x_test = list(np.array(x_test)*np.array(x_weights))

filename = 'treeModel.sav'
regr = pickle.load(open(filename, 'rb'))
pred = regr.predict([list(x_test)])
y = []
for j in range(len(pred[0])):
    y.append(pred[0, j].item())
print(y)
y = list(np.array(y)/np.array(y_weights))
info = ""
f = open("predictOutput.txt", "w")

for i in range(len(pred[0])):
    info += str(y[i]) + "\n"
f.write(info)
f.close()
