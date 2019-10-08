import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from T5.Perceptron import multi_Layered_perceptron_Logistic

base = np.loadtxt("Datasets/iris.data", usecols=range(4), delimiter=",")
classes = np.loadtxt("Datasets/iris.data", dtype=float, usecols=-1, delimiter=",")
classes = classes.reshape((150, 1))
mpl = multi_Layered_perceptron_Logistic(0.1, (4, 10,10, 3))
mpl.train_classifier(base, classes, 2000)
y_pred = []
y_true = []
for i in range(base.shape[0]):
    y_pred.append(mpl.predict(base[i]))
    y_true.append(int(classes[i]))
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
plt.plot(mpl.et)
plt.show()