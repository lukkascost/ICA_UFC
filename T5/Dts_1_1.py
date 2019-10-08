import numpy as np

from T5.Perceptron import multi_Layered_perceptron_Logistic

base = np.loadtxt("Datasets/artifitial1.data", usecols=range(2), delimiter=",")
classes = np.loadtxt("Datasets/artifitial1.data", dtype=float, usecols=-1, delimiter=",")
classes = classes.reshape((150,1))
mpl = multi_Layered_perceptron_Logistic(0.1, (2, 10, 10, 3))
mpl.train_classifier(base,classes,10)
