# coding=utf-8
import numpy as np
from matplotlib import cm
from MachineLearn.Classes import Experiment, DataSet, Data

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from T5.Perceptron import multi_Layered_perceptron_Logistic

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 0.5
epochs = 200
K_FOLD = 5
GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/iris.data", usecols=range(4), delimiter=",")
classes = np.loadtxt("Datasets/iris.data", dtype=object, usecols=-1, delimiter=",")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()

for j in range(20):
    slices = KFold(n_splits=K_FOLD)
    oData = Data(len(oDataSet.labelsNames), 31, samples=50)
    oData.random_training_test_by_percent(np.unique(classes, return_counts=True)[1], 0.8)
    grid_result = np.zeros((len(GRID), K_FOLD))
    for g1, g_param in enumerate(GRID):
        k_slice = 0
        for train, test in slices.split(oData.Training_indexes):
            mpl = multi_Layered_perceptron_Logistic(LEARNING_RATE, (base.shape[1], g_param, len(oDataSet.labelsNames)))
            mpl.train_classifier(oDataSet.attributes[oData.Training_indexes[train]],
                                 oDataSet.labels[oData.Training_indexes[train]], epochs)
            y_pred = []
            y_true = []
            for i in test:
                y_pred.append(mpl.predict(oDataSet.attributes[oData.Training_indexes[i]]))
                y_true.append(int(oDataSet.labels[oData.Training_indexes[i]]))
            grid_result[g1, k_slice] = accuracy_score(y_true, y_pred, normalize=True)
            k_slice+=1
    print(grid_result)
    best_p = GRID[np.argmax(np.mean(grid_result, axis=1))]
    mpl = multi_Layered_perceptron_Logistic(LEARNING_RATE, (base.shape[1], best_p, len(oDataSet.labelsNames)))
    mpl.train_classifier(oDataSet.attributes[oData.Training_indexes],
                         oDataSet.labels[oData.Training_indexes], epochs)
    y_pred = []
    y_true = []
    for i in oData.Testing_indexes:
        y_pred.append(mpl.predict(oDataSet.attributes[i]))
        y_true.append(int(oDataSet.labels[i]))
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    oData.confusion_matrix = confusion_matrix(y_true, y_pred)
    oData.model = mpl
    oData.params = {"k_fold" : K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES": GRID, "LEARNING RATE": LEARNING_RATE, "EPOCHS": epochs   }
    oDataSet.append(oData)
    print(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento iris LP 20 realizaçoes.".format())
oExp.save("Objects/EXP01_2_LP_20.gzip".format())
