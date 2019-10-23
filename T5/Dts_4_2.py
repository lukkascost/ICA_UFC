# coding=utf-8
import numpy as np
from matplotlib import cm

from MachineLearn.Classes import Experiment, DataSet, Data

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold

from T5.Perceptron_r import multi_Layered_perceptron_linear

import matplotlib.pyplot as plt

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 0.05
epochs = 300
K_FOLD = 5
GRID = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/pmsm_temperature_data.csv", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12], skiprows=1,
                  delimiter=",")
classes = np.loadtxt("Datasets/pmsm_temperature_data.csv", usecols=[8], skiprows=1, delimiter=",")
#for x, y in enumerate(base):
#    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = base
oDataSet.labels = classes
oDataSet.number_of_samples = base.shape[0]
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
oDataSet.labels = np.array([classes]).T
print("Carregou")
for j in range(20):
    slices = KFold(n_splits=K_FOLD)
    oData = Data(1, 31, samples=50)
    indices = np.arange(oDataSet.attributes.shape[0])
    np.random.shuffle(indices)
    oData.Testing_indexes = indices[int(oDataSet.attributes.shape[0] * 0.05):]
    oData.Training_indexes = indices[:int(oDataSet.attributes.shape[0] * 0.05)]

    grid_result = np.zeros((len(GRID), K_FOLD))
    for g1, g_param in enumerate(GRID):
        k_slice = 0
        for train, test in slices.split(oData.Training_indexes):
            print("Treinando....")
            mpl = multi_Layered_perceptron_linear(LEARNING_RATE, (oDataSet.attributes.shape[1], g_param, 1))
            mpl.train_regression(oDataSet.attributes[oData.Training_indexes[train]],
                                 oDataSet.labels[oData.Training_indexes[train]], epochs)
            y_pred = []
            y_true = []
            print("Testando.....")
            for i in test:
                y_pred.append(mpl.predict(oDataSet.attributes[oData.Training_indexes[i]])[0, 0])
                y_true.append(oDataSet.labels[oData.Training_indexes[i]])
            grid_result[g1, k_slice] = mean_squared_error(y_true, y_pred)
            k_slice += 1
            print(g_param, k_slice)
    print(grid_result, j)
    best_p = GRID[np.argmin(np.mean(grid_result, axis=1))]
    mpl = multi_Layered_perceptron_linear(LEARNING_RATE, (oDataSet.attributes.shape[1], best_p, 1))
    mpl.train_regression(oDataSet.attributes[oData.Training_indexes],
                         oDataSet.labels[oData.Training_indexes], epochs)
    y_pred = []
    y_true = []
    for i in oData.Testing_indexes:
        y_pred.append(mpl.predict(oDataSet.attributes[i])[0, 0])
        y_true.append(oDataSet.labels[i])
    oData.model = mpl
    oData.params = {"k_fold": K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES": GRID, "LEARNING RATE": LEARNING_RATE,
                    "EPOCHS": epochs, "MSE": mean_squared_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}

    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento Temperatura rotor MLP 20 realizaçoes.".format())
oExp.save("Objects/EXP02_4_LP_20.gzip".format())
