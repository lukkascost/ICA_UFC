# coding=utf-8
import numpy as np
from matplotlib import cm
from MachineLearn.Classes import Experiment,DataSet, Data

from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import *
from keras.utils import to_categorical
from rbflayer import RBFLayer, InitCentersRandom
from sklearn.preprocessing import LabelBinarizer
import  matplotlib.pyplot as plt

from T6_2.kmeans_initializer import InitCentersKMeans

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 0.1
epochs = 300
K_FOLD = 3
GRID_NEURON = [20, 15, 10, 5]
GRID_B = [.25, .5, .75, 1]
_OPTIMIZER = RMSprop(learning_rate=LEARNING_RATE)

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/artifitial1.data", usecols=range(1), delimiter=",")
classes = np.loadtxt("Datasets/artifitial1.data", usecols=-1, delimiter=",")


for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list([np.float32(y)]) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
oDataSet.labels = np.array([classes]).T

for j in range(2):
    slices = KFold(n_splits=K_FOLD, shuffle=True)
    oData = Data(1, 31, samples=50)
    indices = np.arange(oDataSet.attributes.shape[0])
    np.random.shuffle(indices)
    oData.Testing_indexes = indices[int(oDataSet.attributes.shape[0] * 0.85):]
    oData.Training_indexes = indices[:int(oDataSet.attributes.shape[0] * 0.85)]

    grid_result = np.zeros((len(GRID_NEURON), len(GRID_B), K_FOLD))
    for g1, g_param in enumerate(GRID_NEURON):
        for g2, g2_param in enumerate(GRID_B):
            k_slice = 0
            for train, test in slices.split(oData.Training_indexes):
                model = Sequential()
                rbflayer = RBFLayer(g_param,
                                    initializer=InitCentersRandom(oDataSet.attributes[oData.Training_indexes[train]]),
                                    betas=g2_param,
                                    input_shape=(1,))
                model.add(rbflayer)
                model.add(Dense(1))
                model.compile(loss='mse',
                              optimizer=_OPTIMIZER)

                model.fit(oDataSet.attributes[oData.Training_indexes[train]],
                          oDataSet.labels[oData.Training_indexes[train]],
                          batch_size=50,
                          epochs=epochs,
                          verbose=0)

                y_pred = model.predict(oDataSet.attributes[oData.Training_indexes[test]])
                y_true = oDataSet.labels[oData.Training_indexes[test]]
                grid_result[g1, g2, k_slice] = mean_squared_error(y_true, y_pred)
                k_slice += 1
                print(grid_result)
    best_p = GRID_NEURON[np.unravel_index(np.argmin(np.mean(grid_result, axis=2)), grid_result.shape[:2])[0]]
    best_b = GRID_B[np.unravel_index(np.argmin(np.mean(grid_result, axis=2)), grid_result.shape[:2])[1]]

    model = Sequential()
    rbflayer = RBFLayer(best_p,
                        initializer=InitCentersRandom(oDataSet.attributes[oData.Training_indexes]),
                        betas=best_b,
                        input_shape=(1,))

    model.add(rbflayer)
    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer=_OPTIMIZER)
    model.fit(oDataSet.attributes[oData.Training_indexes],
              oDataSet.labels[oData.Training_indexes],
              batch_size=50,
              epochs=epochs,
              verbose=1)

    y_pred = model.predict(oDataSet.attributes[oData.Testing_indexes])
    y_true = oDataSet.labels[oData.Testing_indexes]

    random_matrix = np.random.random((1000, 1))
    plt.scatter(random_matrix, model.predict(random_matrix), label="Curva modelo")
    plt.scatter(oDataSet.attributes, oDataSet.labels, label="Curva dataset")
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.show()

    model.save('model.h5')
    myArr = None
    with open("model.h5", "rb") as binaryfile:
        myArr = bytearray(binaryfile.read())
    oData.model = myArr, model.history.history['loss']
    oData.params = {"k_fold": K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES": (best_b,best_p), "LEARNING RATE": LEARNING_RATE,
                    "EPOCHS": epochs, "MSE": mean_squared_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}

    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento Artificial MLP 20 realizaçoes.".format())
oExp.save("Objects/EXP02_1_LP_20.gzip".format())
