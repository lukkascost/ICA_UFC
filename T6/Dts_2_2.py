# coding=utf-8
from comet_ml import Experiment
import numpy as np
from matplotlib import cm
from MachineLearn.Classes import DataSet, Data

from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import *
from keras.utils import to_categorical
from rbflayer import RBFLayer, InitCentersRandom
from sklearn.preprocessing import LabelBinarizer
import  matplotlib.pyplot as plt

from T6.kmeans_initializer import InitCentersKMeans

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 0.1
epochs = 200
K_FOLD = 3
GRID_NEURON = [20, 15, 10, 5]
GRID_B = [.25, .5, .75, 1]
_OPTIMIZER = RMSprop(learning_rate=LEARNING_RATE)

oDataSet = DataSet()
base = np.loadtxt("Datasets/abalone.data", usecols=range(8), delimiter=",")
classes = np.loadtxt("Datasets/abalone.data", usecols=-1, delimiter=",")


for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
oDataSet.labels = np.array([classes]).T


for j in range(20):
    experiment = Experiment(api_key="9F7edG4BHTWFJJetI2XctSUzM",
                            project_name="mest-rn-t6-abalone",
                            workspace="lukkascost",
                            )
    experiment.set_name("REALIZACAO_{:02d}".format(j + 1))
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
                                    input_shape=(base.shape[1],))
                model.add(rbflayer)
                model.add(Dense(1))
                model.compile(loss='mse',
                              optimizer=_OPTIMIZER)

                model.fit(oDataSet.attributes[oData.Training_indexes[train]],
                          oDataSet.labels[oData.Training_indexes[train]],
                          batch_size=25,
                          epochs=epochs,
                          verbose=0)

                y_pred = model.predict(oDataSet.attributes[oData.Training_indexes[test]])
                y_true = oDataSet.labels[oData.Training_indexes[test]]
                grid_result[g1,g2, k_slice] = mean_squared_error(y_true, y_pred)
                # print(grid_result)
                k_slice += 1
                print(grid_result)
    best_p = GRID_NEURON[np.unravel_index(np.argmin(np.mean(grid_result, axis=2)), grid_result.shape[:2])[0]]
    best_b = GRID_B[np.unravel_index(np.argmin(np.mean(grid_result, axis=2)), grid_result.shape[:2])[1]]

    model = Sequential()
    rbflayer = RBFLayer(best_p,
                        initializer=InitCentersRandom(oDataSet.attributes[oData.Training_indexes]),
                        betas=best_b,
                        input_shape=(base.shape[1],))

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
    plt.plot(model.history.history['loss'])
    experiment.log_figure(figure=plt, figure_name='Loss curve')
    plt.show()


    experiment.log_metric("test_accuracy", mean_squared_error(y_true, y_pred))
    experiment.log_metric("test_accuracy_rmse", np.sqrt(mean_squared_error(y_true, y_pred)))
    experiment.log_metric("beta", best_b)
    experiment.log_metric("neurons", best_p)
    model.save('model.h5')
    experiment.log_asset("model.h5")
    model.save_weights('model.weights')
    experiment.log_asset("model.weights")

    print(mean_squared_error(y_true, y_pred))
    oData.model = model
    oData.params = {"k_fold": K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES_NEURON": GRID_NEURON,"GRID_VALUES_BETA": GRID_B, "LEARNING RATE": LEARNING_RATE,
                    "EPOCHS": epochs}
    experiment.log_other("params", oData.params)
    y_pred = model.predict(oDataSet.attributes[oData.Training_indexes])
    y_true = oDataSet.labels[oData.Training_indexes]
    experiment.log_metric("train_accuracy", mean_squared_error(y_true, y_pred))
    experiment.log_metric("train_accuracy_rmse", np.sqrt(mean_squared_error(y_true, y_pred)))

    experiment.end()
    oDataSet.append(oData)