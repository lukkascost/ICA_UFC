# coding=utf-8
from comet_ml import Experiment
from comet_ml.utils import ConfusionMatrix

import numpy as np
from matplotlib import cm
from MachineLearn.Classes import DataSet, Data

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import *
from keras.utils import to_categorical
from rbflayer import RBFLayer, InitCentersRandom
from sklearn.preprocessing import LabelBinarizer

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 0.1
epochs = 50
K_FOLD = 3
GRID_NEURON = [5, 10, 15, 20]
GRID_B = [.25, .5, .75, 1]
_OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0.0, decay=0.0, nesterov=False)

oDataSet = DataSet()
base = np.loadtxt("Datasets/dermatology.data", usecols=range(34), dtype=int,delimiter=",")
classes = np.loadtxt("Datasets/dermatology.data", dtype=int, usecols=-1, delimiter=",")


for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()

lb = LabelBinarizer()
lb.fit(oDataSet.labels)

for j in range(9,20):
    experiment = Experiment(api_key="9F7edG4BHTWFJJetI2XctSUzM",
                            project_name="mest-rn-t6-dermatology",
                            workspace="lukkascost",
                            )
    experiment.set_name("REALIZACAO_{:02d}".format(j + 1))

    slices = KFold(n_splits=K_FOLD, shuffle=True)
    oData = Data(len(oDataSet.labelsNames), 31, samples=50)
    oData.random_training_test_by_percent(np.unique(classes, return_counts=True)[1], 0.8)
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
                model.add(Dense(len(lb.classes_), activation='sigmoid'))
                model.compile(loss='categorical_crossentropy',
                              optimizer=_OPTIMIZER)
                model.fit(oDataSet.attributes[oData.Training_indexes[train]],
                          lb.transform(oDataSet.labels[oData.Training_indexes[train]]),
                          batch_size=1,
                          epochs=epochs,
                          verbose=0)

                y_pred = model.predict(oDataSet.attributes[oData.Training_indexes[test]]).argmax(axis=1)
                y_true = oDataSet.labels[oData.Training_indexes[test]]
                grid_result[g1,g2, k_slice] = accuracy_score(y_true, y_pred)
                # print(grid_result)
                k_slice += 1
                print(grid_result)
    best_p = GRID_NEURON[np.unravel_index(np.argmax(np.mean(grid_result, axis=2)), grid_result.shape[:2])[0]]
    best_b = GRID_B[np.unravel_index(np.argmax(np.mean(grid_result, axis=2)), grid_result.shape[:2])[1]]

    model = Sequential()
    rbflayer = RBFLayer(best_p,
                        initializer=InitCentersRandom(oDataSet.attributes[oData.Training_indexes]),
                        betas=best_b,
                        input_shape=(base.shape[1],))

    model.add(rbflayer)
    model.add(Dense(len(lb.classes_), activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=_OPTIMIZER)
    model.fit(oDataSet.attributes[oData.Training_indexes],
              lb.transform(oDataSet.labels[oData.Training_indexes]),
              batch_size=1,
              epochs=epochs,
              verbose=1)

    y_pred = model.predict(oDataSet.attributes[oData.Testing_indexes]).argmax(axis=1)
    y_true = oDataSet.labels[oData.Testing_indexes]

    experiment.log_metric("test_accuracy", accuracy_score(y_true, y_pred))
    experiment.log_metric("beta", best_b)
    experiment.log_metric("neurons", best_p)
    experiment.log_confusion_matrix(matrix=confusion_matrix(y_true, y_pred).tolist(), labels=oDataSet.labelsNames)
    # model.save('model.h5')
    # experiment.log_asset("model.h5")
    model.save_weights('model.weights')
    experiment.log_asset("model.weights")

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    oData.confusion_matrix = confusion_matrix(y_true, y_pred)
    oData.model = model
    oData.params = {"k_fold": K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES_NEURON": GRID_NEURON,"GRID_VALUES_BETA": GRID_B, "LEARNING RATE": LEARNING_RATE,
                    "EPOCHS": epochs}
    experiment.log_other("params", oData.params)
    y_pred = model.predict(oDataSet.attributes[oData.Training_indexes]).argmax(axis=1)
    y_true = oDataSet.labels[oData.Training_indexes]
    experiment.log_metric("train_accuracy", accuracy_score(y_true, y_pred))
    experiment.end()
    oDataSet.append(oData)