# coding=utf-8
import numpy as np
from matplotlib import cm
from MachineLearn.Classes import Experiment, DataSet, Data

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import *
from keras.utils import to_categorical
from rbflayer import RBFLayer, InitCentersRandom
from sklearn.preprocessing import LabelBinarizer

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 0.5
epochs = 200
K_FOLD = 5
GRID = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
_OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0.0, decay=0.0, nesterov=False)

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/iris.data", usecols=range(4), delimiter=",")
classes = np.loadtxt("Datasets/iris.data", dtype=object, usecols=-1, delimiter=",")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()

lb = LabelBinarizer()
lb.fit(oDataSet.labels)

for j in range(20):
    slices = KFold(n_splits=K_FOLD,shuffle=True)
    oData = Data(len(oDataSet.labelsNames), 31, samples=50)
    oData.random_training_test_by_percent(np.unique(classes, return_counts=True)[1], 0.8)
    grid_result = np.zeros((len(GRID), K_FOLD))
    for g1, g_param in enumerate(GRID):
        k_slice = 0
        for train, test in slices.split(oData.Training_indexes):
            model = Sequential()
            rbflayer = RBFLayer(g_param,
                                initializer=InitCentersRandom(oDataSet.attributes[oData.Training_indexes[train]]),
                                betas=2,
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

            y_pred = model.predict(oDataSet.attributes[oData.Training_indexes[test]])
            y_true = oDataSet.labels[oData.Training_indexes[test]]
            grid_result[g1, k_slice] = accuracy_score(y_true, y_pred.argmax(axis=1))
            print(grid_result)
            k_slice+=1
    print(grid_result)
    best_p = GRID[np.argmax(np.mean(grid_result, axis=1))]

    model = Sequential()
    rbflayer = RBFLayer(best_p,
                        initializer=InitCentersRandom(oDataSet.attributes[oData.Training_indexes]),
                        betas=2,
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

    y_pred = model.predict(oDataSet.attributes[oData.Training_indexes])
    y_true = oDataSet.labels[oData.Training_indexes]

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    oData.confusion_matrix = confusion_matrix(y_true, y_pred)
    oData.model = model
    oData.params = {"k_fold" : K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES": GRID, "LEARNING RATE": LEARNING_RATE, "EPOCHS": epochs   }
    oDataSet.append(oData)
    print(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento iris LP 20 realizaçoes.".format())
oExp.save("Objects/EXP01_2_LP_20.gzip".format())
