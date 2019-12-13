# coding=utf-8
from comet_ml import Experiment
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from MachineLearn.Classes import DataSet, Data

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

from T8.algoritmo_genetico import fit

COLOR = cm.rainbow(np.linspace(0, 1, 5))
LEARNING_RATE = 50
epochs = 1000
K_FOLD = 2
GRID_NEURON = [50]
GRID_B = [1000]

oDataSet = DataSet()
base = np.loadtxt("Datasets/XOR.txt", usecols=range(2), delimiter=",")
classes = np.loadtxt("Datasets/XOR.txt", dtype=float, usecols=-1, delimiter=",")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()

for j in range(20):
    experiment = Experiment(api_key="9F7edG4BHTWFJJetI2XctSUzM",
                            project_name="mest-rn-t8-xor",
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
                model = fit(oDataSet.attributes[oData.Training_indexes[train]],
                            oDataSet.labels[oData.Training_indexes[train]], LEARNING_RATE, epochs, 0.2, 0.1, 0.7)

                y_pred = model._predict(oDataSet.attributes[oData.Training_indexes[test]]).argmax(axis=1).T.tolist()[0]
                y_true = oDataSet.labels[oData.Training_indexes[test]]
                grid_result[g1, g2, k_slice] = accuracy_score(y_true, y_pred)
                # print(grid_result)
                k_slice += 1
                print(grid_result)

    model = fit(oDataSet.attributes[oData.Training_indexes],
                oDataSet.labels[oData.Training_indexes],  LEARNING_RATE, epochs, 0.2, 0.1, 0.7)

    y_pred = model._predict(oDataSet.attributes[oData.Testing_indexes]).argmax(axis=1).T.tolist()[0]
    y_true = oDataSet.labels[oData.Testing_indexes]
    experiment.log_other("pesos", str(model.genes))
    experiment.log_metric("test_accuracy", accuracy_score(y_true, y_pred))
    experiment.log_metric("beta", LEARNING_RATE)
    experiment.log_metric("neurons", epochs)
    experiment.log_confusion_matrix(matrix=confusion_matrix(y_true, y_pred).tolist(), labels=oDataSet.labelsNames)
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    oData.confusion_matrix = confusion_matrix(y_true, y_pred)
    oData.model = model
    oData.params = {"k_fold": K_FOLD, "GRID_RESULT": grid_result, "GRID_VALUES_NEURON": GRID_NEURON,
                    "GRID_VALUES_BETA": GRID_B, "LEARNING RATE": LEARNING_RATE,
                    "EPOCHS": epochs}
    experiment.log_other("params", oData.params)
    y_pred = model._predict(oDataSet.attributes[oData.Training_indexes]).argmax(axis=1).T.tolist()[0]
    y_true = oDataSet.labels[oData.Training_indexes]
    experiment.log_metric("train_accuracy", accuracy_score(y_true, y_pred))
    oDataSet.append(oData)

    random_matrix = np.random.random((1000, 2))
    y_pred = model._predict(random_matrix).argmax(axis=1).T.tolist()[0]

    for i in range(1000):
        plt.scatter(random_matrix[i,0], random_matrix[i,1], color=COLOR[y_pred[i]])
    experiment.log_figure(figure=plt)
    plt.show()
    experiment.end()

