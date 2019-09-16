from matplotlib import cm
import numpy as np

from MachineLearn.Classes import Experiment, DataSet, Data
from T3.Perceptron import Layered_perceptron
import matplotlib.pyplot as plt

from T4.Perceptron import Layered_perceptron_Logistic

COLOR = cm.rainbow(np.linspace(0, 1, 5))
learning_rate = 0.01
epochs = 5000

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/column_3C.dat", usecols=range(6), delimiter=" ")
classes = np.loadtxt("Datasets/column_3C.dat", dtype=object, usecols=-1, delimiter=" ")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
for j in range(20):
    print(j)
    oData = Data(len(oDataSet.labelsNames), 31, samples=50)
    oData.random_training_test_by_percent([60, 150, 100], 0.8)
    perc = Layered_perceptron_Logistic(learning_rate, len(oDataSet.labelsNames))
    perc.train(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes], epochs)
    oData.model = perc
    oData.confusion_matrix = np.zeros((len(oDataSet.labelsNames), len(oDataSet.labelsNames)))
    for i in oData.Testing_indexes:
        data = np.matrix(np.hstack(([-1], oDataSet.attributes[i]))).T
        predicted = perc.predict(data)
        oData.confusion_matrix[int(oDataSet.labels[i]), predicted] += 1
    print(oData)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento COLUNA 3C LP 20 realizaçoes.".format())
oExp.save("Objects/EXP01_3_LP_20.gzip".format())

oExp = Experiment.load("Objects/EXP01_3_LP_20.gzip".format())
print(oExp)
print(oExp.experimentResults[0].sum_confusion_matrix)
