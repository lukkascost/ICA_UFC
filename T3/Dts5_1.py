from matplotlib import cm
import numpy as np

from MachineLearn.Classes import Experiment, DataSet, Data
from T3.Perceptron import Layered_perceptron

COLOR = cm.rainbow(np.linspace(0, 1, 5))
learning_rate = 0.01
epochs = 5000

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/breast-cancer-wisconsin.data", usecols=range(1,10), dtype=int,delimiter=",")
classes = np.loadtxt("Datasets/breast-cancer-wisconsin.data", dtype=int, usecols=-1, delimiter=",")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
for j in range(20):
    print(j)
    oData = Data(len(oDataSet.labelsNames), 31, samples=50)
    oData.random_training_test_by_percent(np.unique(classes, return_counts=True)[1], 0.8)
    perc = Layered_perceptron(learning_rate, len(oDataSet.labelsNames))
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
                  description="  Experimento Cancer LP 20 realizaçoes.".format())
oExp.save("Objects/EXP01_5_LP_20.gzip".format())

oExp = Experiment.load("Objects/EXP01_5_LP_20.gzip".format())
print(oExp)
print(oExp.experimentResults[0].sum_confusion_matrix)
