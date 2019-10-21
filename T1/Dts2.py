from matplotlib import cm
import numpy as np

from MachineLearn.Classes import Experiment, DataSet, Data
from T1.Perceptron import Perceptron

COLOR = cm.rainbow(np.linspace(0, 1, 5))
learning_rate = 0.01
epochs = 200000

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/dataset2.txt", usecols=range(2), delimiter=",")
classes = np.loadtxt("Datasets/dataset2.txt", usecols=-1, delimiter=",")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
# oDataSet.normalize_data_set()
for j in range(20):
    print(j)
    oData = Data(2, 31, samples=50)
    oData.random_training_test_by_percent([600, 600], 0.8)
    perc = Perceptron(learning_rate)
    perc.train(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes], epochs)
    oData.model = perc
    oData.confusion_matrix = np.zeros((2, 2))
    for i in oData.Testing_indexes:
        data = np.matrix(np.hstack(([-1], oDataSet.attributes[i]))).T
        oData.confusion_matrix[int(oDataSet.labels[i]), perc.predict(data)] += 1
    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento dataset2 PS 20 realizaçoes.".format())
oExp.save("Objects/EXP02_PS_20.gzip".format())

oExp = Experiment.load("Objects/EXP02_PS_20.gzip".format())
print(oExp)
print(oExp.experimentResults[0].sum_confusion_matrix)
