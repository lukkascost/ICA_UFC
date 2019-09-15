from matplotlib import cm
import numpy as np

from MachineLearn.Classes import Experiment, DataSet, Data
from T3.Perceptron import Layered_perceptron
import matplotlib.pyplot as plt

COLOR = cm.rainbow(np.linspace(0, 1, 5))
learning_rate = 0.01
epochs = 1000

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/artifitial1.data", usecols=range(2), delimiter=",")
classes = np.loadtxt("Datasets/artifitial1.data", dtype=object, usecols=-1, delimiter=",")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
# oDataSet.normalize_data_set()
for j in range(20):
    print(j)
    oData = Data(len(oDataSet.labelsNames), 31, samples=50)
    oData.random_training_test_by_percent([50, 50, 50], 0.75)
    perc = Layered_perceptron(learning_rate, len(oDataSet.labelsNames))
    perc.train(oDataSet.attributes[oData.Training_indexes], oDataSet.labels[oData.Training_indexes], epochs)
    oData.model = perc
    oData.confusion_matrix = np.zeros((len(oDataSet.labelsNames), len(oDataSet.labelsNames)))
    for i in oData.Testing_indexes:
        data = np.matrix(np.hstack(([-1], oDataSet.attributes[i]))).T
        predicted = perc.predict(data)
        oData.confusion_matrix[int(oDataSet.labels[i]), predicted] += 1
        plt.scatter(oDataSet.attributes[i,0],oDataSet.attributes[i,1], color=COLOR[int(oDataSet.labels[i])])
    for k1 in range(150):
        print("K ", k1)
        for k2 in range(30):
            x1 = k1 / 50
            x2 = k2 / 10
            result = perc.predict(np.array([[-1], [x1], [x2]]))
            plt.scatter(x1, x2, color=COLOR[result])
    print(oData)
    plt.show()
    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  Experimento iris  PS 20 realizaçoes.".format())
oExp.save("Objects/EXP01_1_PS_20.gzip".format())

oExp = Experiment.load("Objects/EXP01_1_PS_20.gzip".format())
print(oExp)
print(oExp.experimentResults[0].sum_confusion_matrix)
