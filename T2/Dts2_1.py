from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from MachineLearn.Classes import Experiment, DataSet, Data
from T2.Perceptron import Perceptron_Adaline
from mpl_toolkits.mplot3d import Axes3D


COLOR = cm.rainbow(np.linspace(0, 1, 5))
learning_rate = 0.1
epochs = 1000

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("Datasets/dt_2.txt", usecols=range(2), delimiter=" ")
classes = np.loadtxt("Datasets/dt_2.txt", usecols=-1, delimiter=" ")

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
classes = np.array([classes]).T
for j in range(20):
    print(j)
    oData = Data(2, 31, samples=50)
    indices = np.arange(oDataSet.attributes.shape[0])
    np.random.shuffle(indices)
    oData.Testing_indexes = indices[int(oDataSet.attributes.shape[0] * 0.85):]
    oData.Training_indexes = indices[:int(oDataSet.attributes.shape[0] * 0.85)]

    perc = Perceptron_Adaline(learning_rate)
    perc.train(oDataSet.attributes[oData.Training_indexes], classes[oData.Training_indexes].copy(), epochs)
    oData.model = perc
    ert = 0
    plotar = []
    for i in oData.Testing_indexes:
        data = np.matrix(np.hstack(([-1], oDataSet.attributes[i]))).T
        predict = perc.predict(data)[0, 0]
        plotar.append([classes[i,0], predict])
        ert += (classes[i, 0] - predict)**2
    plotar = np.array(plotar)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    p = [oDataSet.attributes[0],  oDataSet.attributes[-1]]
    res = []
    for i in p:
        data = np.matrix(np.hstack(([-1], i))).T
        predict = perc.predict(data)[0, 0]
        res.append([i, predict])
    res = np.array(res)
    ax.plot(base[[0,-1],0], base[[0,-1],1], res[:,1])

    p = [base[0], base[-1]]
    res = []
    for i in p:
        predict = 2*i[0] + 3*i[1] -20
        res.append([i, predict])
    res = np.array(res)
    ax.plot(base[[0,-1],0], base[[0,-1],1], res[:,1])

    plt.show()
    oData.params = {"MSE": ert/oData.Testing_indexes.shape[0],"RMSE":np.sqrt(ert/oData.Testing_indexes.shape[0] ) }

    print(oData.params)
    oDataSet.append(oData)

oExp.add_data_set(oDataSet)
oExp.save("Objects/EXP01_DT2_20.gzip")
