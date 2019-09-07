import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

oExp11 = Experiment.load("Objects/EXP01_DT2_20.gzip".format())
COLORS = ['GREEN', 'RED', 'BLUE']
base1 = np.loadtxt("Datasets/dt_2.txt", delimiter=" ")


# Etapa 1
def getBestTrain(exp):
    """Etapa 1: Matriz confusao e grafico para melhor treinamento."""
    oDataSet = exp.experimentResults[0]
    best = 1000000
    oBestData = None
    for oData in oDataSet.dataSet:
        txAcc = oData.params['MSE']
        if txAcc < best:
            best = txAcc
            oBestData = oData
    return oBestData


oData11 = getBestTrain(oExp11)
oDataSet11 = oExp11.experimentResults[0]
print("EXPERIMENTO 2 MELHOR RESULTADO", oData11.params)

RMSE = []
MSE = []
for i in oDataSet11.dataSet:
    RMSE.append(i.params['RMSE'])
    MSE.append(i.params['MSE'])
MSE = np.array(MSE)
RMSE = np.array(RMSE)
print("\tRMSE MEDIO ", np.mean(RMSE), "DESVIO", np.std(RMSE))
print("\tMSE MEDIO ", np.mean(MSE), "DESVIO", np.std(MSE))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(11, 70)

for x1 in range(0, 100):
    for x2 in range(0, 100):
        ax.scatter(x1, x2, (2 * x1) + (3 * x2) - 20, color=COLORS[0], linewidths=1, marker=".")
        data = (x1 - oDataSet11.normalize_between[0, 1]) / (
                oDataSet11.normalize_between[0, 0] - oDataSet11.normalize_between[0, 1])
        data = np.array([ data, (x2 - oDataSet11.normalize_between[1, 1]) / (
                oDataSet11.normalize_between[1, 0] - oDataSet11.normalize_between[1, 1])])
        data = np.matrix(np.hstack(([-1], data))).T

        pred = oData11.model.predict(data)[0, 0]
        ax.scatter(x1, x2, pred, color=COLORS[1], linewidths=1, marker=".")
ax.scatter(x1,x2, pred,  color=COLORS[1], linewidths=1, marker=".",label='PREDITO')
ax.scatter(x1,x2, (2 *x1) + (3*x2) -20,  color=COLORS[0], linewidths=1, marker=".",label='REAL')
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig("Results/et1_ex2_surface.png", dpi=100, bbox_inches="tight")
plt.clf()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(11, 90)

ax.scatter(base1[:, 0], base1[:, 1], base1[:, 2], color=COLORS[2], linewidths=1, marker=".", label='DATASET')
ax.scatter(base1[:, 0], base1[:, 1], (2 * base1[:, 0]) + (3 * base1[:, 1]) - 20, color=COLORS[0], label='REAL')
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig("Results/dataset2.png", dpi=100, bbox_inches="tight")
plt.show()
