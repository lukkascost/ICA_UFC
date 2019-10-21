import numpy as np
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

oExp11 = Experiment.load("Objects/EXP01_DT1_20.gzip".format())
COLORS = ['GREEN', 'RED', 'BLUE']
base1 = np.loadtxt("Datasets/dt_1.txt", delimiter=" ")


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
print("EXPERIMENTO 1 MELHOR RESULTADO" ,oData11.params)

RMSE = []
MSE = []
for i in oDataSet11.dataSet:
    RMSE.append(i.params['RMSE'])
    MSE.append(i.params['MSE'])
MSE = np.array(MSE)
RMSE = np.array(RMSE)
print("\tRMSE MEDIO ", np.mean(RMSE), "DESVIO", np.std(RMSE))
print("\tMSE MEDIO ", np.mean(MSE), "DESVIO", np.std(MSE))


for k in range(-120, 120):
    x = k / 10
    plt.scatter(x, 2 * x + 3, color=COLORS[0], linewidths=1, marker=".")
    data = (x - oDataSet11.normalize_between[0, 1]) / (
                oDataSet11.normalize_between[0, 0] - oDataSet11.normalize_between[0, 1])
    data = np.matrix([-1, data]).T
    pred = oData11.model.predict(data)[0, 0]
    plt.scatter(x, pred , color=COLORS[1], linewidths=1, marker=".")
plt.scatter(x, pred,  color=COLORS[1], linewidths=1, marker=".",label='PREDITO')
plt.scatter(x, 2 * x + 3,  color=COLORS[0], linewidths=1, marker=".",label='REAL')
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig("Results/et1_ex1_surface.png", dpi=100, bbox_inches="tight")
plt.clf()



plt.scatter(base1[:,0], base1[:,1], color=COLORS[2], linewidths=1, marker=".",label='DATASET' )
plt.plot(base1[:,0], 2*base1[:,0] + 3, color=COLORS[0],label='REAL' )
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig("Results/dataset1.png", dpi=100, bbox_inches="tight")
plt.show()

