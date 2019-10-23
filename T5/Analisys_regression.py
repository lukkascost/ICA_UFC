import numpy as np
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

oExp11 = Experiment.load("Objects/EXP02_1_LP_20.gzip".format())
oExp12 = Experiment.load("Objects/EXP02_2_LP_20.gzip".format())
oExp13 = Experiment.load("Objects/EXP02_3_LP_20.gzip".format())

COLORS = ['GREEN', 'RED', 'BLUE']
MARKER = ['o', '^', "*"]
base1 = np.loadtxt("Datasets/artifitial1.data", delimiter=",")

def imprimir_resultado(oexp, name, oData):
    oDataSet = oexp.experimentResults[0]
    print("EXPERIMENTO "+name+" MELHOR RESULTADO MSE: ", oData.params['MSE'], )

    RMSE = []
    MSE = []
    for i in oDataSet.dataSet:
        RMSE.append(i.params['RMSE'])
        MSE.append(i.params['MSE'])
    MSE = np.array(MSE)
    RMSE = np.array(RMSE)
    print("\tRMSE MEDIO ", np.mean(RMSE), "DESVIO", np.std(RMSE))
    print("\tMSE MEDIO ", np.mean(MSE), "DESVIO", np.std(MSE))
    print()

def getBestTrain(exp, name):
    """Etapa 1: Matriz confusao e grafico para melhor treinamento."""
    oDataSet = exp.experimentResults[0]
    best = np.inf;
    oBestData = None
    for oData in oDataSet.dataSet:
        txAcc = oData.params['MSE']
        if txAcc < best:
            best = txAcc
            oBestData = oData
    perc = oBestData.model
    plt.plot(range(1, len(perc.et) + 1), perc.et)
    plt.savefig(("Results/" + name), dpi=100, bbox_inches="tight")
    plt.show()
    return oBestData

oData11 = getBestTrain(oExp11, "et1_ex1_1_reg.png")
oData12 = getBestTrain(oExp12, "et1_ex2_1_reg.png")
oData13 = getBestTrain(oExp13, "et1_ex3_1_reg.png")

imprimir_resultado(oExp11,"Artificial1",oData11)
imprimir_resultado(oExp12,"Abalone",oData12)
imprimir_resultado(oExp13,"Consumo gasolina",oData13)

print("\nMatriz confusao: Artificial\n", oData11.params)
print("\nMatriz confusao: Abalone \n", oData12.params)
print("\nMatriz confusao: Consumo gasolina 3c\n", oData13.params)

for i in base1:
    plt.scatter(i[0], i[1], marker=MARKER[0], edgecolors='none', color=COLORS[0])
plt.savefig(("Results/et1_ex1_dataset_reg.png"), dpi=100, bbox_inches="tight")
plt.show()

max = oExp11.experimentResults[0].normalize_between[0,0]
min = oExp11.experimentResults[0].normalize_between[0,1]
for i in range(720):
    plt.scatter(i, oData11.model.predict((i-min)/(max-min))[0,0], marker=MARKER[0], edgecolors='none', color=COLORS[0])
    plt.scatter(i, 3 * np.sin(i*np.pi/180) +1 , marker=MARKER[1], edgecolors='none', color=COLORS[1])
plt.savefig(("Results/et1_ex1_surface_reg.png"), dpi=100, bbox_inches="tight")
plt.show()