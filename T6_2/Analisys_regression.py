import numpy as np
from keras.engine.saving import load_model
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

from T6_2.rbflayer import RBFLayer

oExp11 = Experiment.load("Objects/EXP02_1_LP_20.gzip".format())
oExp12 = Experiment.load("Objects/EXP02_2_LP_20.gzip".format())
oExp13 = Experiment.load("Objects/EXP02_3_LP_20.gzip".format())

COLORS = ['GREEN', 'RED', 'BLUE']
COLORS = ['BLUE', 'ORANGE', 'RED']

MARKER = ['o', '^', "*"]
base1 = np.loadtxt("Datasets/artifitial1.data", delimiter=",")


def imprimir_resultado(oexp, name, oData):
    oDataSet = oexp.experimentResults[0]
    print("EXPERIMENTO " + name + " MELHOR RESULTADO MSE: ", oData.params['MSE'], )

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
    best = np.inf
    oBestData = None
    for oData in oDataSet.dataSet:
        txAcc = oData.params['MSE']
        if txAcc < best:
            best = txAcc
            oBestData = oData

    with open("model.h5", "wb") as newFile:
        newFile.write(oBestData.model[0])
    perc = oBestData.model[1]
    oBestData.model = load_model("model.h5", custom_objects={'RBFLayer': RBFLayer})

    plt.plot(range(1, len(perc) + 1), perc)
    plt.savefig(("Results/" + name), dpi=100, bbox_inches="tight")
    plt.show()
    return oBestData


oData11 = getBestTrain(oExp11, "et1_ex1_1_reg.png")
oData12 = getBestTrain(oExp12, "et1_ex2_1_reg.png")
oData13 = getBestTrain(oExp13, "et1_ex3_1_reg.png")

imprimir_resultado(oExp11, "Artificial1", oData11)
imprimir_resultado(oExp12, "Abalone", oData12)
imprimir_resultado(oExp13, "Consumo gasolina", oData13)

print("\nMatriz confusao: Artificial\n", oData11.params)
print("\nMatriz confusao: Abalone \n", oData12.params)
print("\nMatriz confusao: Consumo gasolina 3c\n", oData13.params)

for i in base1:
    plt.scatter(i[0], i[1], marker=MARKER[0], edgecolors='none', color=COLORS[0])
plt.savefig(("Results/et1_ex1_dataset_reg.png"), dpi=100, bbox_inches="tight")
plt.show()

max = oExp11.experimentResults[0].normalize_between[0, 0]
min = oExp11.experimentResults[0].normalize_between[0, 1]
random_matrix = np.arange(500)
random_matrix = (random_matrix - min) / (max - min)
y_pred = oData11.model.predict(random_matrix)
for i in range(500):
    plt.scatter(i, y_pred[i], marker=MARKER[0], edgecolors='none',
                color=COLORS[0])
    plt.scatter(i, 3 * np.sin(i * np.pi / 180) + 1, marker=MARKER[1], edgecolors='none', color=COLORS[1])
plt.savefig(("Results/et1_ex1_surface_reg.png"), dpi=100, bbox_inches="tight")
plt.show()
