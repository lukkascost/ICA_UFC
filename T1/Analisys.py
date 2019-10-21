import numpy as np
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

oExp11 = Experiment.load("Objects/EXP01_1_PS_20.gzip".format())
oExp12 = Experiment.load("Objects/EXP01_2_PS_20.gzip".format())
oExp13 = Experiment.load("Objects/EXP01_3_PS_20.gzip".format())
oExp2 = Experiment.load("Objects/EXP02_PS_20.gzip".format())
COLORS = ['RED', 'BLUE']

print(oExp11)
print()
print(oExp12)
print()
print(oExp13)
print()
print(oExp2)
print()


# Etapa 1
def getBestTrain(exp, name):
    """Etapa 1: Matriz confusao e grafico para melhor treinamento."""
    oDataSet = exp.experimentResults[0]
    best = 0;
    oBestData = None
    for oData in oDataSet.dataSet:
        txAcc = oData.get_metrics()[1, -1]
        if txAcc > best:
            best = txAcc
            oBestData = oData

    perc = oBestData.model
    plt.plot(range(1, len(perc.et) + 1), perc.et)
    plt.savefig(("Results/" + name), dpi=100, bbox_inches="tight")
    plt.show()
    return oBestData


def extractTrain_Test(odata, odataSet, path):
    labels = []
    for i in odataSet.labels:
        labels.append(odataSet.labelsNames[int(i)])
    data = np.hstack((odataSet.attributes[odata.Training_indexes], np.array([labels]).T[odata.Training_indexes]))
    np.savetxt(path + "Train.txt", data, fmt='%s', delimiter=',')
    data = np.hstack((odataSet.attributes[odata.Testing_indexes], np.array([labels]).T[odata.Testing_indexes]))
    np.savetxt(path + "Test.txt", data, fmt='%s', delimiter=',')



oData11 = getBestTrain(oExp11, "et1_ex1_1.png")
oData12 = getBestTrain(oExp12, "et1_ex1_2.png")
oData13 = getBestTrain(oExp13, "et1_ex1_3.png")
oData2 = getBestTrain(oExp2, "et1_ex2.png")

print("\nMatriz confusao: Setosa vs Outras\n", oData11.confusion_matrix)
print("\nMatriz confusao: versicolor vs Outras\n", oData12.confusion_matrix)
print("\nMatriz confusao: virginica vs Outras\n", oData13.confusion_matrix)
print("\nMatriz confusao: AND \n", oData2.confusion_matrix)

extractTrain_Test(oData11, oExp11.experimentResults[0], "CasosTreinoTeste/caso_1_1_")
extractTrain_Test(oData12, oExp12.experimentResults[0], "CasosTreinoTeste/caso_1_2_")
extractTrain_Test(oData13, oExp13.experimentResults[0], "CasosTreinoTeste/caso_1_3_")
extractTrain_Test(oData2, oExp2.experimentResults[0], "CasosTreinoTeste/caso_2_")

# Etapa 2
plt.scatter(0, 0, marker='o', edgecolors='none', color='black', label='Amostra Treino')
plt.scatter(0, 0, marker='v', edgecolors='none', color='black', label='Amostra Teste')
for x1 in range(-20, 121):
    print(x1)
    for x2 in range(-20, 121):
        plt.scatter([x1 / 100], [x2 / 100], color=COLORS[oData2.model.predict(np.matrix([[-1, x1 / 100, x2 / 100]]).T)])

for i in oData2.Training_indexes:
    plt.plot(oExp2.experimentResults[0].attributes[i][0],
             oExp2.experimentResults[0].attributes[i][1],
             fillstyle='full',
             color='white',
             marker='o',
             markerfacecoloralt='white'
             )
    plt.plot(oExp2.experimentResults[0].attributes[i][0],
             oExp2.experimentResults[0].attributes[i][1],
             fillstyle='none',
             color=COLORS[int(oExp2.experimentResults[0].labels[i])],
             marker='o',
             markerfacecoloralt='white'
             )
for i in oData2.Testing_indexes:
    plt.plot(oExp2.experimentResults[0].attributes[i][0],
             oExp2.experimentResults[0].attributes[i][1],
             fillstyle='full',
             color='white',
             marker='v',
             markerfacecoloralt='white'
             )
    plt.plot(oExp2.experimentResults[0].attributes[i][0],
             oExp2.experimentResults[0].attributes[i][1],
             fillstyle='none',
             color=COLORS[int(oExp2.experimentResults[0].labels[i])],
             marker='v',
             markerfacecoloralt='white',
             )
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.grid(True)
plt.savefig(("Results/et2_ex2_surface.png"), dpi=100, bbox_inches="tight")
plt.show()
