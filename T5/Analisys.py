import numpy as np
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

oExp11 = Experiment.load("Objects/EXP01_1_LP_20.gzip".format())


COLORS = ['GREEN', 'RED', 'BLUE']
MARKER = ['o', '^', "*"]
base1 = np.loadtxt("Datasets/artifitial1.data", delimiter=",")


print(oExp11)
print()



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

oData11 = getBestTrain(oExp11, "et1_ex1_1.png")

print("\nMatriz confusao: Artificial\n", oData11.confusion_matrix)

for i in base1:
    plt.scatter(i[0], i[1], marker=MARKER[int(i[2])], edgecolors='none', color=COLORS[int(i[2])])
plt.savefig(("Results/et1_ex1_dataset.png"), dpi=100, bbox_inches="tight")
plt.show()

# Etapa 2
plt.scatter(0, 0, marker='o', edgecolors='none', color='black', label='Amostra Treino')
plt.scatter(0, 0, marker='v', edgecolors='none', color='black', label='Amostra Teste')
for x1 in range(-10, 75):
    print(x1)
    for x2 in range(-20, 75):
        plt.scatter([x1 / 25], [x2 / 25], color=COLORS[oData11.model.predict([x1 / 25,x2 / 25 ])])

for i in oData11.Training_indexes:
    plt.plot(oExp11.experimentResults[0].attributes[i][0],
             oExp11.experimentResults[0].attributes[i][1],
             fillstyle='full',
             color='white',
             marker='o',
             markerfacecoloralt='white'
             )
    plt.plot(oExp11.experimentResults[0].attributes[i][0],
             oExp11.experimentResults[0].attributes[i][1],
             fillstyle='none',
             color=COLORS[int(oExp11.experimentResults[0].labels[i])],
             marker='o',
             markerfacecoloralt='white'
             )
for i in oData11.Testing_indexes:
    plt.plot(oExp11.experimentResults[0].attributes[i][0],
             oExp11.experimentResults[0].attributes[i][1],
             fillstyle='full',
             color='white',
             marker='v',
             markerfacecoloralt='white'
             )
    plt.plot(oExp11.experimentResults[0].attributes[i][0],
             oExp11.experimentResults[0].attributes[i][1],
             fillstyle='none',
             color=COLORS[int(oExp11.experimentResults[0].labels[i])],
             marker='v',
             markerfacecoloralt='white',
             )
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(("Results/et1_ex1_surface.png"), dpi=100, bbox_inches="tight")
plt.show()

