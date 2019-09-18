import numpy as np
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

oExp11 = Experiment.load("Objects/EXP01_1_LP_20.gzip".format())
oExp12 = Experiment.load("Objects/EXP01_2_LP_20.gzip".format())
oExp22 = Experiment.load("Objects/EXP02_2_LP_20.gzip".format())
oExp13 = Experiment.load("Objects/EXP01_3_LP_20.gzip".format())
oExp14 = Experiment.load("Objects/EXP01_4_LP_20.gzip".format())
oExp15 = Experiment.load("Objects/EXP01_5_LP_20.gzip".format())

COLORS = ['GREEN', 'RED', 'BLUE']
MARKER = ['o', '^', "*"]
base1 = np.loadtxt("Datasets/artifitial1.data", delimiter=",")


print(oExp11)
print()
print(oExp12)
print()
print(oExp22)
print()
print(oExp13)
print()
print(oExp14)
print()
print(oExp15)
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
    plt.plot(range(1, len(perc.et) + 1), np.sum(perc.et, axis=1))
    plt.savefig(("Results/" + name), dpi=100, bbox_inches="tight")
    plt.show()
    return oBestData

oData11 = getBestTrain(oExp11, "et1_ex1_1.png")
oData12 = getBestTrain(oExp12, "et1_ex2_1.png")
oData22 = getBestTrain(oExp22, "et1_ex2_2.png")
oData13 = getBestTrain(oExp13, "et1_ex3_1.png")
oData14 = getBestTrain(oExp14, "et1_ex4_1.png")
oData15 = getBestTrain(oExp15, "et1_ex5_1.png")

print("\nMatriz confusao: Artificial\n", oData11.confusion_matrix)
print("\nMatriz confusao: Iris \n", oData12.confusion_matrix)
print("\nMatriz confusao: Iris 2 atributos \n", oData22.confusion_matrix)
print("\nMatriz confusao: Coluna 3c\n", oData13.confusion_matrix)
print("\nMatriz confusao: Dermatologia \n", oData14.confusion_matrix)
print("\nMatriz confusao: Cancer \n", oData15.confusion_matrix)


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
        plt.scatter([x1 / 25], [x2 / 25], color=COLORS[oData11.model.predict(np.matrix([[-1, x1 / 25, x2 / 25]]).T)])

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




for k,i in enumerate(oExp22.experimentResults[0].attributes):
    plt.scatter(i[0], i[1], marker=MARKER[int(oExp22.experimentResults[0].labels[k])], edgecolors='none', color=COLORS[int(oExp22.experimentResults[0].labels[k])])
plt.savefig(("Results/et1_ex2_dataset.png"), dpi=100, bbox_inches="tight")
plt.show()

# Etapa 2
plt.scatter(0, 0, marker='o', edgecolors='none', color='black', label='Amostra Treino')
plt.scatter(0, 0, marker='v', edgecolors='none', color='black', label='Amostra Teste')
for x1 in range(-10, 175):
    print(x1)
    for x2 in range(-20, 75):
        plt.scatter([x1 / 25], [x2 / 25], color=COLORS[oData22.model.predict(np.matrix([[-1, x1 / 25, x2 / 25]]).T)])

for i in oData22.Training_indexes:
    plt.plot(oExp22.experimentResults[0].attributes[i][0],
             oExp22.experimentResults[0].attributes[i][1],
             fillstyle='full',
             color='white',
             marker='o',
             markerfacecoloralt='white'
             )
    plt.plot(oExp22.experimentResults[0].attributes[i][0],
             oExp22.experimentResults[0].attributes[i][1],
             fillstyle='none',
             color=COLORS[int(oExp22.experimentResults[0].labels[i])],
             marker='o',
             markerfacecoloralt='white'
             )
for i in oData22.Testing_indexes:
    plt.plot(oExp22.experimentResults[0].attributes[i][0],
             oExp22.experimentResults[0].attributes[i][1],
             fillstyle='full',
             color='white',
             marker='v',
             markerfacecoloralt='white'
             )
    plt.plot(oExp22.experimentResults[0].attributes[i][0],
             oExp22.experimentResults[0].attributes[i][1],
             fillstyle='none',
             color=COLORS[int(oExp22.experimentResults[0].labels[i])],
             marker='v',
             markerfacecoloralt='white',
             )
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(("Results/et1_ex2_surface.png"), dpi=100, bbox_inches="tight")
plt.show()
