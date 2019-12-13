import numpy as np
from keras.engine.saving import load_model
from matplotlib.lines import Line2D

from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt

from T6_2.rbflayer import RBFLayer

oExp11 = Experiment.load("Objects/EXP01_1_LP_20.gzip".format())
oExp12 = Experiment.load("Objects/EXP01_2_LP_20.gzip".format())
oExp13 = Experiment.load("Objects/EXP01_3_LP_20.gzip".format())
oExp14 = Experiment.load("Objects/EXP01_4_LP_20.gzip".format())
oExp15 = Experiment.load("Objects/EXP01_5_LP_20.gzip".format())


COLORS = ['GREEN', 'RED', 'BLUE']
#COLORS = ['BLUE', 'ORANGE', 'RED']

MARKER = ['o', '^', "*"]
base1 = np.loadtxt("Datasets/XOR.txt", delimiter=",")


print(oExp11)
print()
print(oExp12)
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
    best = 0
    oBestData = None
    for oData in oDataSet.dataSet:
        txAcc = oData.get_metrics()[1, -1]
        if txAcc > best:
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

oData11 = getBestTrain(oExp11, "et1_ex1_1.png")
oData12 = getBestTrain(oExp12, "et1_ex2_1.png")
oData13 = getBestTrain(oExp13, "et1_ex3_1.png")
oData14 = getBestTrain(oExp14, "et1_ex4_1.png")
oData15 = getBestTrain(oExp15, "et1_ex5_1.png")

print("\nMatriz confusao: Artificial\n", oData11.confusion_matrix)
print("\nMatriz confusao: Iris \n", oData12.confusion_matrix)
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
atts = []
for x1 in range(-10, 35):
    print(x1)
    for x2 in range(-20, 35):
        atts.append([x1/25, x2/25])
atts = np.matrix(atts)
y_pred = oData11.model.predict(atts).argmax(axis=1)
for i in range(atts.shape[0]):
    plt.scatter(atts[i,0], atts[i,1], color=COLORS[y_pred[i]])

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

