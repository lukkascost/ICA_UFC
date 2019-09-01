from MachineLearn.Classes import Experiment
import matplotlib.pyplot as plt


oExp11 = Experiment.load("Objects/EXP01_1_PS_20.gzip".format())
oExp12 = Experiment.load("Objects/EXP01_2_PS_20.gzip".format())
oExp13 = Experiment.load("Objects/EXP01_3_PS_20.gzip".format())
oExp2 = Experiment.load("Objects/EXP02_PS_20.gzip".format())

def getBestTrain(exp):
    """Etapa 1: Matriz confusao e grafico para melhor treinamento."""
    oDataSet = exp.experimentResults[0]
    best = 0;
    oBestData = None
    for oData in oDataSet.dataSet:
        txAcc = oData.get_metrics()[1,-1]
        if  txAcc > best:
            best = txAcc
            oBestData = oData
    return oBestData

oData11 = getBestTrain(oExp11)
oData12 = getBestTrain(oExp12)
oData13 = getBestTrain(oExp13)

print(oData11)
print(oData12)
print(oData13)