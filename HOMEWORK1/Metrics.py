import numpy as np
import scipy.stats as sp

index = np.loadtxt("../glass.data.txt", usecols=0, delimiter=",", dtype=int)
attributes = np.loadtxt("../glass.data.txt", usecols=[x for x in range(1, 10)], delimiter=",")
labels = np.loadtxt("../glass.data.txt", usecols=-1, delimiter=",", dtype=int)

resultTable = np.zeros((3, 9))
for i in range(9):
    resultTable[0, i] = np.mean(attributes[:, i])
    resultTable[1, i] = np.std(attributes[:, i])
    resultTable[2, i] = sp.skew(attributes[:, i])

## Show results uncoditional per attribute
strres = ""
for i in range(9):
    strres += "Attribute {:02d}: {:.04f} &\t\t{:.04f} &\t\t{:.04f}\n".format(i + 1, resultTable[0, i],
                                                                             resultTable[1, i], resultTable[2, i])
print strres


