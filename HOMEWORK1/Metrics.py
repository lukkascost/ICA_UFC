import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from numpy import linalg as LA

LABELS = ["{RI}", "{Na}", "{Mg}", "{Al}", "{Si}", "{K}", "{Ca}", "{Ba}", "{Fe}"]
index = np.loadtxt("../glass.data.txt", usecols=0, delimiter=",", dtype=int)
attributes = np.loadtxt("../glass.data.txt", usecols=[x for x in range(1, 10)], delimiter=",")
labels = np.loadtxt("../glass.data.txt", usecols=-1, delimiter=",", dtype=int)


""" Unconditional analysis Mean, std and skewness results.
"""
resultTable = np.zeros((3, 9))
strres = ""
for i in range(9):
    resultTable[0, i] = np.mean(attributes[:, i])
    resultTable[1, i] = np.std(attributes[:, i])
    resultTable[2, i] = sp.skew(attributes[:, i])
    strres += "\\textit{}&{:.04f} & {:.04f} & {:.04f}\\\\\n".format(LABELS[i], resultTable[0, i],
                                                                             resultTable[1, i], resultTable[2, i])
print strres

""" class conditional analysis Mean, std and skewness results per class.
"""
resultTable = np.zeros((7, 3, 9))
for j in [1, 2, 3, 5, 6, 7]:
    for i in range(9):
        data = attributes[labels == j, i]
        resultTable[j - 1, 0, i] = np.mean(data)
        resultTable[j - 1, 1, i] = np.std(data)
        resultTable[j - 1, 2, i] = sp.skew(data)

for i in range(9):
    strres = " \\textit{} ".format(LABELS[i])
    for j in [1, 2, 3, 5, 6, 7]:
        strres += "&{:.02f}$\pm${:.04f}".format(resultTable[j - 1, 0, i],resultTable[j - 1, 1, i])
    strres += "\\\\ "
    print strres
print
for i in range(9):
    strres = " \\textit{} ".format(LABELS[i])
    for j in [1, 2, 3, 5, 6, 7]:
        strres += "&{:.04f}".format(resultTable[j - 1, 2, i])
    strres += "\\\\ "
    print strres


""" Unconditional bi-variate analysis
"""
strres = ""
covTable = np.zeros((9, 9))
for i in range(9):
    strres += "\n\\textit{}".format(LABELS[i])
    for j in range(9):
        covTable[i, j] = np.corrcoef(attributes[:, i], attributes[:, j])[1, 0]
        strres += "&{:.04f}".format(covTable[i, j])
    strres += "\\\\"
print strres


""" Unconditional multi-variate analysis
"""
mu = attributes.mean(axis=0)
sigma = attributes.std(axis=0)
attributes = (attributes - mu)/sigma

resultPca = PCA(attributes, standardize=False)
plt.clf()
result = np.dot(attributes, resultPca.Wt)
print resultPca.fracs
for i in range(9):
    plt.scatter(index, resultPca.Y[:, i], label="Attribute {}".format(i+1))
plt.legend()
plt.show()