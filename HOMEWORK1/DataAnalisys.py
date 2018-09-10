import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import matplotlib.cm as cm

index = np.loadtxt("../glass.data.txt", usecols=0, delimiter=",", dtype=int)
attributes = np.loadtxt("../glass.data.txt", usecols=[x for x in range(1, 10)], delimiter=",")
labels = np.loadtxt("../glass.data.txt", usecols=-1, delimiter=",", dtype=int)
MK = [".", "_"]
COLOR = cm.rainbow(np.linspace(0, 1, 8))


""" Unconditional analysis: generate histogram for each attribute
    independent of they class.
"""
for i in range(9):
    plt.clf()
    plt.hist(attributes[:, i], 100)
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, "ALL"), dpi=100,  bbox_inches="tight")

""" class conditional analysis: generate histogram for each attribute in each class.
"""
for i in range(9):
    for j in [1, 2, 3, 5, 6, 7]:
        plt.clf()
        x = index[labels == j]
        y = attributes[x - 1, i]
        plt.hist(y, 100)
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
        plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, j), dpi=100,  bbox_inches="tight")

""" Unconditional bi-variate analysis
"""
covTable = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        covTable[i, j] = np.corrcoef(attributes[:, i], attributes[:, j])[1, 0]

plt.matshow(covTable)
plt.xticks(range(9), ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])
plt.yticks(range(9), ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])
plt.savefig("FIGURE/Covariance_table.png", dpi=100, pad_inches=100)

plt.clf()
ok = []
for i in range(9):
    for j in range(9):
        if i != j and not ([i, j] in ok):
            plt.clf()
            for k in [1, 2, 3, 5, 6, 7]:
                plt.scatter(attributes[labels == k, i], attributes[labels == k, j], label="class {}".format(k),
                            color=COLOR[k], marker=MK[0])
            plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
            plt.xlabel("Predictor {}".format(i+1))
            plt.ylabel("Predictor {}".format(j+1))
            plt.savefig("FIGURE/att-{}_vs_{}_.png".format(i + 1, j + 1), dpi=300, bbox_inches="tight")
            ok.append([i, j])
            ok.append([j, i])

""" Unconditional multi-variate analysis
"""
mu = attributes.mean(axis=0)
sigma = attributes.std(axis=0)
attributes = (attributes - mu) / sigma

resultPca = PCA(attributes, standardize=False)
plt.clf()
result = np.dot(attributes, resultPca.Wt)

for j in [1, 2, 3, 5, 6, 7]:
    plt.scatter(resultPca.Y[labels == j, 0], resultPca.Y[labels == j, 1], label="class {}".format(j), marker=MK[0],
                color=COLOR[j])
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("FIGURE/scatter_PCA_.png", dpi=300, bbox_inches="tight")
plt.clf()
for i in range(2):
    for j in [1, 2, 3, 5, 6, 7]:
        plt.scatter(index[labels == j], attributes[labels == j, i], label="Pre. {} class {}".format(i + 1, j),
                    marker=MK[i], color=COLOR[j])
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig("FIGURE/scatter_PCA_before.png", dpi=300, bbox_inches="tight")
