import numpy as np
import matplotlib.pyplot as plt

index = np.loadtxt("../glass.data.txt", usecols=0, delimiter=",", dtype=int)
attributes = np.loadtxt("../glass.data.txt", usecols=[x for x in range(1, 10)], delimiter=",")
labels = np.loadtxt("../glass.data.txt", usecols=-1, delimiter=",", dtype=int)
#
# """ Unconditional analysis: generate histogram for each attribute
#     independent of they class.
# """
# for i in range(9):
#     plt.clf()
#     plt.hist(attributes[:, i], 100)
#     plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, "ALL"), dpi=100)
#
# """ class conditional analysis: generate histogram for each attribute in each class.
# """
# for i in range(9):
#     for j in [1, 2, 3, 5, 6, 7]:
#         plt.clf()
#         x = index[labels == j]
#         y = attributes[x - 1, i]
#         plt.hist(y, 100)
#         plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, j), dpi=100)

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
            plt.scatter(index, attributes[:, i], label="Attribute {}".format(i+1))
            plt.scatter(index, attributes[:, j], label="Attribute {}".format(j+1))
            plt.legend()
            plt.savefig("FIGURE/att-{}_vs_{}_.png".format(i+1, j+1), dpi=100, pad_inches=100,)
            ok.append([i, j])
            ok.append([j, i])


""" Unconditional multi-variate analysis
"""
