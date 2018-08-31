import numpy as np
import matplotlib.pyplot as plt

index = np.loadtxt("../glass.data.txt", usecols=0, delimiter=",", dtype=int)
attributes = np.loadtxt("../glass.data.txt", usecols=[x for x in range(1, 10)], delimiter=",")
labels = np.loadtxt("../glass.data.txt", usecols=-1, delimiter=",", dtype=int)
# for i in range(9):
#     plt.clf()
#     plt.scatter(index, attributes[:, i], marker=".")
#     plt.savefig("FIGURE/sc_att-{:02d}_cls-{}.png".format(i + 1, "ALL"), dpi=100)
#
# for i in range(9):
#     for j in [1, 2, 3, 5, 6, 7]:
#         plt.clf()
#         x = index[labels == j]
#         y = attributes[x-1, i]
#         plt.scatter(x, y, marker=".")
#         plt.savefig("FIGURE/sc_att-{:02d}_cls-{}.png".format(i + 1, j), dpi=100)

for i in range(9):
    plt.clf()
    plt.hist(attributes[:, i], 100)
    plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, "ALL"), dpi=100)
