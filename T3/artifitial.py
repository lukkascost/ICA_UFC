import matplotlib.pyplot as plt

import numpy as np

X11 = np.random.random(50)
X21 = np.random.random(50)+2
c1 = np.ones(50)*0
d1 = np.vstack((X11,X21,c1)).T

X12 = np.random.random(50)+1
X22 = np.random.random(50)
c2 = np.ones(50)
d2 = np.vstack((X12,X22,c2)).T



X13 = np.random.random(50)+2
X23 = np.random.random(50)+2
c3 = np.ones(50)*2
d3 = np.vstack((X13,X23,c3)).T


data = np.vstack((d1,d2,d3))
np.savetxt("Datasets/artifitial1.data", data, delimiter=",")
plt.scatter(X11,X21, marker=">")
plt.scatter(X12,X22, marker=".")
plt.scatter(X13,X23, marker="s")
plt.show()