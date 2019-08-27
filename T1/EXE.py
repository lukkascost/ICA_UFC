from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from T1.Perceptron import Perceptron

COLOR = cm.rainbow(np.linspace(0, 1, 5))

mu, sigma = 1, 0.05
mu2, sigma2 = 0, 0.05
qtd = 500

learning_rate = 0.01
epochs = 200000

s = np.random.normal(mu, sigma, qtd)
s2 = np.random.normal(mu, sigma, qtd)
c1 = np.column_stack((s, s2, np.ones(qtd) * 1))

s3 = np.random.normal(mu2, sigma2, int(qtd / 4))
s4 = np.random.normal(mu, sigma2, int(qtd / 4))
c2 = np.column_stack((s3, s4, np.ones(int(qtd / 4)) * 0))

s5 = np.random.normal(mu, sigma2, int(qtd / 4))
s6 = np.random.normal(mu2, sigma2, int(qtd / 4))
c3 = np.column_stack((s5, s6, np.ones(int(qtd / 4)) * 0))

s7 = np.random.normal(mu2, sigma2, int(qtd / 4))
s8 = np.random.normal(mu2, sigma2, int(qtd / 4))
c4 = np.column_stack((s7, s8, np.ones(int(qtd / 4)) * 0))

data = np.row_stack((c1, c2, c3, c4))
print(data.shape)

# plt.scatter(s, s2, color=COLOR[1])
# plt.scatter(s3, s4, color=COLOR[0])
# plt.scatter(s5, s6, color=COLOR[0])
# plt.scatter(s7, s8, color=COLOR[0])

perc = Perceptron(learning_rate)
perc.train(data, epochs)

plt.yticks([0,1])
plt.yticks([0,1])
print(perc.w)
result = []
for i in range(101):
    for j in range(101):
        result.append([i/100, j/100, perc.predict(np.matrix([[-1], [i/100], [j/100]]))])
        plt.scatter(result[-1][0], result[-1][1], color=COLOR[result[-1][2]], linewidths=0.1, marker="s")
    print(i)

plt.show()
