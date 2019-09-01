import numpy as np


class Perceptron():
    def __init__(self, learning_rate):
        self.w = []
        self.wt = []
        self.lr = learning_rate

    def train(self, att, labels, epochs):
        self.w = np.random.random((att.shape[1] + 1, 1))
        self.wt.append(self.w)
        data = np.hstack((att, labels))
        for ep in range(epochs):
            erro = 0
            np.random.shuffle(data)
            labels = data[:, -1]
            cpdata = data[:, :-1]
            for i in range(data.shape[0]):
                X = np.matrix(np.hstack(([-1], cpdata[i]))).T
                y = self.predict(X)
                e = int(labels[i]) - y
                self.w = self.w + (self.lr * e * X)
                erro += abs(e)
            if erro == 0:
                break
        print("EPOCA: ", ep + 1, "ERRO", erro, "W", self.w)


    def predict(self, x):
        u = self.w.T * x
        if u[0, 0] >= 0:
            return 1
        else:
            return 0
