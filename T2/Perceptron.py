import numpy as np


class Perceptron_Adaline():
    def __init__(self, learning_rate):
        self.w = []
        self.wt = []
        self.et = []
        self.lr = learning_rate

    def train(self, att, labels, epochs):
        self.wt = []
        self.et = []
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
                y = self.predict(X)[0, 0]
                e = labels[i] - y
                self.w = self.w + (self.lr * e * X)
                erro += abs(e)
            self.et.append(erro)
            if erro < 30:
                break
        print("EPOCA: ", ep + 1, "ERRO", erro, "W", self.w)

    def predict(self, x):
        u = self.w.T * x
        return u
