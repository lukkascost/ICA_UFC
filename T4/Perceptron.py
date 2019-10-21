import numpy as np


class Layered_perceptron_Logistic():
    def __init__(self, learning_rate, nclass):
        self.w = []
        self.wt = []
        self.et = []
        self.lr = learning_rate
        self.nclass = nclass

    def train(self, att, labels, epochs):
        self.wt = []
        self.et = []
        self.w = np.random.random((self.nclass, att.shape[1] + 1))
        self.wt.append(self.w)
        data = np.hstack((att, labels))
        for ep in range(epochs):
            erro = np.zeros((self.nclass, 1))
            np.random.shuffle(data)
            labels = data[:, -1]
            cpdata = data[:, :-1]
            for i in range(data.shape[0]):
                X = np.matrix(np.hstack(([-1], cpdata[i]))).T
                y = self.predict_intern(X)
                labels_new = np.zeros((self.nclass, 1))
                labels_new[int(labels[i])] = 1
                y_ = y - np.power(y, 2)
                e = labels_new - y
                self.w = self.w + (self.lr * np.multiply(e,y_) * X.T)
                erro += np.abs(e)
            self.et.append(erro)
            if max(erro) == 0:
                break
        print("EPOCA: ", ep + 1, "ERRO", erro, "W", self.w)

    def predict_intern(self, x):
        u = self.w * x
        return 1 / (1 + np.exp(-u))

    def predict(self, x):
        u = self.w * x
        y = 1 / (1 + np.exp(-u))
        return np.argmax(y)
