import numpy as np


class multi_Layered_perceptron_Logistic(object):
    def __init__(self, learning_rate, architecture):
        self.w_layers = []
        self.lr = learning_rate
        self.architecture = architecture
        self.et = []

    def train_classifier(self, att, labels, epochs):
        self._initialize_weights()
        input_data = np.hstack((att, labels))
        for i in range(epochs):
            np.random.shuffle(input_data)
            erro = 0
            labels = input_data[:, -1]
            cpdata = input_data[:, :-1]
            for input_index in range(input_data.shape[0]):
                layered_input = []
                layered_output = []
                ## Forward
                _input = np.matrix(np.hstack(([-1], cpdata[input_index]))).T
                for layer in range(len(self.architecture) - 1):
                    layered_input.append(_input)
                    current_y = self._activate_input(_input, layer)
                    layered_output.append(current_y)
                    _input = np.vstack(([-1], current_y))
                labels_new = np.zeros(current_y.shape)
                labels_new[int(labels[input_index])] = 1
                current_error = labels_new - current_y
                erro += np.sum(np.abs(current_error))
                ## Propagation
                for layer in range(len(self.architecture) - 2, -1, -1):
                    y_ = layered_output[layer] - np.power(layered_output[layer], 2)
                    self.w_layers[layer] = self.w_layers[layer] + self.lr * np.multiply(current_error, y_) * layered_input[layer].T
                    current_error = (np.multiply(current_error, y_).T * self.w_layers[layer][:,1:]).T
            self.et.append(erro)

    def _initialize_weights(self):
        for k in range(len(self.architecture) - 1):
            layer_w = (np.random.random((self.architecture[k + 1], self.architecture[k] + 1)))
            self.w_layers.append(layer_w)

    def _activate_input(self, _input, layer):
        u = self.w_layers[layer] * _input
        return 1 / (1 + np.exp(-u))

    def predict(self, input_data):
        _input = np.matrix(np.hstack(([-1], input_data))).T
        for layer in range(len(self.architecture) - 1):
            current_y = self._activate_input(_input, layer)
            _input = np.vstack(([-1], current_y))
        labels_new = np.zeros(current_y.shape)
        return np.argmax(current_y)

