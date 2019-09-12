import numpy as np
from regressor import Regressor



class LinearRegressor(Regressor):
    def __init__(self, num_features, reg_gamma, loss_name='RMSE'):
        super().__init__(num_features)
        self.__w = np.zeros(self._num_features, dtype=np.float)    # w vector
        self.__b = 0    # bias
        self.__loss = loss_name
        self.__acceptable_loss = ['MSE', 'RMSE', 'R2']
        self.__reg_gamma = reg_gamma    # regularization coefficient

        assert self.__loss in self.__acceptable_loss


    def getPrediction(self, x):
        prediction = np.dot(x, self.__w) + self.__b
        return prediction


    def evaluateLossPerBatch(self, z_batch, x_batch, batch_size, regularize=False):
        if self.__loss == 'MSE':
            loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
            loss = loss + np.linalg.norm(self.__w, 2) if regularize else loss
            return loss
        elif self.__loss == 'RMSE':    
            loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
            loss = loss + np.linalg.norm(self.__w, 2) if regularize else loss
            return np.sqrt(loss)
        elif self.__loss == 'R2':
            MSE_loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
            MSE_loss = MSE_loss + np.linalg.norm(self.__w, 2) if regularize else MSE_loss
            loss = 1 - MSE_loss / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
            return loss


    def updateParameters(self, z_batch, x_batch, batch_size, lr=0.01, regularize=False):  # dL / dw ?
        if self.__loss == 'MSE':
            dw = -2 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / batch_size
            db = -2 * np.sum(z_batch - self.getPrediction(x_batch)) / batch_size
            self.__w -= lr * dw
            self.__b -= lr * db
        elif self.__loss == 'RMSE':
            dw = -1 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / \
                (batch_size * self.evaluateLossPerBatch(z_batch, x_batch, batch_size, regularize))
            db = -1 * np.sum(z_batch - self.getPrediction(x_batch)) / \
                (batch_size * self.evaluateLossPerBatch(z_batch, x_batch, batch_size, regularize))
            self.__w -= lr * dw
            self.__b -= lr * db
        elif self.__loss == 'R2':
            MSE_dw = -2 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / batch_size
            MSE_db = -2 * np.sum(z_batch - self.getPrediction(x_batch)) / batch_size
            dw = 1 - MSE_dw / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
            db = 1 - MSE_db / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
            self.__w -= lr * dw
            self.__b -= lr * db


    def resetWeights(self):
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__b = 0
