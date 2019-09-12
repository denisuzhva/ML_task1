import numpy as np
from regressor import Regressor



class LinearRegressor(Regressor):
    def __init__(self, num_features, reg_gamma):    # add loss name argument
        super().__init__(num_features)
        self.__w = np.zeros(self._num_features, dtype=np.float)    # w vector
        self.__b = 0    # bias
        self.__loss = 'RMSE'
        self.__acceptable_loss = ['MSE', 'RMSE', 'R2']
        self.__reg_gamma = reg_gamma    # regularization coefficient


    def setLoss(self, loss_name):   # delet this
        if not loss_name in self.__acceptable_loss:
            raise Exception('Invalid loss function name')


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
            loss = 1
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
            pass


    def resetWeights(self):
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__b = 0
