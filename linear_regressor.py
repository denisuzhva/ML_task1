import numpy as np
from regressor import Regressor



class LinearRegressor(Regressor):
    def __init__(self, num_features, reg_gamma):
        super().__init__(num_features)
        self.__w = np.zeros(self._num_features, dtype=np.float)    # w vector
        self.__b = 0    # bias
        self.__acceptable_loss = ['MSE', 'RMSE', 'R2']
        self.__reg_gamma = reg_gamma    # regularization coefficient


    def getPrediction(self, x):
        prediction = np.dot(x, self.__w) + self.__b
        return prediction


    def evaluateLoss(self, z_batch, x_batch, batch_size, loss_type='RMSE', regularize=False):
        assert loss_type in self.__acceptable_loss
        if loss_type == 'MSE':
            loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
            loss = loss + np.linalg.norm(self.__w, 2) if regularize else loss
            return loss
        elif loss_type == 'RMSE':    
            loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
            loss = loss + np.linalg.norm(self.__w, 2) if regularize else loss
            return np.sqrt(loss)
        elif loss_type == 'R2':
            MSE_loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
            #MSE_loss = MSE_loss + np.linalg.norm(self.__w, 2) if regularize else MSE_loss
            loss = 1 - MSE_loss / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
            return loss


    def updateParameters(self, z_batch, x_batch, batch_size, lr=0.01, loss_type='RMSE', regularize=False):  # dL / dw ?
        assert loss_type in self.__acceptable_loss
        if loss_type == 'MSE':
            dw = -2 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / batch_size
            db = -2 * np.sum(z_batch - self.getPrediction(x_batch)) / batch_size
            self.__w -= lr * dw
            self.__b -= lr * db
        elif loss_type == 'RMSE':
            dw = -1 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / \
                (batch_size * self.evaluateLoss(z_batch, x_batch, batch_size, 'RMSE', regularize))
            db = -1 * np.sum(z_batch - self.getPrediction(x_batch)) / \
                (batch_size * self.evaluateLoss(z_batch, x_batch, batch_size, 'RMSE', regularize))
            self.__w -= lr * dw
            self.__b -= lr * db
        elif loss_type == 'R2':
            MSE_dw = -2 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / batch_size
            MSE_db = -2 * np.sum(z_batch - self.getPrediction(x_batch)) / batch_size
            dw = -1 * MSE_dw / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
            db = -1 * MSE_db / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
            self.__w += lr * dw
            self.__b += lr * db


    def resetWeights(self):
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__b = 0

    
    def getWeights(self):
        return self.__w, self.__b
