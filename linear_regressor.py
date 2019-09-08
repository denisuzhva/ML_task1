import numpy as np
from regressor import Regressor



class LinearRegressor(Regressor):
    def __init__(self, num_features, reg_gamma):
        super().__init__(num_features)
        self.__w = np.zeros(self._num_features, dtype=np.float)    # w vector
        self.__b = 0    # bias
        self.__reg_gamma = reg_gamma    # regularization coefficient


    def getPrediction(self, x):
        prediction = np.dot(x, self.__w) + self.__b
        return prediction


    def evaluateLossPerBatch(self, z_batch, x_batch, batch_size, regularize=False): # RMSE
        loss = np.sum((z_batch - np.dot(x_batch, self.__w) - self.__b) ** 2) / batch_size
        loss = loss + np.linalg.norm(self.__w, 2) if regularize else loss
        return np.sqrt(loss)

    
    def updateParameters(self, z_batch, x_batch, batch_size, lr):  # ...for the next time step
        dw = -2 * np.dot((z_batch - np.dot(x_batch, self.__w) - self.__b), x_batch) / batch_size
        db = -2 * np.sum(z_batch - np.dot(x_batch, self.__w) - self.__b) / batch_size
        self.__w -= lr * dw
        self.__b -= lr * db
