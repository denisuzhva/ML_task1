import numpy as np
from Regressor import Regressor



class LinearRegressor(Regressor):
    def __init__(self, num_features):
        super().__init__(num_features)  # initialize parent fields
        self.__w = np.zeros(self._num_features, dtype=np.float)    # w vector
        self.__b = 0    # bias
        self.__acceptable_metrics = ['MSE', 'RMSE', 'R2']  # for asserts


    def getPrediction(self, x):
        prediction = np.dot(x, self.__w) + self.__b
        return prediction


    def updateParameters(self, x_batch, z_batch, batch_size, lr=0.01):  # dL / dw ?

        # d(RMSE) / dw = (1/2) * d(MSE) / RMSE
        loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
        loss = np.sqrt(loss)

        dw = -1 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / \
            (batch_size * loss)
        db = -1 * np.sum(z_batch - self.getPrediction(x_batch)) / \
            (batch_size * loss)

        self.__w -= lr * dw
        self.__b -= lr * db


    def resetWeights(self):
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__b = 0

    
    def getWeights(self):
        return self.__w, self.__b
