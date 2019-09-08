import numpy as np
import abc



class Regressor(metaclass=abc.ABCMeta):
    def __init__(self, num_features):
        self._num_features = num_features


    ## Model Tools
    @abc.abstractmethod
    def getPrediction(self, x):
        pass


    @abc.abstractmethod
    def evaluateLossPerBatch(self, z_batch, x_batch, batch_size, regularize):
        pass


    @abc.abstractmethod
    def updateParameters(self, z_batch, x_batch, batch_size, lr):
        pass
    
