import numpy as np
from linear_regressor import LinearRegressor
from trainer import Trainer


NUM_FEATURES = 53
LR = 1e-2
BATCH_SIZE = 8000
NUM_EPOCHS = 10000
NUM_FOLDS = 5
REG_GAMMA = 0.1


if __name__ == '__main__':
    train_dataset = np.load('./Dataset/FV1_ds.npy')
    train_labels = np.load('./Dataset/FV1_l.npy')

    linear_regressor = LinearRegressor(NUM_FEATURES, REG_GAMMA)
    trainer = Trainer(linear_regressor, LR, BATCH_SIZE, NUM_EPOCHS, NUM_FOLDS, 
                      train_dataset, train_labels,
                      train_dataset, train_labels)

    trainer.normalizeDatasets()
    trainer.reduceTrainDataset()
    trainer.trainModel()