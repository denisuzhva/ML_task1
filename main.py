import numpy as np
from linear_regressor import LinearRegressor
from trainer import Trainer


NUM_FEATURES = 53
LR = 1e-3
BATCH_LIST = [100, 200, 400, 800, 1600]
#BATCH_LIST = [8000]
NUM_EPOCHS = 10000
EPOCH_QUANTIZER = 10
NUM_FOLDS = 5
REG_GAMMA = 0.1


if __name__ == '__main__':
    train_dataset = np.load('./Dataset/FV1_ds.npy')
    train_labels = np.load('./Dataset/FV1_l.npy')

    linear_regressor = LinearRegressor(NUM_FEATURES, REG_GAMMA)
    trainer = Trainer(linear_regressor, LR, BATCH_LIST, NUM_EPOCHS, NUM_FOLDS, 
                      train_dataset, train_labels,
                      EPOCH_QUANTIZER)

    trainer.reduceTrainDataset()
    trainer.normalizeDatasets()
    train_loss_data, time_data = trainer.trainModel()

    np.save('./TrainDataProperRMSE/train_loss.npy', train_loss_data)
    np.save('./TrainDataProperRMSE/time.npy', time_data)
