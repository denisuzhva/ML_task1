import numpy as np
from linear_regressor import LinearRegressor
from trainer import Trainer


NUM_FEATURES = 53
LR = 5 * 1e-3
#BATCH_LIST = [100, 200, 400, 800, 1600]
BATCH_LIST = [1600]
NUM_EPOCHS = 10000
EPOCH_QUANTIZER = 10
NUM_FOLDS = 5
REG_GAMMA = 0.1


if __name__ == '__main__':
    train_dataset = np.load('./Dataset/FV1_ds.npy')
    train_labels = np.load('./Dataset/FV1_l.npy')

    linear_regressorRMSE = LinearRegressor(NUM_FEATURES, REG_GAMMA, 'RMSE')
    linear_regressorR2 = LinearRegressor(NUM_FEATURES, REG_GAMMA, 'R2')
    trainer = Trainer(linear_regressorRMSE, LR, BATCH_LIST, NUM_EPOCHS, NUM_FOLDS,
                      train_dataset, train_labels,
                      EPOCH_QUANTIZER)

    trainer.reduceTrainDataset()
    trainer.normalizeDatasets()

    train_loss_dataRMSE, time_dataRMSE = trainer.trainModel()

    trainer.setModel(linear_regressorR2)
    train_loss_dataR2, time_dataR2 = trainer.trainModel()

    np.save('./TrainData/train_lossRMSE.npy', train_loss_dataRMSE)
    np.save('./TrainData/timeRMSE.npy', time_dataRMSE)

    np.save('./TrainData/train_lossR2.npy', train_loss_dataR2)
    np.save('./TrainData/timeR2.npy', time_dataR2)