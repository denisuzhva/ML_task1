import numpy as np
from linear_regressor import LinearRegressor
from trainer import Trainer


NUM_FEATURES = 53
LR = 1 * 1e-1
BATCH_LIST = [100, 200, 400, 800, 1600]
#BATCH_LIST = [1600]
NUM_EPOCHS = 1000
EPOCH_QUANTIZER = 100
NUM_FOLDS = 5
LOSSES_TO_WRITE = ['RMSE', 'R2']
REG_GAMMA = 0.1


if __name__ == '__main__':
    train_dataset = np.load('./Dataset/FV1_ds.npy')
    train_labels = np.load('./Dataset/FV1_l.npy')

    linear_regressor = LinearRegressor(NUM_FEATURES, REG_GAMMA)
    trainer = Trainer(linear_regressor, 
                      LR, BATCH_LIST, NUM_EPOCHS, NUM_FOLDS,
                      train_dataset, train_labels,
                      LOSSES_TO_WRITE,
                      EPOCH_QUANTIZER)

    trainer.reduceTrainDataset()
    trainer.normalizeDatasets()

    train_loss_data, train_weights_data, time_data = trainer.trainModel()

    np.save('./TrainData/train_loss.npy', train_loss_data)
    np.save('./TrainData/train_weights.npy', train_weights_data)
    np.save('./TrainData/time.npy', time_data)
