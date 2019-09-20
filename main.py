import numpy as np
from linear_regressor import LinearRegressor
from trainer import Trainer


NUM_FEATURES = 53   # number of features
LR = 1 * 1e-1   # learning rate constant
BATCH_LIST = [100, 200, 400, 800, 1600] # batches to test
#BATCH_LIST = [1600]
NUM_EPOCHS = 1000   # number of epochs
EPOCH_QUANTIZER = 100   # write loss values each "EPOCH_QUANTIZER" time
NUM_FOLDS = 5   # number of folds
LOSSES_TO_WRITE = ['RMSE', 'R2']    # list of losses to write
REG_GAMMA = 0.1 # gamma parameter (for regularization)


if __name__ == '__main__':
    train_dataset = np.load('./Dataset/FV1_ds.npy')
    train_labels = np.load('./Dataset/FV1_l.npy')

    linear_regressor = LinearRegressor(NUM_FEATURES, REG_GAMMA)
    trainer = Trainer(linear_regressor, 
                      LR, BATCH_LIST, NUM_EPOCHS, NUM_FOLDS,
                      train_dataset, train_labels,
                      LOSSES_TO_WRITE,
                      EPOCH_QUANTIZER)

    trainer.reduceTrainDataset()    # reduce the number of data samples so it could be divisible by batch_size*num_folds 
    trainer.normalizeDatasets() # normalize data set
    trainer.shuffleDataset()    # shuffle data samples

    train_loss_data, train_weights_data, time_data = trainer.trainModel()

    np.save('./TrainData/train_loss.npy', train_loss_data)
    np.save('./TrainData/train_weights.npy', train_weights_data)
    np.save('./TrainData/time.npy', time_data)
