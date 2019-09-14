import numpy as np


if __name__ == '__main__':
    weight_data = np.load('./TrainData/train_weights.npy')
    weights = weight_data[0, :, -1]
    weights = weights.T

    
    
    weights_mean = np.mean(weights, axis=1)
    weights_mean = np.array(weights_mean)[np.newaxis]
    weights_mean = weights_mean.T
    weights_std = np.std(weights, axis=1)
    weights_std = np.array(weights_std)[np.newaxis]
    weights_std = weights_std.T
    weights = np.append(weights, weights_mean, axis=1)
    weights = np.append(weights, weights_std, axis=1)
    print(weights.shape)


    np.savetxt('./TrainData/weights.csv', weights, delimiter=' ', fmt='%.2f')