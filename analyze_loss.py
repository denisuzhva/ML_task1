import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



def plotLoss(loss_data, fold_num, is_rmse=True, is_train=True):
    epoch_data = np.arange(0, 1000, 10)
    plot_data = loss_data[:, fold_num, :, 0 if is_rmse else 1, 0 if is_train else 1]
    plt.figure(figsize=(12, 8))
    for batch_iter in range(plot_data.shape[0]):
        plt.plot(epoch_data, plot_data[batch_iter, :], label='BS = %d' % (100 * 2 ** (0+batch_iter)))
    
    plt.axis([0, 1000, 0, np.max(plot_data) * 1.2])

    plt.title('Loss on epoch (fold number: %d, %s loss)' % (fold_num, 'train' if is_train else 'validation'))
    plt.legend(fontsize=10)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    
    plt.show()


if __name__ == '__main__':
    loss_data = np.load('./TrainData/train_loss.npy')
    #plotLoss(loss_data, 2, True, True)
    RMSE_row = loss_data[0, :, -1, 0, 1]
    RMSE_mean = np.mean(RMSE_row)
    RMSE_std = np.std(RMSE_row)
    RMSE_row = np.append(RMSE_row, RMSE_mean)
    RMSE_row = np.append(RMSE_row, RMSE_std)
    RMSE_row = np.round(RMSE_row, 2)

    R2_row = loss_data[0, :, -1, 1, 1]
    R2_mean = np.mean(R2_row)
    R2_std = np.std(R2_row)
    R2_row = np.append(R2_row, R2_mean)
    R2_row = np.append(R2_row, R2_std)
    R2_row = np.round(R2_row, 2)

    loss_rows = np.stack((RMSE_row, R2_row), axis=0)
    RMSE_string = ' '.join(map(str, RMSE_row))
    R2_string = ' '.join(map(str, R2_row))
    print(RMSE_string)
    print(R2_string)

    np.savetxt('./TrainData/loss.csv', loss_rows, delimiter=' ', fmt='%.2f')

