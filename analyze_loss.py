import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



def plotLoss(loss_data, fold_num, is_train=True):
    epoch_data = np.arange(0, 1000, 10)
    plot_data = loss_data[:, fold_num, :, 0 if is_train else 1]
    plt.figure(figsize=(12, 8))
    for batch_iter in range(plot_data.shape[0]):
        plt.plot(epoch_data, plot_data[batch_iter, :], label='BS = %d' % 2 ** (5+batch_iter))
    
    plt.axis([0, 1000, 0, np.max(plot_data) * 1.2])

    plt.title('RMSE loss on epoch (fold number: %d, %s loss)' % (fold_num, 'train' if is_train else 'validation'))
    plt.legend(fontsize=10)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE loss', fontsize=14)
    
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    
    plt.show()


if __name__ == '__main__':
    loss_data = np.load('./TrainDataProperRMSE/train_loss.npy')
    plotLoss(loss_data, 2, True)

