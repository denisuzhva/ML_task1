import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



def prepareRow(row):
    row_mean = np.mean(row)
    row_std = np.std(row)
    row = np.append(row, row_mean)
    row = np.append(row, row_std)
    row = np.round(row, 2)
    print(row.shape)
    return row


def plotLoss(metric_data, fold_num, is_rmse=True, is_train=True):
    epoch_data = np.arange(0, 1000, 10)
    plot_data = metric_data[:, fold_num, :, 0 if is_rmse else 1, 0 if is_train else 1]
    plt.figure(figsize=(12, 8))
    for batch_iter in range(plot_data.shape[0]):
        plt.plot(epoch_data, plot_data[batch_iter, :], label='BS = %d' % (100 * 2 ** (0+batch_iter)))
    
    plt.axis([0, 1000, 0, np.max(plot_data) * 1.2])

    plt.title('Metric on epoch (fold number: %d, %s loss)' % (fold_num, 'train' if is_train else 'validation'))
    plt.legend(fontsize=10)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Metric value', fontsize=14)
    
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    
    plt.show()


if __name__ == '__main__':
    metric_data = np.load('./TrainData/metrics.npy')
    #plotLoss(metric_data, 2, True, False)
    
    batch = 0
    rows = [metric_data[batch, :, -1, 0, 1],   # RMSE val 
            metric_data[batch, :, -1, 0, 0],   # RMSE train 
            metric_data[batch, :, -1, 1, 1],   # R2 val
            metric_data[batch, :, -1, 1, 0]   # R2 train
            ]

    for row_count, row  in enumerate(rows, start=0):
        rows[row_count] = prepareRow(row)


    loss_rows = np.stack((rows[0], 
                          rows[1], 
                          rows[2],
                          rows[3]), axis=0)
    RMSE_val_string = ' '.join(map(str, rows[0]))
    RMSE_train_string = ' '.join(map(str, rows[1]))
    R2_val_string = ' '.join(map(str, rows[2]))
    R2_train_string = ' '.join(map(str, rows[3]))


    np.savetxt('./TrainData/metrics.csv', loss_rows, delimiter=' ', fmt='%.2f')

