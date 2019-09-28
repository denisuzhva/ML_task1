import numpy as np



def regressor(fx_batch, z_batch, metric_type='RMSE'):

    acceptable_metrics = ['MSE', 'RMSE', 'R2']
    assert metric_type in acceptable_metrics

    batch_size = fx_batch.shape[0]
    
    if metric_type == 'MSE':
        metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
        return metric
    elif metric_type == 'RMSE':    
        metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
        return np.sqrt(metric)
    elif metric_type == 'R2':
        mse_metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
        metric = 1 - mse_metric / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
        return metric


## with regularization
def regressorReg(fx_batch, z_batch, weights, metric_type='RMSE', order=2):

    acceptable_metrics = ['MSE', 'RMSE', 'R2']
    assert metric_type in acceptable_metrics
    
    batch_size = fx_batch.shape[0]

    if metric_type == 'MSE':
        metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
        metric = metric + np.linalg.norm(weights, order)
        return metric
    elif metric_type == 'RMSE':    
        metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
        metric = metric + np.linalg.norm(weights, order)
        return np.sqrt(metric)
    elif metric_type == 'R2':
        mse_metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
        mse_metric = mse_metric + np.linalg.norm(weights, order)
        metric = 1 - mse_metric / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
        return metric

