import numpy as np
import time



class Trainer:
    def __init__(self, model, lr, batch_list, num_epochs, num_folds, 
                 train_dataset, train_labels,
                 metrics_to_write=['RMSE'],
                 epoch_quantize_param=100,
                 regularize=False):
        self._model = model # a model (linear regressor in our case)
        self._lr = lr   # learning rate
        self._batch_list = batch_list   # list with batch sizes
        self._num_epochs = num_epochs   # number of epochs
        self._epoch_quantize_param = epoch_quantize_param   # write loss values each "epoch_quantize_param" time
        self._num_folds = num_folds # number of folds 
        self._metrics_to_write = metrics_to_write # list with names of metric functions
        self._reg = regularize  # set True if regularize
        self._train_dataset = train_dataset # dataset matrix
        self._train_labels = train_labels   # label vector

        self._train_dataset_size = self._train_dataset.shape[0] # number of data samples
        self._num_features = self._train_dataset.shape[1]   # number of features

        assert self._train_labels.shape[0] == self._train_dataset_size


    ## Setters
    def setModel(self, model):  # just in case
        self._model = model

    
    ## Dataset Tools
    def shuffleDataset(self):
        p = np.random.permutation(self._train_dataset_size)

        self._train_dataset = self._train_dataset[p, :]
        self._train_labels = self._train_labels[p]

    
    def shuffleBatch(self, z_batch, x_batch, batch_size):
        p = np.random.permutation(batch_size)

        x_batch = x_batch[p, :]
        z_batch = z_batch[p]

        return z_batch, x_batch


    def reduceTrainDataset(self):  # make its size divisible by batch_size*num_folds
        multiple_size = self._train_dataset_size // \
                        (self._batch_list[-1] * self._num_folds) * \
                         self._batch_list[-1] * self._num_folds

        self._train_dataset = self._train_dataset[0:multiple_size, :]
        self._train_labels = self._train_labels[0:multiple_size]
        self._train_dataset_size = multiple_size


    def normalizeDatasets(self):
        fold_size = self._train_dataset_size // self._num_folds
        for fold_iter in range(self._num_folds):
            for feature_iter in range(self._num_features):
                f_mean = np.mean(self._train_dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, 
                                                     feature_iter])
                f_std = np.std(self._train_dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, 
                                                   feature_iter])
                f_max = np.max(self._train_dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, 
                                                   feature_iter])
                if not f_std == 0:
                    self._train_dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, feature_iter] -= f_mean
                    self._train_dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, feature_iter] /= f_std
                elif not f_max == 0:
                    self._train_dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, feature_iter] /= f_max
                

    ## Training Algo
    def trainModel(self):
        fold_size = self._train_dataset_size // self._num_folds

        metrics_tensor = np.zeros((len(self._batch_list),  # write at each batch
                                   self._num_folds,  # at each fold
                                   self._epoch_quantize_param,   # ...at each self._epoch_quantize_param's epoch
                                   len(self._metrics_to_write),    # for all the metrics to write
                                   2),   # for train and validation metric
                                   dtype=np.float)  

        weight_tensor = np.zeros((len(self._batch_list),
                                  self._num_folds,
                                  self._epoch_quantize_param,
                                  self._num_features + 1),  # w1..wN, w0; N = 53
                                  dtype=np.float) 

        time_tensor = np.zeros(len(self._batch_list), dtype=np.float)   # vector of time values (for each batch size)

        for batch_size_counter, batch_size in enumerate(self._batch_list, start=0):
            print('=== Current batch size: %d ===' % batch_size)
            batches_per_fold = fold_size // batch_size

            start_time = time.time()    # start timer
            
            for fold_iter in range(self._num_folds):
                print('== Current validation fold: %d ==' % fold_iter)
                self._model.resetWeights()
                
                start_index = fold_size * fold_iter
                end_index = fold_size * (fold_iter+1)
                train_folds = np.delete(self._train_dataset, 
                                        slice(start_index, end_index),
                                        axis=0)
                validation_folds = self._train_dataset[start_index:end_index, :]
                
                train_labels = np.delete(self._train_labels, 
                                         slice(start_index, end_index),
                                         axis=0)
                validation_labels = self._train_labels[start_index:end_index]            

                quantized_epoch_iter = 0            
                for epoch_iter in range(self._num_epochs):
                    train_labels, train_folds = self.shuffleBatch(train_labels,
                                                                  train_folds,
                                                                  fold_size * 4)
                    validation_labels, validation_folds = self.shuffleBatch(validation_labels,
                                                                            validation_folds,
                                                                            fold_size)

                    for batch_iter in range(batches_per_fold):
                        #print('- Current batch: %d -' % batch_iter)
                        train_dataset_batch = train_folds[batch_iter*batch_size:(batch_iter+1)*batch_size, :]
                        train_labels_batch = train_labels[batch_iter*batch_size:(batch_iter+1)*batch_size]

                        self._model.updateParameters(train_labels_batch, 
                                                     train_dataset_batch, 
                                                     batch_size, 
                                                     self._lr,
                                                     'RMSE',
                                                     self._reg)

                    if epoch_iter % (self._num_epochs // self._epoch_quantize_param) == 0:

                        for metric_counter, metric_type in enumerate(self._metrics_to_write, start=0):
                            train_metric = self._model.evaluateMetric(train_labels, 
                                                                      train_folds,
                                                                      fold_size * 4,
                                                                      metric_type,
                                                                      self._reg)
                            val_metric = self._model.evaluateMetric(validation_labels, 
                                                                    validation_folds,
                                                                    fold_size,
                                                                    metric_type,
                                                                    self._reg)
                            assert ~np.isnan(train_metric)
                            assert ~np.isnan(val_metric)   
                            #print('train loss (%s): %f' % (metric_type, train_metric))
                            #print('validation loss (%s): %f' % (metric_type, val_metric))
                            metrics_tensor[batch_size_counter][fold_iter][quantized_epoch_iter][metric_counter][0] = train_metric
                            metrics_tensor[batch_size_counter][fold_iter][quantized_epoch_iter][metric_counter][1] = val_metric
                        
                        model_w, model_b = self._model.getWeights()
                        model_w_full = np.append(model_w, model_b)
                        weight_tensor[batch_size_counter][fold_iter][quantized_epoch_iter] = model_w_full

                        quantized_epoch_iter += 1
                        
            end_time = time.time()  # end timer
            time_tensor[batch_size_counter] = end_time - start_time

        return metrics_tensor, weight_tensor, time_tensor