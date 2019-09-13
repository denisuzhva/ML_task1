import numpy as np
import time
from regressor import Regressor



class Trainer:
    def __init__(self, model, lr, batch_list, num_epochs, num_folds, 
                 train_dataset, train_labels,
                 losses_to_write=['RMSE'],
                 epoch_quantize_param=100,
                 regularize=False):
        self._model = model
        self._lr = lr
        self._batch_list = batch_list
        self._num_epochs = num_epochs
        self._epoch_quantize_param = epoch_quantize_param
        self._num_folds = num_folds
        self._losses_to_write = losses_to_write
        self._reg = regularize
        self._train_dataset = train_dataset
        self._train_labels = train_labels

        self._train_dataset_size = self._train_dataset.shape[0]
        self._num_features = self._train_dataset.shape[1]

        assert self._train_labels.shape[0] == self._train_dataset_size


    ## Setters
    def setModel(self, model):  # just in case
        self._model = model

    
    ## Dataset Tools
    def shuffleDataset(self):
        p = np.random.permutation(self._train_dataset_size)

        self._train_dataset = self._train_dataset[p, :]
        self._train_labels = self._train_labels[p]


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

        loss_tensor = np.zeros((len(self._batch_list),  # write at each batch
                                  self._num_folds,  # at each fold
                                  self._epoch_quantize_param,   # ...at each self._epoch_quantize_param's epoch
                                  len(self._losses_to_write),    # for all the losses to write
                                  2),   # for train and validation loss
                                  dtype=np.float)   

        time_tensor = np.zeros(len(self._batch_list), dtype=np.float)

        for batch_size_counter, batch_size in enumerate(self._batch_list, start=0):
            print('=== Current batch size: %d ===' % batch_size)
            batches_per_fold = fold_size // batch_size

            start_time = time.time()
            
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
                    for batch_iter in range(batches_per_fold):
                        #print('- Current batch: %d -' % batch_iter)
                        train_dataset_batch = train_folds[batch_iter*batch_size:(batch_iter+1)*batch_size, :]
                        train_labels_batch = train_labels[batch_iter*batch_size:(batch_iter+1)*batch_size]

                        self._model.updateParameters(train_labels_batch, 
                                                     train_dataset_batch, 
                                                     batch_size, 
                                                     self._lr,
                                                     self._reg)

                    if epoch_iter % (self._num_epochs // self._epoch_quantize_param) == 0:

                        for loss_counter, loss_type in enumerate(self._losses_to_write, start=0):
                            train_loss = self._model.evaluateLoss(train_labels, 
                                                                  train_folds,
                                                                  fold_size * 5,
                                                                  loss_type,
                                                                  self._reg)
                            val_loss = self._model.evaluateLoss(validation_labels, 
                                                                     validation_folds,
                                                                     fold_size,
                                                                     loss_type,
                                                                     self._reg)
                            assert ~np.isnan(train_loss)
                            assert ~np.isnan(val_loss)   
                            #print('train loss (%s): %f' % (loss_type, train_loss))
                            print('validation loss (%s): %f' % (loss_type, val_loss))
                            loss_tensor[batch_size_counter][fold_iter][quantized_epoch_iter][loss_counter][0] = train_loss
                            loss_tensor[batch_size_counter][fold_iter][quantized_epoch_iter][loss_counter][1] = val_loss
                        
                        quantized_epoch_iter += 1
                        
            end_time = time.time()
            time_tensor[batch_size_counter] = end_time - start_time

        return loss_tensor, time_tensor