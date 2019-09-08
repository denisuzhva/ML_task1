import numpy as np
from regressor import Regressor



class Trainer:
    def __init__(self, model, lr, batch_size, num_epochs, num_folds, 
                 train_dataset, train_labels,
                 test_dataset, test_labels):
        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._num_folds = num_folds
        self._train_dataset = train_dataset
        self._train_labels = train_labels
        self._test_dataset = test_dataset
        self._test_labels = test_labels

        self._train_dataset_size = self._train_dataset.shape[0]
        self._test_dataset_size = self._test_dataset.shape[0]
        self._num_features = self._train_dataset.shape[1]

        assert self._train_labels.shape[0] == self._train_dataset_size
        assert self._test_labels.shape[0] == self._test_dataset_size

    
    ## Dataset Tools
    def shuffleDataset(self):
        p = np.random.permutation(self._train_dataset_size)

        self._train_dataset = self._train_dataset[p, :]
        self._train_labels = self._train_labels[p]


    def normalizeDatasets(self):
        
        for feature_iter in range(self._num_features):
            f_mean = np.mean(self._train_dataset[:, feature_iter])
            f_std = np.std(self._train_dataset[:, feature_iter])
            

            self._train_dataset[:, feature_iter] -= f_mean
            self._test_dataset[:, feature_iter] -= f_mean
            if not f_std == 0:
                self._train_dataset[:, feature_iter] /= f_std
                self._test_dataset[:, feature_iter] /= f_std



    def reduceTrainDataset(self):  # make its size divisible by batch_size*num_folds
        multiple_size = self._train_dataset_size // \
                        (self._batch_size * self._num_folds) * \
                        self._batch_size * self._num_folds

        self._train_dataset = self._train_dataset[0:multiple_size, :]
        self._train_labels = self._train_labels[0:multiple_size]


    ## Training Algo
    def trainModel(self):
        fold_size = self._train_dataset_size // self._num_folds
        batches_per_fold = fold_size // self._batch_size
        
        for fold_iter in range(self._num_folds):
            print('=== Current validation fold: %d ===' % fold_iter)

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
        
            for epoch_iter in range(self._num_epochs):

                for batch_iter in range(batches_per_fold):
                    #print('== Current batch: %d ==' % batch_iter)
                    train_dataset_batch = train_folds[batch_iter*self._batch_size:(batch_iter+1)*self._batch_size, :]
                    train_labels_batch = train_labels[batch_iter*self._batch_size:(batch_iter+1)*self._batch_size]
                    self._model.updateParameters(train_labels_batch, 
                                                train_dataset_batch, 
                                                self._batch_size, 
                                                self._lr)


                if epoch_iter % (self._num_epochs // 10) == 0:
                    print('== Current epoch: %d ==' % epoch_iter)
                    train_loss = self._model.evaluateLossPerBatch(train_labels, 
                                                                train_folds,
                                                                self._batch_size)
                    val_loss = self._model.evaluateLossPerBatch(validation_labels, 
                                                                validation_folds,
                                                                self._batch_size)
                    print('train loss: %f' % train_loss)
                    print('validation loss: %f' % val_loss)
                    
