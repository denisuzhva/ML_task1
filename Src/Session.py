import numpy as np
import time
import DatasetToolbox as dt
import Evaluator as eva
import Optimizer as opt



class Session:
    def __init__(self):
        pass


    ## perform cross-validation
    def crossValidation(self, model, 
                        dataset, labels,
                        epochs, epoch_quant,
                        batch_size,
                        num_folds,
                        learning_rate,
                        metric_list):

        dataset_size = dataset.shape[0]
        num_features = dataset.shape[1]

        metrics_tensor = np.zeros((num_folds,
                                   epoch_quant,
                                   len(metric_list),
                                   2),
                                  dtype=float)  

        weight_tensor = np.zeros((num_folds,
                                  epoch_quant,
                                  num_features + 1),
                                 dtype=float)

        time_var = 0

        start_time = time.time()

        for fold_iter in range(num_folds):
            print('== Current validation fold: %d ==' % fold_iter)
            model.resetWeights()
            
            start_index = (dataset_size // num_folds) * fold_iter
            end_index = (dataset_size // num_folds) * (fold_iter+1)

            train_folds, train_labels = dt.makeFolds(dataset, labels, 
                                                     start_index, end_index, 
                                                     is_train=True)
            val_folds, val_labels = dt.makeFolds(dataset, labels, 
                                                 start_index, end_index, 
                                                 is_train=False)

            epoch_quant_iter = 0            
            for epoch_iter in range(epochs):

                opt.optimize(model, train_folds, train_labels, batch_size, learning_rate)

                if epoch_iter % (epochs // epoch_quant) == 0:

                    for metric_counter, metric_type in enumerate(metric_list, start=0):
                        train_pred = model.getPrediction(train_folds)
                        val_pred = model.getPrediction(val_folds)

                        train_metric = eva.regressor(train_labels, 
                                                     train_pred,
                                                     metric_type)
                        val_metric = eva.regressor(val_labels, 
                                                   val_pred,
                                                   metric_type)
                        assert ~np.isnan(train_metric)
                        assert ~np.isnan(val_metric)   

                        metrics_tensor[fold_iter][epoch_quant_iter][metric_counter][0] = train_metric
                        metrics_tensor[fold_iter][epoch_quant_iter][metric_counter][1] = val_metric 
                   
                    model_w, model_b = model.getWeights()
                    model_w_full = np.append(model_w, model_b)
                    weight_tensor[fold_iter][epoch_quant_iter] = model_w_full

                    epoch_quant_iter  += 1

        end_time = time.time()
        time_var = end_time - start_time

        return metrics_tensor, weight_tensor, time_var

