import numpy as np
import csv



def parseToNp(dataset_path, label_col):
    with open(dataset_path, 'r') as dataset_csv:
        dataset_list = list(csv.reader(dataset_csv, delimiter=','))
    dataset_np = np.array(dataset_list, dtype=np.float)
    labels_np = dataset_np[:, label_col]
    dataset_np = np.delete(dataset_np, label_col, 1)
    return dataset_np, labels_np


if __name__ == '__main__':
    dataset_name = 'Features_Variant_1.csv'
    dataset_dir = './Dataset/'
    label_col = 5
    dataset_np, labels_np = parseToNp(dataset_dir + dataset_name, label_col)
    np.save(dataset_dir + 'FV1_ds.npy', dataset_np)
    np.save(dataset_dir + 'FV1_l.npy', labels_np)