import numpy as np
import csv


with open('./Dataset/Features_TestSet.csv', 'r') as dataset_csv:
    dataset_list = list(csv.reader(dataset_csv, delimiter=','))
dataset_np = np.array(dataset_list, dtype=np.float)

print(dataset_np.shape)