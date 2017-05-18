#!/usr/bin/env python

from __future__ import division
import pandas as pd
import numpy as np

import base
import random
import forest

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print "Need to specify data file number"
        sys.exit()
        
    file_num = sys.argv[1]
    ft = forest.ForestTrainer()
    train_data = np.load('sku_train/train_data{}.npy'.format(file_num))
    test_data = np.load('sku_train/test_data{}.npy'.format(file_num))
    train_label = np.load('sku_train/train_label{}.npy'.format(file_num))
    test_label = np.load('sku_train/test_label{}.npy'.format(file_num))

    print "Partitions: ", (train_data.shape[1] - 2) / 4
    ft.train(train_data, train_label)
    print ft.clf_randomCV.best_params_
    test_predict = ft.predict(test_data)
    train_predict = ft.predict(train_data)

    print forest.score_func(train_label, train_predict)
    print forest.score_func(test_label, test_predict)
    ft.save('sku_model' + str(file_num))
