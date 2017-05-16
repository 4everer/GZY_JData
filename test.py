#!/usr/bin/env python

from __future__ import division
import pandas as pd
import numpy as np

import base
import random
import forest
from forest.customScorer import *

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print "Need to specify data file number"
        sys.exit()
        
    file_num = sys.argv[1]
    ft = forest.ForestTrainer()
    train_data = np.load('train_data{}.npy'.format(file_num))
    test_data = np.load('test_data{}.npy'.format(file_num))
    train_label = np.load('train_label{}.npy'.format(file_num))
    test_label = np.load('test_label{}.npy'.format(file_num))

    ft.train(train_data, train_label)
    print ft.clf_randomCV.best_params_
    test_predict = ft.clf_randomCV.predict(test_data)
    train_predict = ft.clf_randomCV.predict(train_data)

    print JD_custom_score_func(train_label, train_predict)
    print JD_custom_score_func(test_label, test_predict)
