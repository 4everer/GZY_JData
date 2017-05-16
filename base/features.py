#!/usr/bin/env python

from __future__ import division
import pandas as pd
import numpy as np

import base
from base.sampler import UserSampler, ActionSampler
import random

class TrainPreparer(ActionSampler):
    
    def __init__(self, user_data='data/user_simple.npy', 
                 action_data='data/action_simple.npy'):
        self.users = UserSampler(user_data)
        ActionSampler.__init__(self, action_data)
        self.num_user = len(self.users['user_id'])
        
    def get_user_action(self, header, action_type, start_time, end_time):
        raw_data = self.all_action[np.logical_and(self.__getitem__('time') > start_time,
                                                  self.__getitem__('time') < end_time), :]
        mask = raw_data[:, self.get_column('type')] == action_type
        action_time = raw_data[mask, self.get_column('time')]
        data = pd.DataFrame({'user_id': raw_data[mask, self.get_column('user_id')],
                             header: np.ones(np.sum(mask), dtype=int)})
        data_sum = data.groupby('user_id')[header].sum()
        return data_sum
    
    def get_all_data(self, start_time, end_time, division=10, buy_window=5 * 24 * 3600):
        interval = (end_time - start_time) * 1.0 / division
        col = self.users.columns.items()
        col.sort(key=lambda x: x[1])
        col = [_[0] for _ in col]
        data = pd.DataFrame(self.users.user_data, columns=col)
        data = data.set_index('user_id')
        
        for interval_id in range(division):
            for action in range(self.num_action):
                action_data = self.get_user_action('action{}_{}'.format(action, 
                                                                        interval_id), 
                                                   action + 1, 
                                                   start_time + interval * interval_id, 
                                                   start_time + interval * (interval_id + 1))
                data = data.join(action_data, how='outer')
        buy_data = self.get_user_action('buy', self.TYPE_BUY, end_time,
                                        end_time + buy_window)
        data = data.join(buy_data, how='outer')
        data[pd.isnull(data)] = 0
        
        train_label = np.array(data['buy'] > 0, dtype=int)
        train_data = data.drop('buy', 1).as_matrix()
        
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        train_data = (train_data - train_mean) / train_std
        return train_data, train_label
    
    def get_train_and_test(self, start_time, end_time, division=10, test_part=0.3):
        train, label = self.get_all_data(start_time, end_time, division)
        nrow = len(train)
        test_size = int(nrow * test_part)
        idx = random.sample(xrange(nrow), test_size)
        mask = np.ones(nrow, dtype=bool)
        mask[idx] = False
        
        return (train[mask, :], label[mask], train[idx, :], label[idx]) 


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print "usage: script [partition_num] [data_file_num]"
        sys.exit()
        
    print "Initializing..."
    tp = TrainPreparer()
    print 'Getting training data...'
    train_data, train_label, test_data, test_label = tp.get_train_and_test(0, 6e6, int(sys.argv[1]))
    print 'Saving data...'
    np.save('train_data' + sys.argv[2], train_data)
    np.save('train_label4' + sys.argv[2], train_label)
    np.save('test_data4' + sys.argv[2], test_data)
    np.save('test_label4' + sys.argv[2], test_label)
 
