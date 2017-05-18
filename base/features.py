#!/usr/bin/env python

from __future__ import division
import pandas as pd
import numpy as np

import base
from base import User, Action, Product, Comment
import random

class UserTrainSet(Action):
    
    def __init__(self, user_data='data/user_simple.npy', 
                 action_data='data/action_simple.npy'):
        self.users = User(user_data)
        Action.__init__(self, action_data)

    def get_user_action(self, header, action_type, start_time, end_time):
        raw_data = self.data[np.logical_and(self['time'] > start_time,
                                                  self['time'] < end_time), :]
        mask = raw_data[:, self.get_column('type')] == action_type
        data = pd.DataFrame({'user_id': raw_data[mask, self.get_column('user_id')],
                             header: np.ones(np.sum(mask), dtype=int)})
        data_sum = data.groupby('user_id')[header].sum()
        return data_sum
    
    def get_all_data(self, start_time, end_time, 
                     division=10, buy_window=5 * 24 * 3600):
        interval = (end_time - start_time) * 1.0 / division
        col = self.users.columns.items()
        col.sort(key=lambda x: x[1])
        col = [_[0] for _ in col]
        data = pd.DataFrame(self.users.data, columns=col)
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
        return train_data, train_label, data.index
    
    def get_train_and_test(self, start_time, end_time, division=10, test_part=0.3):
        train, label, idx = self.get_all_data(start_time, end_time, division)
        nrow = len(train)
        test_size = int(nrow * test_part)
        idx = random.sample(xrange(nrow), test_size)
        mask = np.ones(nrow, dtype=bool)
        mask[idx] = False
        
        return (train[mask, :], label[mask], train[idx, :], label[idx]) 

class ProductUserSet(Action):
    
    def __init__(self, product_data='data/comment_simple.npy',
                 action_data='data/action_simple.npy',
                 product_cate=None):
        self.comments = Comment(product_data)
        Action.__init__(self, action_data)
        self.user_max = np.max(self['user_id'])
        if product_cate is not None:
            self.data = self.data[self['cate'] == product_cate, :]
        self.user_sku = self['sku_id'] * self.user_max + self['user_id']
        
        decart = np.array(self['type'] == self.TYPE_DECART, dtype=int)
        self.cart = np.array(self['type'] == self.TYPE_CART, dtype=int)
        self.cart -= decart
        
        self.comment_time = np.unique(self.comments['dt'])

    def get_user_sku_action(self, header, action_type, start_time, end_time):
        mask = np.logical_and(self['time'] > start_time,
                              self['time'] < end_time)
        user_sku = self.user_sku[mask]
        action_data = self.data[mask, self.get_column('type')]
        mask = action_data == action_type
        data = pd.DataFrame({'user_sku': user_sku[mask],
                             header: np.ones(np.sum(mask), dtype=int)})
        data_sum = data.groupby('user_sku')[header].sum()
        return data_sum
    
    def get_user_sku_cart(self, header, start_time, end_time, first=False):
        mask = np.logical_and(self['time'] > start_time,
                              self['time'] < end_time)
        user_sku = self.user_sku[mask]
        action_data = self.cart[mask]
        mask = action_data != 0
        data = pd.DataFrame({'user_sku': user_sku[mask],
                             header: action_data[mask]})
        data_group = data.groupby('user_sku')[header] 
        data_sum = data_group.sum()
        if first:
            data_check = np.array(data_group.first() == -1, dtype=int)
            data_sum += data_check
        return data_sum
    
    def get_user_sku_comment(self, end_time):
        till_time = np.max(self.comment_time)
        for time in self.comment_time:
            if  end_time < time:
                till_time = time
                break
        comments = self.comments.data[np.logical_and(self.comments['dt'] < till_time + 1,
                                                     self.comments['dt'] > till_time - 1),
                                       :]
        col = self.comments.columns.items()
        col.sort(key=lambda x: x[1])
        col = [_[0] for _ in col]
        comments = pd.DataFrame(comments, columns=col)
        comments = comments.drop('dt', 1)
        comments = comments.set_index('sku_id')
        
        return comments
    
    def get_all_data(self, start_time, end_time, 
                     division=10, buy_window=5 * 24 * 3600):
        interval = (end_time - start_time) * 1.0 / division
        data = None
        for interval_id in range(division):
            if data is None:
                cart_data = self.get_user_sku_cart('cart' + str(interval_id),
                                     start_time + interval * interval_id,
                                     start_time + interval * (interval_id + 1),
                                     True)
                data = pd.DataFrame(cart_data)
            else:
                cart_data = self.get_user_sku_cart('cart' + str(interval_id),
                                     start_time + interval * interval_id,
                                     start_time + interval * (interval_id + 1))
                data = data.join(cart_data, how='outer')
            
            for action in [self.TYPE_VIEW, self.TYPE_BUY, self.TYPE_FAVOR]:
                action_data = self.get_user_sku_action('action{}_{}'.format(action, 
                                                                  interval_id), 
                                    action, 
                                    start_time + interval * interval_id, 
                                    start_time + interval * (interval_id + 1))
                data = data.join(action_data, how='outer')
        comment_data = self.get_user_sku_comment(end_time)
        sku_list = data.index / self.user_max
        data['comment_num_end'] = np.array(comment_data['comment_num'][sku_list])
        data['bad_comment_rate_end'] = np.array(comment_data['bad_comment_rate'][sku_list])
        
        buy_data = self.get_user_sku_action('buy', self.TYPE_BUY, end_time,
                                        end_time + buy_window)
        data = data.join(buy_data, how='outer')
        
        
        data[pd.isnull(data)] = 0
        
        train_label = np.array(data['buy'] > 0, dtype=int)
        train_data = data.drop('buy', 1).as_matrix()
        
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        train_data = (train_data - train_mean) / train_std
        return train_data, train_label, data.index
    
    def get_train_and_test(self, start_time, end_time, division=10, test_part=0.3):
        train, label, idx = self.get_all_data(start_time, end_time, division)
        nrow = len(train)
        test_size = int(nrow * test_part)
        idx = random.sample(xrange(nrow), test_size)
        mask = np.ones(nrow, dtype=bool)
        mask[idx] = False
        
        return (train[mask, :], label[mask], train[idx, :], label[idx])


class ProductTrainSet(Action):
    
    def __init__(self, product_data='data/product_simple.npy', 
                 action_data='data/action_simple.npy'):
        self.products = Product(product_data)
        Action.__init__(self, action_data)
        self.num_product = len(self.products['sku_id'])
        
    def get_all_data(self):
        buy_data = self.data[np.logical_and(self['type'] == self.TYPE_BUY,
                                            self['cate'] == self.products.cate),
                              self.get_column('sku_id')]
        label = np.array([_ in buy_data for _ in self.products['sku_id']], 
                         dtype=int)
        return self.products.data[:, :-1], label
    
    def get_train_and_test(self, test_part=0.3):
        train, label = self.get_all_data()
        nrow = len(train)
        test_size = int(nrow * test_part)
        idx = random.sample(xrange(nrow), test_size)
        mask = np.ones(nrow, dtype=bool)
        mask[idx] = False
        
        return (train[mask, :], label[mask], train[idx, :], label[idx])

def prepare_user_training(partition, file_num):
    print "Initializing..."
    tp = UserTrainSet()
    print 'Getting training data...'
    (train_data, train_label, 
     test_data, test_label) = tp.get_train_and_test(0, 6e6, partition)
    print 'Saving data...'
    num = str(file_num)
    np.save('train_data' + num, train_data)
    np.save('train_label' + num, train_label)
    np.save('test_data' + num, test_data)
    np.save('test_label' + num, test_label)
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print "usage: script [partition_num] [data_file_num]"
    else:
        prepare_user_training(int(sys.argv[1]), sys.argv[2])
