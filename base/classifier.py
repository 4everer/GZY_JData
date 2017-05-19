#!/usr/bin/env python

from __future__ import division
import pandas as pd
import numpy as np
from base import *

import random

class ProductClassifier(Action):
    def __init__(self, action_data='data/action_simple.npy',
                 product_data='data/product_simple.npy',
                 action_type=None):
        Action.__init__(self, action_data)
        if action_type is not None:
            self.data = self.data[self['type'] == action_type, :]
        self.sku_ids = np.unique(self['sku_id'])
        self.data = pd.DataFrame(self.data, columns=self.get_columns())
        products = Product(product_data)
        self.interest_products = products['sku_id']
        self.num_products = len(self.sku_ids)
        self.assoc_table = None

    def calc_associates(self, cate=8, data_file='data/assoc_table'):
        self.assoc_table = pd.DataFrame(self.interest_products,
                                        columns=['sku_id'],
                                        dtype=int)
        self.assoc_table = self.assoc_table.set_index('sku_id')
        group_data = self.data.groupby('user_id')
        for name, group in group_data:
            if cate in group['cate'].values:
                sku_ids = group['sku_id']
                interest = []
#                others = []
                for sku_id in sku_ids:
                    if sku_id in self.interest_products:
                        interest.append(sku_id)
#                    else:
#                        others.append(sku_id)
                if len(interest) == 0:
                    continue
                for sku_id in sku_ids:
                    if sku_id in self.assoc_table:
                        self.assoc_table.loc[interest, sku_id] += 1
                    else:
                        self.assoc_table[sku_id] = np.zeros(len(self.assoc_table),
                                                            dtype=int)
                        self.assoc_table.loc[interest, sku_id] += 1
        self.assoc_table = self.assoc_table.transpose()
        mask = np.sum(self.assoc_table)
        self.assoc_table = self.assoc_table.loc[:, mask > 0]
        if data_file is not None:
            self.assoc_table.to_hdf(data_file, 'assoc_table')
            
    def load_associates(self, data_file):
        self.assoc_table = pd.read_hdf(data_file)
        
    def get_neighbors(self, sku_list):
        idx = np.intersect1d(sku_list, self.assoc_table.index, 
                             assume_unique=True)
        if len(idx) == 0:
            return None, None
        try:
            data = self.assoc_table.loc[idx, :]
            sku_val = np.sum(data.as_matrix(), axis=0)
            sku_ids = data.columns[sku_val > 0]
            sku_val = sku_val[sku_val > 0]
            return sku_ids, sku_val
        
        except KeyError:
            return None, None
    
    def get_user_recommend(self):
        group_data = self.data.groupby('user_id')
        user_ids = np.unique(self.data['user_id'])
        user_items = pd.DataFrame({'user_id': user_ids,
                                   'items': [None] * len(user_ids)})
        user_items = user_items.set_index('user_id')
        for name, group in group_data:
            sku_ids = np.unique(group['sku_id'])
            sku_list, sku_val = self.get_neighbors(sku_ids)
            user_items.loc[name, 'items'] = sku_list
        return user_items
            
        
