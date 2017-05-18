#!/usr/env/python

import base
import forest
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
import sys

if __name__ == '__main__':
    partition = int(sys.argv[1])
    duration = int(sys.argv[2])
    
    with open('data/user_model', 'r') as f:
        user_forest = pickle.load(f)
    user_train_set = base.UserTrainSet()
    (time_min, time_max) = user_train_set.get_duration()
    (user_data, user_label) = user_train_set.get_all_data(time_max - duration, 
                                                          time_max,
                                                          partition)
    user_predict = user_forest.predict(user_data)
    users_to_buy = user_train_set.users['user_id'][user_predict > 0]
    
    buy_data = user_train_set.data[user_train_set['cate'] == 8, :]
    
    for user in users_to_buy:
        current_data = base.DataHolder(buy_data[buy_data[:, 
                            user_train_set.get_column('user_id')] == user, :],
                                       user_train_set.columns)
        sku_viewed = np.unique(current_data['sku_id'])
        sku_in_cart = np.zeros_like(sku_viewed)
        sku_fav = np.zeros_like(sku_viewed)
        sk
        for sku in sku_viewed:
            actions = current_data['type'][current_data['sku_id'] == sku]
            
        
        
    
    