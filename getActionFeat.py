from __future__ import division
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
import numpy as np

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

def get_action_feature_by_cate(action_all,start_date,end_date):
    #action_paths = [action_1_path, action_2_path, action_3_path]
    #action1, action2, action3 = [pd.read_csv(p) for p in action_paths]
    #action_all = pd.concat([action1, action2, action3])
    batch_size=10**6
    for batch in range(1,int(math.ceil(action_all.shape[0]/batch_size))+1):
    #for batch in range(1,2):
        dump_path='./cache/basic_action_feat_%d.pkl' % batch
        if not os.path.exists(dump_path):
            if batch<int(math.ceil(action_all.shape[0]/batch_size)):
                action_timeRange=action_all.iloc[(batch-1)*batch_size:batch*batch_size,:]
            else:
                action_timeRange=action_all.iloc[(batch-1)*batch_size:,:]
            action_timeRange=action_timeRange[(action_timeRange.time>=start_date) & (action_timeRange.time<=end_date)]
            action_timeRange_onehot_type=pd.get_dummies(action_timeRange['type'],prefix='act_type')
            action_timeRange_onehot_cate=pd.get_dummies(action_timeRange['cate'],prefix='cate')
            action_timeRange_onehot_mix=pd.DataFrame()
            for i in action_timeRange_onehot_type.columns:
                for j in action_timeRange_onehot_cate.columns:
                    action_timeRange_onehot_mix[i+'_'+j]=action_timeRange_onehot_type[i]*action_timeRange_onehot_cate[j]
            action_timeRange_onehot_mix=action_timeRange_onehot_mix.to_sparse(fill_value=0)
            # too many brands/model_id categories and ~40% of mode_id are NaN, therefore ditch these two features,
            # for predicting if will buy cate8 anyway
            action_timeRange=pd.concat((action_timeRange.user_id,action_timeRange.time,action_timeRange_onehot_mix),axis=1)
            days_fromStart=action_timeRange.time.map(lambda x: (datetime.strptime(x,'%Y-%m-%d %H:%M:%S')-datetime.strptime(start_date,'%Y-%m-%d')).days)
            days_fromEnd=action_timeRange.time.map(lambda x: (datetime.strptime(end_date,'%Y-%m-%d')-datetime.strptime(x,'%Y-%m-%d %H:%M:%S')).days)
            # additional features: actions weighted by reverse of the days, and negative exponential of the days
            # though recent actions might seem to indicate more heavily on the users' behaviour
            # recent purcahse could also indicate the users do not need to buy them anytime soon
            # Therefore, we have 'inv_recent', 'exp_recent' as well as 'inv_oldest', 'exp_oldest'

            #not to add the inverse features, to speed up groupby
			#tempFeat=action_timeRange_onehot_mix.copy()
            #for feat in tempFeat.columns:
            #    tempFeat[feat]=tempFeat[feat]/(days_fromEnd+1)
            #tempFeat.columns+='_inv_recent'
            #action_timeRange=pd.concat((action_timeRange,tempFeat),axis=1)

            #tempFeat=action_timeRange_onehot_mix.copy()
            #for feat in tempFeat.columns:
            #    tempFeat[feat]=tempFeat[feat]/(days_fromStart+1)
            #tempFeat.columns+='_inv_oldest'
            #action_timeRange=pd.concat((action_timeRange,tempFeat),axis=1)

            expWeight_recent=days_fromEnd.map(lambda x: math.exp(-x/15)) # not to decay too quickly
            expWeight_oldest=days_fromStart.map(lambda x: math.exp(-x/15))

            tempFeat=action_timeRange_onehot_mix.copy()
            for feat in tempFeat.columns:
                tempFeat[feat]=tempFeat[feat]*expWeight_recent
            tempFeat.columns+='_exp_recent'
            action_timeRange=pd.concat((action_timeRange,tempFeat),axis=1)

            tempFeat=action_timeRange_onehot_mix.copy()
            for feat in tempFeat.columns:
                tempFeat[feat]=tempFeat[feat]*expWeight_oldest
            tempFeat.columns+='_exp_oldest'
            action_timeRange=pd.concat((action_timeRange,tempFeat),axis=1)
            action_timeRange=action_timeRange.drop('time',axis=1)
            action_timeRange=action_timeRange.groupby('user_id',axis=0,as_index=False).sum()
            pickle.dump(action_timeRange, open(dump_path,'w'))

action_paths = [action_1_path, action_2_path, action_3_path]
action1, action2, action3 = [pd.read_csv(p) for p in action_paths]
action_all = pd.concat([action1, action2, action3])
action1=None
action2=None
action3=None
t0=time.time()
get_action_feature_by_cate(action_all,'2016-02-01','2016-04-10')
print time.time()-t0

