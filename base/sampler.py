# -*- coding: UTF-8 -*-

from base import *

def convert_age(age_str):
        if age_str == u'-1':
            return 0
        elif age_str == u'15岁以下':
            return 1
        elif age_str == u'16-25岁':
            return 2
        elif age_str == u'26-35岁':
            return 3
        elif age_str == u'36-45岁':
            return 4
        elif age_str == u'46-55岁':
            return 5
        elif age_str == u'56岁以上':
            return 6
        else:
            return -1
        
class UserSampler(object):
    columns = {'user_id': 0, 'age': 1, 'sex': 2, 'user_lv_cd': 3, 'user_reg_tm': 4}
    
    def __init__(self, file):
        if file[-3:] == 'csv':
            self.load_csv(file)
        else:
            self.load_save(file)
        self.sequence = np.arange(len(self.user_data))
        random.shuffle(self.sequence)
        self.sequence_index = 0
        
    def __getitem__(self, key):
        col = self.columns[key]
        return self.user_data[:, col]
    
    def get_column(self, key):
        return self.columns[key]
    
    def load_csv(self, file):
        user_data_file = 'data/user_simple'
        self.user_data = pd.read_csv(file, encoding='gbk')
        self.user_data['age'] = self.user_data['age'].map(convert_age)
        #self.user_data['sex'][pd.isnull(self.user_data['sex'])] = 0
        self.user_data['sex'] = self.user_data['sex'].map(lambda x: int(x) if not pd.isnull(x) else 0)
        self.user_data = self.user_data[self.user_data['user_reg_tm'] < test_start]
        self.user_data['user_reg_tm'] = pd.to_datetime(test_start) - pd.to_datetime(self.user_data['user_reg_tm'])
        self.user_data['user_reg_tm'] = self.user_data['user_reg_tm'].map(lambda x: int(x.days) if not pd.isnull(x) else 0)
        self.user_data = self.user_data.as_matrix()
        np.save(user_data_file, self.user_data)
    
    def load_save(self, file):
        self.user_data = np.load(file)
            
    def get_user_batch(self, size=300):
        idx = self.sequence[self.sequence_index:(self.sequence_index + size)]
        self.sequence_index += size
        return self.user_data[idx, :]


class ActionSampler(object):
    
    TYPE_VIEW = 1
    TYPE_CART = 2
    TYPE_DECART = 3
    TYPE_BUY = 4
    TYPE_FAVOR = 5
    TYPE_CLICK = 6
    
    columns = {'user_id': 0, 'sku_id': 1, 'time': 2, 'type': 3, 'cate': 4, 'brand': 5}
    num_action = 6
    
    def __init__(self, action_npy):
        self.all_action = np.load(action_npy)
        self.kernel_list = []
        
    def __getitem__(self, key):
        col = self.columns[key]
        return self.all_action[:, col]
    
    def add_kernel(self, weights):
        assert len(weights) == self.num_action, "Each action type requires a weight"
        self.kernel_list.append(weights)
    
    def get_kernel(self, kernel_id):
        return self.kernel_list[kernel_id]
    
    def get_column(self, column_name):
        return self.columns[column_name]
    
    def get_impulse(self, user_id, till_time, kernel_id=0):
        data = self.all_action[self.all_action[:, self.get_column('user_id')] == user_id, :]
        kernel = self.get_kernel(kernel_id)
        category = np.unique(data[:, self.get_column('cate')])
        impulse = np.zeros([len(category), self.num_action + 1])
        for cate_idx, cate in enumerate(category):
            cate_data = data[data[:, self.get_column('cate')] == cate, :]
            impulse[cate_idx, 0] = cate
            for action in range(self.num_action):
                action_time = np.array(cate_data[cate_data[:, 
                                            self.get_column('type')] == action + 1, 
                                            self.get_column('time')])
                action_time = action_time[action_time < till_time]
                impulse[cate_idx, action + 1] = np.sum(np.exp((till_time - action_time)
                                                * kernel[action]))
        return impulse
    
    def get_purchase(self, user_id, start_time, end_time):
        data = self.all_action[self.all_action[:, self.get_column('user_id')] == user_id, :]
        data = data[data[:, self.get_column('type')] == self.TYPE_BUY, :]
        data = data[data[:, self.get_column('time')] > start_time, :]
        data = data[data[:, self.get_column('time')] < end_time, :]
        return data

    def sort(self, key):
        col = self.get_column(key)
        self.all_action.view('i8,i8,i8,i8,i8,i8').sort(order=['f' + str(col)], axis=0)
            