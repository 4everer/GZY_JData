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
        
class User(DataHolder):
    
    def __init__(self, file):
        if file[-3:] == 'csv':
            self.load_csv(file)
        else:
            self.load_save(file)
        self.sequence = np.arange(len(self.data))
        random.shuffle(self.sequence)
        self.sequence_index = 0
        self.columns = {'user_id': 0, 'age': 1, 'sex': 2, 'user_lv_cd': 3, 'user_reg_tm': 4}
    
    def load_csv(self, file):
        user_data_file = 'data/user_simple'
        self.data = pd.read_csv(file, encoding='gbk')
        self.data['age'] = self.data['age'].map(convert_age)
        self.data['sex'] = self.data['sex'].map(lambda x: int(x) if not pd.isnull(x) else 0)
        self.data = self.data[self.data['user_reg_tm'] < test_end]
        self.data['user_reg_tm'] = pd.to_datetime(test_start) - pd.to_datetime(self.data['user_reg_tm'])
        self.data['user_reg_tm'] = self.data['user_reg_tm'].map(lambda x: int(x.days) if not pd.isnull(x) else 0)
        self.data = self.data.as_matrix()
        np.save(user_data_file, self.data)
    
    def load_save(self, file):
        self.data = np.load(file)
            
    def get_user_batch(self, size=300):
        idx = self.sequence[self.sequence_index:(self.sequence_index + size)]
        self.sequence_index += size
        return self.data[idx, :]

           