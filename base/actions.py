from base import *

class Action(DataHolder):
    
    TYPE_VIEW = 1
    TYPE_CART = 2
    TYPE_DECART = 3
    TYPE_BUY = 4
    TYPE_FAVOR = 5
    TYPE_CLICK = 6
    
    num_action = 6
    
    def __init__(self, action_npy):
        self.columns = {'user_id': 0, 'sku_id': 1, 'time': 2, 'type': 3, 'cate': 4, 'brand': 5}
        self.data = np.load(action_npy)
        self.kernel_list = []
    
    def add_kernel(self, weights):
        assert len(weights) == self.num_action, "Each action type requires a weight"
        self.kernel_list.append(weights)
    
    def get_kernel(self, kernel_id):
        return self.kernel_list[kernel_id]
    
    def get_column(self, column_name):
        return self.columns[column_name]
    
    def get_impulse(self, user_id, till_time, kernel_id=0):
        data = self.data[self.data[:, self.get_column('user_id')] == user_id, :]
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
        data = self.data[self.data[:, self.get_column('user_id')] == user_id, :]
        data = data[data[:, self.get_column('type')] == self.TYPE_BUY, :]
        data = data[data[:, self.get_column('time')] > start_time, :]
        data = data[data[:, self.get_column('time')] < end_time, :]
        return data

    def sort(self, key):
        col = self.get_column(key)
        self.data.view('i8,i8,i8,i8,i8,i8').sort(order=['f' + str(col)], axis=0)
    
    def get_duration(self):
        return (np.min(self['time']), np.max(self['time']))

 