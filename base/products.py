from base import *

class Product(DataHolder):
    
    def __init__(self, data_file):
        self.columns = {'a1': 0, 'a2': 1, 'a3': 2, 'brand': 3, 'comment_num': 4,
                       'has_bad_comment': 5, 'bad_comment_rate': 6, 'sku_id': 7}
        self.data = np.load(data_file)
        self.cate = 8
        
    