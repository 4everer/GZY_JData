from base import *

class Product(DataHolder):
    
    def __init__(self, data_file):
        self.columns = {'sku_id': 0, 'a1': 0, 'a2': 1, 'a3': 2, 'brand': 3}
        self.data = np.load(data_file)
        self.cate = 8
        
    