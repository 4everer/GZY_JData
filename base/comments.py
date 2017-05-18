from base import *

class Comment(DataHolder):
    
    def __init__(self, data_file):
        self.columns = {'dt': 0, 'sku_id': 1, 'comment_num': 2,
                       'has_bad_comment': 3, 'bad_comment_rate': 4}
        if data_file[-3:] == 'csv':
            self.load_csv(data_file)
        else:
            self.data = np.load(data_file)
            
    def load_csv(self, data_file):
        npy_file = 'data/comment_simple'
        comm = pd.read_csv(data_file)
        comm['dt'] = pd.to_datetime(comm['dt']) - pd.to_datetime(test_start)
        comm['dt'] = comm['dt'].map(lambda x:int(x.total_seconds()))
        self.data = comm.as_matrix()
        np.save(npy_file, self.data)
