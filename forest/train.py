from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from customScorer import JDcustomScorer
import pickle

from forest import *

class ForestTrainer(object):
    def __init__(self):
        self.clf = RandomForestClassifier()
        self.param_dict = {'n_estimators':randint(20,50),
              'min_samples_leaf':randint(1,4),
              'criterion':['gini', 'entropy'],
              'max_features':['sqrt','log2',None],
            'class_weight':[{0:1,1:x} for x in np.logspace(0,2,20)]}
        self.clf_randomCV = None
        
    def train(self, train_data, train_label, scorer=JDcustomScorer):
        t0 = time.time()
        self.clf_randomCV = RandomizedSearchCV(self.clf, 
                                        param_distributions=self.param_dict,
                                        n_iter=100, scoring=JDcustomScorer,
                                        cv=3, n_jobs=-1)
        self.clf_randomCV.fit(train_data, train_label)

        print time.time() - t0
        
    def save(self, file):
        with open(file, 'w') as f:
            pickle.dump(self.clf, f)
            pickle.dump(self.clf_randomCV, f)
            
    def predict(self, data):
        return self.clf_randomCV.predict(data)
        