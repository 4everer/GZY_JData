from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

import pickle

from forest import *

class ForestTrainer(object):
    def __init__(self):
        self.clf = RandomForestClassifier()
        self.param_dict = {'n_estimators': randint(20, 50),
              'min_samples_leaf': randint(1, 4),
              'criterion': ['gini', 'entropy'],
              'max_features': ['sqrt','log2',None],
            'class_weight': [{0:1, 1:x} for x in np.logspace(0, 2, 20)]}
        self.clf_randomCV = None
        
#        self.param_dict = {'n_estimators': randint(25, 36),
#              'min_samples_leaf': [3],
#              'criterion': ['gini', 'entropy'],
#              'max_features': ['log2'],
#            'class_weight': [{0:1, 1:x} for x in np.logspace(1.8, 1.95, 16)]}
    def train(self, train_data, train_label, scorer=default_scorer, n_jobs=-1):
        t0 = time.time()
        self.clf_randomCV = RandomizedSearchCV(self.clf, 
                                        param_distributions=self.param_dict,
                                        n_iter=100, scoring=scorer,
                                        cv=3, n_jobs=n_jobs)
        self.clf_randomCV.fit(train_data, train_label)

        print time.time() - t0
        
    def save(self, file):
        with open(file, 'w') as f:
            pickle.dump(self.get_model(), f)
            
    def predict(self, data):
        try:
            return self.clf_randomCV.predict(data)
        except AttributeError:
            print "Model needs to be fitted first"
    
    def predict_proba(self, data):
        try:
            return self.clf_randomCV.predict_proba(data)
        except AttributeError:
            print "Model needs to be fitted first"
    
    def get_model(self):
        try:
            return self.clf_randomCV.best_estimator_
        except AttributeError:
            print "Model needs to be fitted first"
