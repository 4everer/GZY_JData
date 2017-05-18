#!/usr/env/python

import base
import forest
import pickle
from sklearn.metrics import make_scorer

pts = base.ProductTrainSet()
train_data, train_label, test_data, test_label = pts.get_train_and_test()

ft = forest.ForestTrainer()
r_scorer = make_scorer(forest.score_func, greater_is_better=True, recall_precision_ratio=5.0)
ft.train(train_data, train_label, r_scorer)

product_data = base.Product('data/product_simple.npy')
buy_probability = ft.predict_proba(product_data.data[:, :-1])
buy_probability[:, 0] = product_data.data[:, -1]

np.save('data/product_prob', buy_probability)
ft.save('data/product_model')
