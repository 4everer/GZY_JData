# skewed data, cannot just use accuracy as evaluation score
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def JD_custom_score_func(label,prediction):
    correct = np.sum(label * prediction) * 1.0
    precision = correct / (np.sum(prediction) + 1e-6) + 1e-6
    recall = correct / (np.sum(label) + 1e-6) + 1e-6
    return 2 * recall * precision / (recall + precision)
#    F11 = 6 * recall * precision / (5 * recall + precision)
#    F12 = 5 * recall * precision / (2 * recall + 3 * precision)
#    Score = 0.4 * F11 + 0.6 * F12
#    return Score

JDcustomScorer = make_scorer(JD_custom_score_func, greater_is_better=True)