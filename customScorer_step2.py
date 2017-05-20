# skewed data, cannot just use accuracy as evaluation score
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def JD_custom_score_func2(label,prediction):
    correct = np.sum(label==prediction) * 1.0
    precision = correct / (len(prediction) + 1e-6) + 1e-6
    recall = correct / (len(label) + 1e-6) + 1e-6
    F12=5*recall*precision/(2*recall+3*precision+0.0001)
    return F12

JDcustomScorer2=make_scorer(JD_custom_score_func2, greater_is_better=True)