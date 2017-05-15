# skewed data, cannot just use accuracy as evaluation score
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def JD_custom_score_func(label,prediction):
    Precise=np.sum(label*prediction)/np.sum(prediction+0.0001)
    Recall=np.sum(label*prediction)/np.sum(label+0.0001)
    F11=6*Recall*Precise/(5*Recall+Precise+0.0001)
    F12=5*Recall*Precise/(2*Recall+3*Precise+0.0001)
    Score=0.4*F11 + 0.6*F12
    return Score

JDcustomScorer=make_scorer(JD_custom_score_func, greater_is_better=True)