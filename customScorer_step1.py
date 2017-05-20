# skewed data, cannot just use accuracy as evaluation score
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def JD_custom_score_func1(label,prediction):
    Precise=np.sum(label*prediction)/np.sum(prediction+0.0001)
    Recall=np.sum(label*prediction)/np.sum(label+0.0001)
    F11=6*Recall*Precise/(5*Recall+Precise+0.0001)
    return F11

JDcustomScorer1=make_scorer(JD_custom_score_func1, greater_is_better=True)