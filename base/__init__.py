#!/usr/bin/env python

from __future__ import division
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
import random

test_start = '2016-02-01'
test_end = '2016-04-15'
total_time = int((pd.to_datetime(test_end) - 
                  pd.to_datetime(test_start)).total_seconds())

class DataHolder(object):
    
    def __init__(self, input_data=None, input_columns={}):
        self.columns = input_columns
        self.data = input_data
    
    def get_column(self, key):
        return self.columns[key]
        
    def __getitem__(self, key):
        col = self.get_column(key)
        return self.data[:, col]
        
        
    

from users import *
from actions import *
from products import *
from comments import *
from features import *
