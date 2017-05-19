#!/usr/bin/env python


import base
import pandas as pd
import tables

if __name__ == '__main__':
    pc = base.ProductClassifier(action_type=4)
    pc.load_associates('assoc_table')
    ut = pc.get_user_recommend()

# ut is the dataframe with user_id and the recommended items in cate 8
