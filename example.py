# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:32:02 2023

@author: zhanghai
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from probability_set_model import ProbabilitySetsModel



if __name__ == "__main__":
    data = pd.read_csv("data/libras.csv")
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
    
    clf = ProbabilitySetsModel(n_trees=100, alpha=0.2, p_dist_type='SQE',random_state=None)
    clf.fit(X_train, y_train)
    
    dist_types = ['SQE','L1','KL'] #, 'L1', 'KL'
    for d_type in dist_types:
        print(d_type)
        clf.p_dist_type = d_type
        print(clf.evaluate(X_test, y_test, 'Maximality'))
        print(clf.evaluate(X_test, y_test, 'EAD'))