# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:52:08 2023

@author: zhanghai

This is an example of how to use the ProbabilitySetsModel

It is like scikit-learn, just create model, fit the model, mank predictions,
and evaluate the scores associated with imprecise classification

"""



import numpy as np
import pandas as pd
from model import ProbabilitySetsModel
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    data = pd.read_csv("data/iris.csv")
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                        random_state=42)
    # create the ProbabilitySetsModel
    clf = ProbabilitySetsModel(n_trees=100, alpha=0,random_state=42)
    
    # train model
    clf.fit(X_train, y_train)
    
    # make predictions
    predictions = clf.predict(X_test)
    for i in range(5,10):
        print('Instance{},\n prediction={},\n real label={}\n\n'.format(i, 
                                                                predictions[i], 
                                                                [y_test[i]]))
    
    
    # evaluate scores, including determinacy, single accuracy, set accuracy,
    # set size, u65 score and u80 score
    scores = clf.evaluate(X_test, y_test)
    print("\nScores:")
    for i, (k, v) in enumerate(scores.items()):
        
        print("{}: {}".format(k, round(v,4)))