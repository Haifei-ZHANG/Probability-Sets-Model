# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:20:29 2023

@author: zhanghai
"""


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances


class ProbabilitySetsModel:
    def __init__(self, n_trees=100, alpha=0, random_state=42):
        self.n_trees = n_trees
        self.alpha = alpha
        self.classes = None
        self.n_class = None
        self.rf = RandomForestClassifier(n_estimators=n_trees, random_state=random_state)
        
        
    def fit(self, train_x, train_y):
        self.rf.fit(train_x, train_y)
        
        self.classes = self.rf.classes_
        self.n_class = len(self.rf.classes_)
    
    
    def predict(self, X, dacc=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_instance = X.shape[0]
        all_probability_sets = np.zeros((n_instance, self.n_trees, self.n_class))
        for t in range(self.n_trees):
            tree = self.rf.estimators_[t]
            all_probability_sets[:,t,:] = tree.predict_proba(X)
            
        predictions = []
        for i in range(n_instance):
            dominated_classes_index = []
            probability_sets = all_probability_sets[i,:,:]
            
            median = probability_sets.mean(axis=0)
            dists_to_median = euclidean_distances(median.reshape((1,-1)), probability_sets)[0]
            probability_sets = probability_sets[np.argsort(dists_to_median),:]
            rest_probability_sets = probability_sets[0:int((1-self.alpha)*self.n_trees), :]
            for j in range(self.n_class):
                compared_classes = np.setdiff1d(range(self.n_class), [j])
                for k in compared_classes:
                    if np.all(rest_probability_sets[:,j] <= rest_probability_sets[:,k]):
                        dominated_classes_index.append(j)
                        
            prediction_index = np.setdiff1d(range(self.n_class), dominated_classes_index)
            predictions.append(self.classes[prediction_index])
           
        
        return predictions
    
        
        
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        determinacy = 0
        single_set_accuracy = 0
        set_accuracy = 0
        set_size = 0
        u65 = 0
        u80 = 0
        for i in range(len(y_test)):
            prediction = predictions[i]
            if len(prediction) == 1:
                determinacy += 1
                if prediction[0] == y_test[i]:
                    single_set_accuracy += 1
                    u65 += 1
                    u80 += 1
            else:
                set_size += len(prediction)
                if y_test[i] in prediction:
                    set_accuracy += 1
                    u65 += (-0.6/(len(prediction)**2) + 1.6/len(prediction))
                    u80 += (-1.2/(len(prediction)**2) + 2.2/len(prediction))
                    
        n_determinate = determinacy
        n_indeterminate = len(y_test) - determinacy
        
        determinacy /= len(y_test)
        if n_determinate == 0:
            single_set_accuracy = None
        else:
            single_set_accuracy /= n_determinate
            
        if n_indeterminate == 0:
            set_accuracy = None
            set_size = None
        else:
            set_accuracy /= n_indeterminate
            set_size /= n_indeterminate
            
        u65 /= len(y_test)
        u80 /= len(y_test)
                
        return {'determinacy': determinacy,
                 'single set accuracy': single_set_accuracy,
                 'set accuracy': set_accuracy,
                 'set size': set_size,
                 'u65 score': u65, 
                 'u80 score': u80}