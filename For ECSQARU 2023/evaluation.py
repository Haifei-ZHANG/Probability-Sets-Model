# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:28:39 2023

@author: zhanghai

Here we compare the Probability Sets Model with Random Forests in terms of
various metrics
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from model import ProbabilitySetsModel


if __name__ == "__main__":
    data_names = ['ecoli', 'balance_scale', 'vehicle', 'vowel', 'wine_quality',
                  'optdigits', 'segment',  'waveform', 'letter']

    alpha_list = np.arange(0,20)/20
    K = 10
    total_evaluations = np.zeros((len(data_names), 9))
    
    
    for d in range(len(data_names)):
        data_name = data_names[d]
        data = pd.read_csv('data/{}.csv'.format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])
        
        data_evaluations = np.zeros((K, 9))
        kf = KFold(n_splits=K, shuffle=False)
        k = 0
        for train_index, test_index in tqdm(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            inner_kf = KFold(n_splits=K, shuffle=False)
            inner_u65_evaluations = np.zeros(len(alpha_list))
            
            for inner_train_index, inner_val_index in inner_kf.split(X_train):
                inner_X_train, inner_X_val = X_train[inner_train_index], X_train[inner_val_index]
                inner_y_train, inner_y_val = y_train[inner_train_index], y_train[inner_val_index]
    
                clf = ProbabilitySetsModel(n_trees=100, alpha=0, random_state=42)
                clf.fit(inner_X_train, inner_y_train)
    
                for a in range(len(alpha_list)):
                    clf.alpha = alpha_list[a]
                    inner_u65_evaluations[a] += clf.evaluate(inner_X_val, inner_y_val)['u65 score']
            
            best_alpha = alpha_list[np.argmax(inner_u65_evaluations)]
            clf = ProbabilitySetsModel(n_trees=100, alpha=best_alpha, random_state=42)
            clf.fit(X_train, y_train)
            data_evaluations[k, 0] = clf.rf.score(X_test, y_test)
            data_evaluations[k, 1:7] = np.array(list(clf.evaluate(X_test, y_test).values()))
            
            precise_predictions = clf.rf.predict(X_test)
            imprecise_predictions  = clf.predict(X_test)
            n_imprecise = 0
            for i in range(len(y_test)):
                if len(imprecise_predictions[i])>1:
                    n_imprecise += 1
                    if y_test[i] == precise_predictions[i]:
                        data_evaluations[k, 7] += 1
                    if y_test[i] in imprecise_predictions[i]:
                        data_evaluations[k, 8] += 1
            if n_imprecise == 0:
                data_evaluations[k, 7] = None
                data_evaluations[k, 8] = None
            else:
                data_evaluations[k, 7] /= n_imprecise
                data_evaluations[k, 8] /= n_imprecise
            k += 1
    
        for j in range(9):
            valid_index = ~np.isnan(data_evaluations[:,j])
            total_evaluations[d, j] = data_evaluations[valid_index,j].mean()
        print(data_names[d])
        print(total_evaluations[d])
    
    np.save('results/total_performance_evaluations.npy', total_evaluations)
    
    columns = ['RF Acc', 'Determinacy', 'Single Acc', 'Set Acc', 'Set Size', 
                'U65', 'U80', 'RF Abstention Acc', 'Abstention Corr']
    df = pd.DataFrame(total_evaluations, index = data_names, columns=columns)
    df.to_csv('results/total_performance_evaluations.csv')
    df.to_excel('results/total_performance_evaluations.xlsx')