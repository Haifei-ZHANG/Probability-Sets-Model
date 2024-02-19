# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:56:47 2023

@author: zhanghai
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from cautious_random_forest import CautiousRandomForest
from probability_set_model import ProbabilitySetsModel


if __name__ == "__main__":
    evaluation_criteria = ['RF Acc', 'Determinacy', 'Single Acc', 'Set Acc', 'Set Size', 'U65', 'U80', 'RF Abstention Acc', 'Best Alpha']
    data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment'] #, 'waveform', 'yeast'
    dist_types = ['SQE', 'L1', 'KLD']
    prediction_types = ['Maximality', 'E-admissibility']
    nt = 100
    msl_list = [1, 5, 10]
    alpha_list = np.arange(0,20)/20
    K = 10
    
    for msl in msl_list:
        print('min_samples_leaf={}'.format(msl))
        total_evaluations = np.zeros((len(evaluation_criteria), len(data_names), 2+len(dist_types)*len(prediction_types))) # n_metrics, n_data, n_models
        
        for d in range(len(data_names)):
            data_name = data_names[d]
            print(data_name)
            data = pd.read_csv('data/{}.csv'.format(data_name))
            X = np.array(data.iloc[:,:-1])
            y = np.array(data.iloc[:,-1]) 
            classes = np.unique(y)
            
            data_evaluations = np.zeros((K, len(evaluation_criteria), 2+len(dist_types)*len(prediction_types))) # n_fold, n_metrics,n_models 
            kf = KFold(n_splits=K, shuffle=False)
            k = -1
            for train_index, test_index in tqdm(kf.split(X)):
                k += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # in X_train, choose certain proportion of instance to change its label
                instance_select = np.random.choice(len(y_train),int(0.25*len(y_train)),replace=False)
                for instance_index in instance_select:
                    candidate_y = np.setdiff1d(classes, y_train[instance_index])
                    y_train[instance_index] = candidate_y[np.random.choice(len(candidate_y), 1)[0]]
                
                # NDC and CDM_Vote
                clf = CautiousRandomForest(n_trees=nt, s=2, min_samples_leaf=msl,combination='ndc', random_state=42)
                clf.fit(X_train, y_train)
                data_evaluations[k,:-1,0] = np.array(list(clf.score(X_test, y_test).values()))
                clf.combination = 'cdm-vote'
                data_evaluations[k,:-1,1] = np.array(list(clf.score(X_test, y_test).values()))
                
    
                clf = ProbabilitySetsModel(n_trees=nt, alpha=0, p_dist_type='SQE', min_samples_leaf=msl, random_state=42)
                clf.fit(X_train, y_train)
                
                inner_kf = KFold(n_splits=K, shuffle=False)
                for i in range(len(dist_types)):
                    dist_type = dist_types[i]
                    clf.p_dist_type = dist_type
                    for j in range(len(prediction_types)):
                        prediction_type = prediction_types[j]
                        index = 2 + i * len(prediction_types) + j
                        
                        inner_u65_evaluations = np.zeros(len(alpha_list))
        
                        for inner_train_index, inner_val_index in inner_kf.split(X_train):
                            inner_X_train, inner_X_val = X_train[inner_train_index], X_train[inner_val_index]
                            inner_y_train, inner_y_val = y_train[inner_train_index], y_train[inner_val_index]
        
                            inner_clf = ProbabilitySetsModel(n_trees=nt, alpha=0, p_dist_type=dist_type, min_samples_leaf=msl, random_state=42)
                            inner_clf.fit(inner_X_train, inner_y_train)
        
                            for a in range(len(alpha_list)):
                                inner_clf.alpha = alpha_list[a]
                                inner_u65_evaluations[a] += inner_clf.evaluate(inner_X_val, inner_y_val, prediction_type)['u65 score']
                
                        best_alpha = alpha_list[np.argmax(inner_u65_evaluations)]
                        clf.alpha = best_alpha
                        data_evaluations[k,-1,index] = best_alpha
                        data_evaluations[k,:-1,index] = np.array(list(clf.evaluate(X_test, y_test, prediction_type).values()))
            
            np.save('results/noisy data imprecise prediction/min_samples_leaf={}/{}_evaluations.npy'.format(msl, data_name), data_evaluations)           
            for i in range(len(evaluation_criteria)):
                if i in [3,4,7]:
                    for j in range(2+len(dist_types)*len(prediction_types)):
                        valid_index = ~np.isnan(data_evaluations[:,i,j])
                        weights = 1 - data_evaluations[valid_index,1,j]
                        weights /= weights.sum()
                        total_evaluations[i,d,j] = sum(data_evaluations[valid_index,i,j] * weights)
                else:
                    for j in range(2+len(dist_types)*len(prediction_types)):
                        total_evaluations[i,d,j] = data_evaluations[:,i,j].mean()
            np.save('results/noisy data imprecise prediction/min_samples_leaf={}/total_evaluations.npy'.format(msl), total_evaluations)
            print(total_evaluations[:,d,:])
            
            
        total_evaluations = np.load('results/noisy data imprecise prediction/min_samples_leaf={}/total_evaluations.npy'.format(msl))
        
        sheet_names = ['RF Acc', 'Determinacy', 'Single Acc', 'Set Acc', 'Set Size', 'U65', 'U80', 'RF Abstention Acc', 'Best Alpha']
        data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras', 'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
        model_names = ['NDC', 'CRF', 'SQE-Max', 'SQE-Ead', 'L1-Max', 'L1-Ead', 'KL-Max', 'KL-Ead']
        
        with pd.ExcelWriter('results/noisy data imprecise prediction/min_samples_leaf={}/total_evaluations.xlsx'.format(msl)) as writer:
            for i in range(len(sheet_names)):
                sheet_name = sheet_names[i]
                df = pd.DataFrame(total_evaluations[i].round(4), columns=model_names, index=data_names)
                df.to_excel(writer, sheet_name=sheet_name, index=True)