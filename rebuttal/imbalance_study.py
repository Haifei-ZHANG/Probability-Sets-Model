# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:09:05 2024

@author: zh106121
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
import sys 
sys.path.append("..")
from probability_set_model import ProbabilitySetsModel


def calculate_imbalance_score(counts):
    # n_class = len(counts)
    # ratio = counts/sum(counts)
    # return  1- sum(-ratio*np.log(ratio))/np.log(n_class)
    return round(min(counts)/max(counts), 4)


def get_imbalance_datasets(data_names):
    imbalance_datasets = []
    for d in range(len(data_names)):
        data_name = data_names[d]
        print(data_name)
        data = pd.read_csv('../data/{}.csv'.format(data_name))
        
        y = np.array(data.iloc[:,-1]) 
        
        classes, class_wise_counts = np.unique(y, return_counts=True)
        
        imbalance_score = calculate_imbalance_score(class_wise_counts)
        if imbalance_score < 0.2:
            imbalance_datasets.append(data_name)
            
        print("Dataset: ", data_name)
        print("number of classes: ", len(classes))
        print("Counts for each class: ", class_wise_counts)
        print("Imbalance score: ", imbalance_score)
        print("=============================\n\n")
        
    return imbalance_datasets
        
        
        
        
if __name__ == "__main__":
    evaluation_criteria = ['Sample Counts', 'Determinacy', 'Single Acc', 'Set Size', 'Set Acc', 'U65', 'U80', "RF ACC"]
    data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
    
    data_names = get_imbalance_datasets(data_names)
    
    K = 10
    
    nt = 100
    msl = 5
    alpha_list = np.arange(0,20)/20
    dist_types = ['SE','KLD']
    prediction_type = 'E-admissibility'
    class_weight_types = [None, 'balanced', 'balanced_subsample']
    
    for dist_type in dist_types:
        for class_weight in class_weight_types:
            for d in range(len(data_names)):
                data_name = data_names[d]
                print(data_name)
                data = pd.read_csv('../data/{}.csv'.format(data_name))
                X = np.array(data.iloc[:,:-1])
                y = np.array(data.iloc[:,-1]) 
                classes, class_wise_counts = np.unique(y, return_counts=True)
                class_index = {}
                for c in range(len(classes)):
                    class_index[classes[c]] = c
                
                data_evaluations = np.zeros((K, len(classes), len(evaluation_criteria)))
                kf = KFold(n_splits=K, shuffle=False)
                k = -1
                for train_index, test_index in tqdm(kf.split(X)):
                    k += 1
                    data_evaluations[k,:,0] = class_wise_counts
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
        
                    clf = ProbabilitySetsModel(n_trees=nt, alpha=0, 
                                               p_dist_type=dist_type, 
                                               min_samples_leaf=msl, 
                                               class_weight=class_weight,
                                               random_state=42)
                    clf.fit(X_train, y_train)
                    
                    inner_kf = KFold(n_splits=K, shuffle=False)
                    
                    inner_u65_evaluations = np.zeros(len(alpha_list))
        
                    for inner_train_index, inner_val_index in inner_kf.split(X_train):
                        inner_X_train, inner_X_val = X_train[inner_train_index], X_train[inner_val_index]
                        inner_y_train, inner_y_val = y_train[inner_train_index], y_train[inner_val_index]
        
                        inner_clf = ProbabilitySetsModel(n_trees=nt, alpha=0, 
                                                         p_dist_type=dist_type, 
                                                         min_samples_leaf=msl, 
                                                         class_weight=class_weight,
                                                         random_state=42)
                        inner_clf.fit(inner_X_train, inner_y_train)
        
                        for a in range(len(alpha_list)):
                            inner_clf.alpha = alpha_list[a]
                            inner_u65_evaluations[a] += inner_clf.evaluate(inner_X_val, inner_y_val, prediction_type)['u65 score']
            
                    best_alpha = alpha_list[np.argmax(inner_u65_evaluations)]
                    clf.alpha = best_alpha
                    
                    # conduct class-wise evaluation
                    for c in range(len(classes)):
                        data_evaluations[k,c,0] = sum(y_test == classes[c])
                    predictions = clf.predict(X_test, prediction_type)
                    precise_predictions = clf.rf.predict(X_test)
                    for i in range(len(y_test)):
                        c = class_index[y_test[i]] # index of the true class
                        prediction = predictions[i]
                        if y_test[i] == precise_predictions[i]:
                            data_evaluations[k,c,7] += 1
                            
                        if len(prediction) == 1:
                            data_evaluations[k,c,1] += 1
                            if prediction[0] == y_test[i]:
                                data_evaluations[k,c,2] += 1
                                data_evaluations[k,c,4] += 1
                                data_evaluations[k,c,5] += 1
                                data_evaluations[k,c,6] += 1
                        else:
                            size = len(prediction)
                            data_evaluations[k,c,3] += size
                            if y_test[i] in prediction:
                                data_evaluations[k,c,4] += 1
                                data_evaluations[k,c,5] += (-0.6/(size**2) + 1.6/size)
                                data_evaluations[k,c,6] += (-1.2/(size**2) + 2.2/size)
            
                
                #average across K folds
                over_all= data_evaluations.sum(axis=0)
                
                n_total = over_all[:,0].copy()
                n_determinate = over_all[:,1].copy()
                n_indeterminate = over_all[:,0] - over_all[:,1]
                n_determinate[n_determinate==0] = -1
                n_indeterminate[n_indeterminate==0] = -1
                
                over_all[:,1] /= n_total
                over_all[:,2] /= n_determinate
                over_all[:,3] /= n_indeterminate
                over_all[:,4] /= n_total
                over_all[:,5] /= n_total
                over_all[:,6] /= n_total
                over_all[:,7] /= n_total
                
                over_all[over_all<0] = np.nan
                
                print(over_all)
                
                if class_weight is None:
                    c_w = 'None'
                else:
                    c_w = class_weight
                
                df = pd.DataFrame(over_all.round(4), columns=evaluation_criteria, index=classes)
                df.sort_values("Sample Counts", ascending=True, inplace=True)
                df.to_excel('./imbalance data/{}/class_weight={}/{}_evaluations.xlsx'.format(dist_type, c_w, data_name), index=True)
        
    
        
        
        
        