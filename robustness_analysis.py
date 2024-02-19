# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:32:02 2023

@author: zhanghai
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from probability_set_model import ProbabilitySetsModel



if __name__ == "__main__":
    data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
    n_it = 10
    K = 5
    alpha_list = np.linspace(0, 1, 11).round(2)
    accuracy2save = np.zeros((len(data_names), len(alpha_list)))
    determinacy2save = np.zeros((len(data_names), len(alpha_list)))
    
    for d in range(len(data_names)):
        data_name = data_names[d]
        data = pd.read_csv("data/{}.csv".format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])
        classes = np.unique(y)
        
        accuracy_array = np.zeros((n_it*K, len(alpha_list)))   
        n_determinate = np.zeros((n_it*K, len(alpha_list)))
        for it in tqdm(range(n_it)):
            kf = KFold(n_splits=K, shuffle=True)
            k = -1
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # in X_train, choose certain proportion of instance to change its label
                # instance_select = np.random.choice(len(y_train),int(0.25*len(y_train)),replace=False)
                # for instance_index in instance_select:
                #     candidate_y = np.setdiff1d(classes, y_train[instance_index])
                #     y_train[instance_index] = candidate_y[np.random.choice(len(candidate_y), 1)[0]]
            
                clf = ProbabilitySetsModel(n_trees=100, alpha=0, min_samples_leaf=5, p_dist_type='SQE',random_state=None)
                clf.fit(X_train, y_train)
                
                determinate_index = {}
                indeterminate_index = np.arange(len(y_test))
                
                k += 1
                for a in range(len(alpha_list)):
                    alpha = alpha_list[a]
                    clf.alpha=alpha
                    if a==0:
                        determinate_index[alpha] = []
                    else:
                        determinate_index[alpha] = determinate_index[alpha_list[a-1]].copy()
                    n_determinate[it*K+k,a] = n_determinate[it*K+k, a-1]
                    
                    for i in indeterminate_index:
                        prediction = clf.predict(X_test[i], prediction_type='E-ad')[0]
                        if len(prediction)==1:
                            n_determinate[it*K+k,a] += 1
                            determinate_index[alpha].append(i)
                            indeterminate_index = np.delete(indeterminate_index, np.where(indeterminate_index == i))
                    if len(determinate_index[alpha])>0:
                        accuracy_array[it*K+k, a] = clf.rf.score(X_test[determinate_index[alpha]], y_test[determinate_index[alpha]])
                    else:
                        accuracy_array[it,a] = np.nan
                    # print(round(alpha,2), len(determinate_index[alpha]), round(accuracy_array[a],4))
        accuary_list = np.zeros(len(alpha_list))
        determinacy_list = np.zeros(len(alpha_list))
        for a in range(len(alpha_list)):
            valid_index = ~np.isnan(accuracy_array[:,a])
            accuracy_a =  accuracy_array[valid_index, a]
            weights = n_determinate[valid_index, a]/sum(n_determinate[valid_index, a])
            accuary_list[a] = sum(accuracy_a*weights)
            
            determinacy_a = n_determinate[:, a] / n_determinate[:, -1]
            determinacy_list[a] = determinacy_a.mean()
            
            
        accuracy2save[d] = accuary_list
        determinacy2save[d] = determinacy_list
        
        print('\n', data_name, '\n', accuracy2save[d], '\n',determinacy2save[d])
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(alpha_list, accuracy2save[d], linewidth=2, color='tab:blue')
        
        plt.xlabel('alpha', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.xticks(alpha_list, fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/robustness analysis/{}_accuracy.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(alpha_list, determinacy2save[d], linewidth=2, color='tab:blue')
        
        plt.xlabel('alpha', fontsize=12)
        plt.ylabel('determinacy', fontsize=12)
        plt.xticks(alpha_list, fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/robustness analysis/{}_determiancy.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
    np.save('results/robustness analysis/accuracy.npy', accuracy2save)
    np.save('results/robustness analysis/determiancy.npy', determinacy2save)  
    # draw figs
    # for d in range(len(data_names)):
        
    #     fig = plt.figure(figsize=(6,4))
        
    #     plt.plot(alpha_list, accuracy2save[d], linewidth=2, color='tab:blue')
        
    #     plt.xlabel('alpha', fontsize=12)
    #     plt.ylabel('accuracy', fontsize=12)
    #     plt.xticks(alpha_list, fontsize=12)
    #     plt.yticks(fontsize=12)
    #     # plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
    #     plt.savefig('results/robustness analysis/{}_accuracy.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
    #     plt.show()
    #     plt.close()
        
        
    #     fig = plt.figure(figsize=(6,4))
        
    #     plt.plot(alpha_list, determinacy2save[d], linewidth=2, color='tab:blue')
        
    #     plt.xlabel('alpha', fontsize=12)
    #     plt.ylabel('determinacy', fontsize=12)
    #     plt.xticks(alpha_list, fontsize=12)
    #     plt.yticks(fontsize=12)
    #     # plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
    #     plt.savefig('results/robustness analysis/{}_determiancy.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
    #     plt.show()
    #     plt.close()
    # print(alpha_list, accuracy_array)
    
    # dist_types = ['SQE'] #, 'L1', 'KL'
    # for d_type in dist_types:
    #     print(d_type)
    #     clf.p_dist_type = d_type
    #     print(clf.evaluate(X_test, y_test, 'Maximality'))
    #     print(clf.evaluate(X_test, y_test, 'EAD'))