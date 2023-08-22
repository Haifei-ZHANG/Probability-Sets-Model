# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:03:34 2023

@author: zhanghai
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from probability_set_model import ProbabilitySetsModel


if __name__ == "__main__":
    
    
    data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'yeast',
               'wine_quality', 'segment', 'waveform', 'optdigits']
    # data_names = ['libras']
    alpha_list = np.arange(0,21)/20
    K = 10
    evaluations_with_alpha = np.zeros((len(data_names), len(alpha_list), 3))
    
    for d in range(len(data_names)):
        data_name = data_names[d]
        print(data_name)
        data = pd.read_csv('data/{}.csv'.format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])
    
        kf = KFold(n_splits=K, shuffle=False)
    
        for train_index, test_index in tqdm(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            clf = ProbabilitySetsModel(n_trees=100, alpha=0, p_dist_type='SQE', min_samples_leaf=3, random_state=42)
            clf.fit(X_train, y_train)
    
            for a in range(len(alpha_list)):
                clf.alpha = alpha_list[a]
                evaluations_with_alpha[d, a, 0] += clf.rf.score(X_test, y_test)
                if alpha_list[a] == 1:
                    imprecise_score = clf.evaluate(X_test, y_test, 'precise')
                else:
                    imprecise_score = clf.evaluate(X_test, y_test, 'Maximality')
                evaluations_with_alpha[d, a, 1] += imprecise_score['u65 score']
                evaluations_with_alpha[d, a, 2] += imprecise_score['u80 score']
                
                
    #     print(evaluations_with_alpha[d]/K)
    evaluations_with_alpha = evaluations_with_alpha/K
    np.save('results/function of alpha/evaluations_as_function_of_alpha.npy', evaluations_with_alpha)
    

    # draw figs
    for d in range(len(data_names)):
        evaluation = evaluations_with_alpha[d,:,:]
        u65_best_alpha = alpha_list[np.argmax(evaluation[:,1])]
        best_u65 = evaluation[:,1].max()
        u80_best_alpha = alpha_list[np.argmax(evaluation[:,2])]
        best_u80 = evaluation[:,2].max()
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(alpha_list, evaluation[:,0], linewidth=2, color='tab:blue', label='RF Accuracy')
        plt.plot(alpha_list, evaluation[:,1], linewidth=2, color='tab:orange', label='Ours $u_{65}$')
        plt.plot(alpha_list, evaluation[:,2], linewidth=2, color='tab:green', label='Ours $u_{80}$')
        
        bottom = plt.ylim()[0]
        plt.plot([u65_best_alpha, u65_best_alpha], [bottom, best_u65], linewidth=2, 
                 alpha=1, linestyle=':', color='tab:orange', label=r'$\alpha^*_{u_{65}}=$'+str(u65_best_alpha))
        plt.plot([u80_best_alpha, u80_best_alpha], [bottom, best_u80], linewidth=2, 
                 alpha=1, linestyle=':', color='tab:green', label=r'$\alpha^*_{u_{80}}=$'+str(u80_best_alpha))
        
        plt.xlabel('alpha', fontsize=12)
        plt.ylabel('score', fontsize=12)
        plt.xticks(alpha_list[::2], fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/function of alpha/{}_plot.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()