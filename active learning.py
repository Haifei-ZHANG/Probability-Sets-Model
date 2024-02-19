# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:10:02 2024

@author: vlNguyen
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from skclean.simulate_noise import flip_labels_uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.model_selection import train_test_split
import random
#from sklearn.ensemble import ExtraTreesClassifier
#import multiprocessing
#import os
#import time
#from sklearn.preprocessing import LabelEncoder
#from tqdm import tqdm

#from scipy import stats
from timeit import default_timer as timer


if __name__ == "__main__":
#    data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
#               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
    data_names = ['wine_quality', 'segment']    
   
    # data_names = ['wine_quality']
    alpha_list = np.linspace(0, 1, 101).round(2)
    n_repetition = 10
    folds = 5
    ensemble_size = 100
    epsilon_list = np.linspace(0, 1, 101).round(2)

    for d in tqdm(range(len(data_names))):
        data_name = data_names[d]
        print(data_name)
        data = pd.read_csv("data/{}.csv".format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])
        
#        pool_length = int((len(y)*3)/4) # 5 % train, 75 % pool, 20 % test
 
        pool_length = int(len(y)*0.77) # 3 % train, 77 % pool, 20 % test

        if pool_length < 500:
           batch_size = 3
        else:
           batch_size = 5 
    
        budget = int(pool_length/batch_size)            
        SM_accuracy = np.zeros(budget+2)  
        QB_accuracy = np.zeros(budget+2)
        SM_classes = np.zeros(budget+2)
        QB_classes = np.zeros(budget+2)       
        SM_uncertainty_scores = np.zeros(budget)  
        QB_uncertainty_scores = np.zeros(budget)
#        EC_uncertainty_scores = np.zeros(budget)
        
        check_stop_place = int(budget/100)
        n_thresholds = 20
        thresholds = [0.05 + n*0.05 for n in range(n_thresholds)]
        print(thresholds)
        SM_used_budget = np.zeros(n_thresholds)  
        QB_used_budget = np.zeros(n_thresholds)
 
        SM_gained_accuracy = np.zeros(n_thresholds)  
        QB_gained_accuracy = np.zeros(n_thresholds)
        noisy_level = 0.25
        
        for repetition in tqdm(range(n_repetition)):
            print("repetition:", repetition)    
#        def active_learning(X, y, pool_length, budget, batch_size, random_seed):
#            probabilistic_accuracy = np.zeros(budget+1)  
#            QB_accuracy = np.zeros(budget+1) 
            fo = 0 
            Kf = KFold(n_splits=folds, random_state= 42 + repetition*10, shuffle=True)
            for train_index, test_index in Kf.split(y):  
                print("repetition, fold = ", repetition, fo)
    #            for test_index, train_index in Kf.split(y):  # swap train indices and test indices to have less training data
    #                print("fold:", fo)
                fo += 1
    #            print(train_index)
#                X_train, y_train,  X_test, y_test = X[train_index],  y[train_index], X[test_index], y[test_index]
                X_train, y_train_clean,  X_test, y_test = X[train_index],  y[train_index], X[test_index], y[test_index]
                y_train = flip_labels_uniform(y_train_clean, noisy_level)  # Flip labels of noisy_level % samples

                clf = RandomForestClassifier(n_estimators = ensemble_size, min_samples_leaf=5,random_state=42)
                clf.fit(X_train, y_train)
                class_list = list(clf.classes_)
                n_class_total = len(class_list)  
                final_accuracy = clf.score(X_test, y_test)
                SM_accuracy[budget+1] += final_accuracy  
                SM_classes[budget+1] += 1  
                QB_accuracy[budget+1] += final_accuracy  
                QB_classes[budget+1] += 1  
#                EC_accuracy[budget+1] += final_score                                 
                index_list = [i for i in range(len(y_train))]
                random.seed(42)
                pool_index = list(random.sample(index_list, pool_length))
                short_train_index = [i for i in range(len(y_train)) if i not in pool_index]
    #                y_train =  y[train_index]
    
                # SM sampling
                start = timer()
                train_index_current = [i for i in short_train_index]
                pool_index_current = [i for i in pool_index]
                clf = RandomForestClassifier(n_estimators = ensemble_size, min_samples_leaf=5,random_state=42)
                clf.fit(X_train[train_index_current], y_train[train_index_current])                
                SM_accuracy[0] += clf.score(X_test, y_test)   
                SM_classes[0] += len(list(clf.classes_))/n_class_total   

                SM_filled_check_point = np.zeros(n_thresholds)  

                for iteration in range(budget):
                    class_list = list(clf.classes_)
                    n_class = len(class_list)    
                    n_instance = len(pool_index_current)                   
                    predicted_probabilities = clf.predict_proba(X_train[pool_index_current])
                    predicted_classes = clf.predict(X_train[pool_index_current])
                    max_p_index = np.argsort(predicted_probabilities)[:, -1]    
                    second_max_p_index = np.argsort(predicted_probabilities)[:, -2]                   
                    margins = np.zeros(n_instance)
                    for i in range(n_instance):
                       margins[i] = predicted_probabilities[i, max_p_index[i]] - predicted_probabilities[i, second_max_p_index[i]]                       
                    instance_order_sm = list(np.argsort(margins))
                    chosen_index = [pool_index_current[x] for x in instance_order_sm[:batch_size]]
                    for c in chosen_index:
                        pool_index_current.remove(c)
                        train_index_current.append(c)                    
                    clf = RandomForestClassifier(n_estimators = ensemble_size, min_samples_leaf=5,random_state=42)
                    clf.fit(X_train[train_index_current], y_train[train_index_current]) 
                    current_accuracy = clf.score(X_test, y_test)
                    SM_accuracy[iteration+1] += current_accuracy
                    SM_classes[iteration+1] += len(list(clf.classes_))/n_class_total 
                    current_score= np.mean([margins[i] for i in instance_order_sm[:batch_size]])                        
                    SM_uncertainty_scores[iteration] += current_score
                    if iteration >= check_stop_place:
                        for th in range(n_thresholds):
                            if current_score > thresholds[th] and SM_filled_check_point[th] == 0:
                                SM_filled_check_point[th] = 1
                                SM_gained_accuracy[th] += current_accuracy
                                SM_used_budget[th] += (iteration+1)/budget                                               
                for th in range(n_thresholds):
                    if SM_filled_check_point[th] == 0:
                        SM_gained_accuracy[th] += final_accuracy
                        SM_used_budget[th] += 1   
                end = timer()
                print("SM sampling (in seconds): ", end - start)                    
                    
                 # QB sampling
                start = timer()
                train_index_current = [i for i in short_train_index]
                pool_index_current = [i for i in pool_index] 
                clf = RandomForestClassifier(n_estimators = ensemble_size, min_samples_leaf=5,random_state=42)
                clf.fit(X_train[train_index_current], y_train[train_index_current])
                QB_accuracy[0] += clf.score(X_test, y_test)                
                QB_classes[0] += len(list(clf.classes_))/n_class_total   
                
                QB_filled_check_point = np.zeros(n_thresholds)  

                for iteration in range(budget):
                     class_list = list(clf.classes_)
                     n_class = len(class_list)    
                     n_instance = len(pool_index_current)
                     all_probability_sets = np.zeros((n_instance, ensemble_size, n_class))
                     for t in range(ensemble_size):
                         tree = clf.estimators_[t]
                         all_probability_sets[:,t,:] = tree.predict_proba(X_train[pool_index_current])
                    
                     predicted_probabilities = clf.predict_proba(X_train[pool_index_current])
                     predicted_classes = clf.predict(X_train[pool_index_current])
                     max_p_index = np.argsort(predicted_probabilities)[:, -1]    
                     second_max_p_index = np.argsort(predicted_probabilities)[:, -2]                   
                     margins = np.zeros(n_instance)
                     for i in range(n_instance):
                        margins[i] = predicted_probabilities[i, max_p_index[i]] - predicted_probabilities[i, second_max_p_index[i]]                       
                    
                     smallest_alphas = np.zeros(n_instance)
                    
                     for i in range(n_instance):
                         probability_set = all_probability_sets[i,:,:]
                         p_star = predicted_probabilities[i]
                         dists_to_p_star = euclidean_distances(p_star.reshape((1,-1)), probability_set)[0]
                         sorted_probability_set = probability_set[np.argsort(dists_to_p_star),:]
                         max_p_index_ensemble_members = np.argsort(sorted_probability_set)[:, -1]
                         second_max_p_index_ensemble_members = np.argsort(sorted_probability_set)[:, -2]
                         smallest_alphas[i] = np.sum([margins[i]]+[sorted_probability_set[i,max_p_index_ensemble_members[i]] - sorted_probability_set[i,second_max_p_index_ensemble_members[i]] for i in range(ensemble_size)])/(ensemble_size+1)
                         for position in range(ensemble_size):
                             if predicted_classes[i] != class_list[max_p_index_ensemble_members[position]]:  
                                 smallest_alphas[i] = np.sum([margins[i]] + [sorted_probability_set[i,max_p_index_ensemble_members[i]]  - sorted_probability_set[i,second_max_p_index_ensemble_members[i]] for i in range(position)])/(ensemble_size+1)
                                 break
                     instance_order_Ead = list(np.argsort(smallest_alphas))
  #                   instance_order_Ead = np.array(instance_order_Ead)[::-1]   
                     chosen_index = [pool_index_current[x] for x in instance_order_Ead[:batch_size]]
                     for c in chosen_index:
                         pool_index_current.remove(c)
                         train_index_current.append(c)
                     clf = RandomForestClassifier(n_estimators = ensemble_size, min_samples_leaf=5,random_state=42)
                     clf.fit(X_train[train_index_current], y_train[train_index_current]) 
                     current_accuracy = clf.score(X_test, y_test)
                     QB_accuracy[iteration+1] += current_accuracy
                     QB_classes[iteration+1] += len(list(clf.classes_))/n_class_total 
                     current_score= np.mean([smallest_alphas[i] for i in instance_order_Ead[:batch_size]])                        
                     QB_uncertainty_scores[iteration] += current_score
                     if iteration >= check_stop_place:
                        for th in range(n_thresholds):
                            if current_score >= thresholds[th] and QB_filled_check_point[th] == 0:
                                QB_filled_check_point[th] = 1
                                QB_gained_accuracy[th] += current_accuracy
                                QB_used_budget[th] += (iteration+1)/budget                
                for th in range(n_thresholds):
                    if QB_filled_check_point[th] == 0:
                        QB_gained_accuracy[th] += final_accuracy
                        QB_used_budget[th] += 1   
                end = timer()
                print("QB sampling (in seconds): ", end - start)
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')

#        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('# of batches', fontsize=12)
        plt.ylabel('Accuracies (%)', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        plt.savefig('results/active learning/{}_SM_noisy_label_accuracy-vs_n_queries.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        
#        plt.savefig('results/active learning/{}_accuracy-vs_n_queries.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_classes/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_classes/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')

#        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('# of batches', fontsize=12)
        plt.ylabel('Classes (%)', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)

        plt.savefig('results/active learning/{}_SM_noisy_label_known_classes-vs_n_queries.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        
#        plt.savefig('results/active learning/{}_known_classes-vs_n_queries.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_uncertainty_scores/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_uncertainty_scores/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
        
        plt.xlabel('# of batches', fontsize=12)
        plt.ylabel('Chosen scores', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/active learning/{}_SM_noisy_label_scores-vs_n_queries.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

#        plt.savefig('results/active learning/{}_scores-vs_n_queries.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_gained_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_gained_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
#        plt.plot(EC_uncertainty_scores/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('Threshold (= 0.05 + x*0.05)', fontsize=12)
        plt.ylabel('Gained accuracy', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)

        plt.savefig('results/active learning/{}_SM_noisy_label_gained-accuracy-vs_threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        
#        plt.savefig('results/active learning/{}_gained-accuracy-vs_threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_used_budget/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_used_budget/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
#        plt.plot(EC_uncertainty_scores/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('Threshold (= 0.05 + x*0.05)', fontsize=12)
        plt.ylabel('Used budget', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/active learning/{}_SM_noisy_label_used_budget-vs_threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

#        plt.savefig('results/active learning/{}_used_budget-vs_threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()