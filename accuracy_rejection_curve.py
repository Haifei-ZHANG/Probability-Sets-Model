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
#from sklearn.model_selection import train_test_split
#from probability_set_model import ProbabilitySetsModel
from skclean.simulate_noise import flip_labels_uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.preprocessing import LabelEncoder
#from tqdm import tqdm
from timeit import default_timer as timer

from scipy import stats

if __name__ == "__main__":
    data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
    
    # data_names = ['wine_quality']
    alpha_list = np.linspace(0, 1, 101).round(2)
    epsilon_list = np.linspace(0, 1, 101).round(2)
    n_repetition = 20
    folds = 5
    ensemble_size = 100
    
    n_thresholds = 20
    thresholds = [0.05 + n*0.05 for n in range(n_thresholds)]
        
    for d in tqdm(range(len(data_names))):
        data_name = data_names[d]
        print(data_name)
        data = pd.read_csv("data/{}.csv".format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])
        
        test_lengths = []
        p_values = []

        for repetition in range(n_repetition):
            Kf = KFold(n_splits=folds, random_state= 42 + repetition*10, shuffle=True)
#            for train_index, test_index in Kf.split(y):   
            for test_index, train_index in Kf.split(y):    # swap train indices and test indices to have less training data           
                test_lengths.append(len(test_index))
        min_test_lengths = min(test_lengths)  
        SM_accuracy_acceptance_rate = np.zeros(min_test_lengths)              
        QB_accuracy_acceptance_rate = np.zeros(min_test_lengths)
        SM_score_acceptance_rate = np.zeros(min_test_lengths)              
        QB_score_acceptance_rate = np.zeros(min_test_lengths)
        SM_accuracy_threshold = np.zeros(min_test_lengths)              
        QB_accuracy_threshold = np.zeros(min_test_lengths)
        SM_acceptance_rate_threshold = np.zeros(min_test_lengths)              
        QB_acceptance_rate_threshold = np.zeros(min_test_lengths)        
        SM_acceptance_rate = np.zeros(n_thresholds)  
        QB_acceptance_rate = np.zeros(n_thresholds)     
        SM_gained_accuracy = np.zeros(n_thresholds)  
        QB_gained_accuracy = np.zeros(n_thresholds)
        
#        noisy_level = 0.25
        for repetition in range(n_repetition):
            print("repetition:", repetition)
            fo = 0 
            Kf = KFold(n_splits=folds, random_state= 42 + repetition*10, shuffle=True)
#            for original_train_index, original_test_index in Kf.split(y):  
            start = timer()

            for original_test_index, original_train_index in Kf.split(y):  # swap train indices and test indices to have less training data
#                print("fold:", fo)
                fo += 1
                shifted_length = len(original_test_index) - min_test_lengths
                test_index = original_test_index[:min_test_lengths]
                train_index = np.concatenate((original_train_index, original_test_index[min_test_lengths:]))
    #            print(train_index)
                X_train, X_test, y_test = X[train_index], X[test_index], y[test_index]
                y_train =  y[train_index]
#                y_train_clean = y[train_index]
#                y_train = flip_labels_uniform(y_train_clean, noisy_level)  # Flip labels of noisy_level % samples
                clf = RandomForestClassifier(n_estimators = ensemble_size, min_samples_leaf=5,random_state=None)
                clf.fit(X_train, y_train)

                SM_filled_check_point = np.zeros(n_thresholds)  
                QB_filled_check_point = np.zeros(n_thresholds)  
                
                n_instance = len(y_test)
                class_list = list(clf.classes_)
                n_class = len(class_list)
                
                all_probability_sets = np.zeros((n_instance, ensemble_size, n_class))
                for t in range(ensemble_size):
                    tree = clf.estimators_[t]
                    all_probability_sets[:,t,:] = tree.predict_proba(X_test)
                
                # probabilistic method
                predicted_probabilities = clf.predict_proba(X_test)
                predicted_classes = clf.predict(X_test)
                correctness = np.where(predicted_classes == y_test, 1, 0)
                max_p_index = np.argsort(predicted_probabilities)[:, -1]
                second_max_p_index = np.argsort(predicted_probabilities)[:, -2]
                
                margins = np.zeros(len(y_test))
                for i in range(len(y_test)):
                    margins[i] = predicted_probabilities[i, max_p_index[i]] - predicted_probabilities[i, second_max_p_index[i]]
                instance_order_sm = list(np.argsort(margins))
                for i in range(len(y_test)):
                    current_accuracy = np.mean(correctness[instance_order_sm[i:]])
                    SM_accuracy_acceptance_rate[i] += current_accuracy
                    current_score = margins[instance_order_sm[i]]  
                    SM_score_acceptance_rate[i] += current_score             
                    for th in range(n_thresholds):
                        if  margins[instance_order_sm[i]] > thresholds[th] and SM_filled_check_point[th] == 0:
                            SM_filled_check_point[th] = 1
                            SM_gained_accuracy[th] += current_accuracy
                            SM_acceptance_rate[th] += len(instance_order_sm[i:])/n_instance  
                for i in range(len(y_test), min_test_lengths):
                    SM_accuracy_acceptance_rate[i] += current_accuracy
                    SM_score_acceptance_rate[i] += current_score             
                for th in range(n_thresholds):
                    if SM_filled_check_point[th] == 0:
                        SM_gained_accuracy[th] += 1
                        SM_acceptance_rate[th] += 0 
                        
                smallest_alphas = np.zeros(n_instance)
                
                for i in range(len(y_test)):
                    probability_set = all_probability_sets[i,:,:]
                    p_star = predicted_probabilities[i]
                    dists_to_p_star = euclidean_distances(p_star.reshape((1,-1)), probability_set)[0]
                    sorted_probability_set = probability_set[np.argsort(dists_to_p_star),:]
                    max_p_index_ensemble_members = np.argsort(sorted_probability_set)[:, -1]
                    second_max_p_index_ensemble_members = np.argsort(sorted_probability_set)[:, -2]
                    smallest_alphas[i] = np.sum([margins[i]] + [sorted_probability_set[i,max_p_index_ensemble_members[i]] - sorted_probability_set[i,second_max_p_index_ensemble_members[i]] for i in range(ensemble_size)])/(ensemble_size +1)
                    for position in range(ensemble_size):
                        if predicted_classes[i] != class_list[max_p_index_ensemble_members[position]]:
                           smallest_alphas[i] = np.sum([margins[i]] + [sorted_probability_set[i,max_p_index_ensemble_members[i]]  - sorted_probability_set[i,second_max_p_index_ensemble_members[i]] for i in range(position)])/(ensemble_size +1)
                           break
                instance_order_Ead = list(np.argsort(smallest_alphas))

                for i in range(len(y_test)):
                    current_accuracy = np.mean(correctness[instance_order_Ead[i:]])  
                    current_score = smallest_alphas[instance_order_Ead[i]]            
                    QB_accuracy_acceptance_rate[i] += current_accuracy  
                    QB_score_acceptance_rate[i] +=  current_score            
                    for th in range(n_thresholds):
                        if  smallest_alphas[instance_order_Ead[i]] > thresholds[th] and QB_filled_check_point[th] == 0:
                            QB_filled_check_point[th] = 1
                            QB_gained_accuracy[th] += current_accuracy
                            QB_acceptance_rate[th] += len(instance_order_Ead[i:])/n_instance                 
                for i in range(len(y_test), min_test_lengths):
                    QB_accuracy_acceptance_rate[i] += current_accuracy  
                    QB_score_acceptance_rate[i] +=  current_score            
                for th in range(n_thresholds):
                    if QB_filled_check_point[th] == 0:
                        QB_gained_accuracy[th] += 1
                        QB_acceptance_rate[th] += 0  
            end = timer()
            print("Done (in seconds): ", end - start)     

#         fig = plt.figure(figsize=(6,4))
        
#         plt.plot(SM_accuracy_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
#         plt.plot(QB_accuracy_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
# #        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
#         plt.xlabel('# rejection', fontsize=12)
#         plt.ylabel('Accuracy', fontsize=12)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
#         plt.savefig('results/score-rejection-curves/{}_SM_noisy_label_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
# #        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

#         plt.show()
#         plt.close()

#         fig = plt.figure(figsize=(6,4))
        
#         plt.plot(SM_score_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
#         plt.plot(QB_score_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
# #        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
#         plt.xlabel('# rejection', fontsize=12)
#         plt.ylabel('Score', fontsize=12)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
#         plt.savefig('results/score-rejection-curves/{}_SM_noisy_label_score-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
# #        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

#         plt.show()
#         plt.close()      
        
        
#         fig = plt.figure(figsize=(6,4))
        
#         plt.plot(SM_gained_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
#         plt.plot(QB_gained_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
# #        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
#         plt.xlabel('Threshold (= 0.05 + x*0.05)', fontsize=12)
#         plt.ylabel('Gained accuracy', fontsize=12)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
#         plt.savefig('results/score-rejection-curves/{}_SM_noisy_label_gained_accuracy-threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
# #        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

#         plt.show()
#         plt.close()     
        
        
#         fig = plt.figure(figsize=(6,4))
        
#         plt.plot(SM_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
#         plt.plot(QB_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
# #        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
#         plt.xlabel('Threshold (= 0.05 + x*0.05)', fontsize=12)
#         plt.ylabel('Acceptance rate', fontsize=12)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
#         plt.savefig('results/score-rejection-curves/{}_SM_noisy_label_acceptance_rate-threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
# #        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

#         plt.show()
        
        plt.close()                      
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_accuracy_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_accuracy_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
#        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('# rejection', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/score-rejection-curves/{}_SM_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
#        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_score_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_score_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
#        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('# rejection', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/score-rejection-curves/{}_SM_score-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
#        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

        plt.show()
        plt.close()      
        
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_gained_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_gained_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
#        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('Threshold (= 0.05 + x*0.05)', fontsize=12)
        plt.ylabel('Gained accuracy', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/score-rejection-curves/{}_SM_gained_accuracy-threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
#        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

        plt.show()
        plt.close()     
        
        
        fig = plt.figure(figsize=(6,4))
        
        plt.plot(SM_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:red', label='SM approach')
        plt.plot(QB_acceptance_rate/(folds*n_repetition), linewidth=0.5, color='tab:cyan', label='QB approach')
#        plt.plot(EC_accuracy/(folds*n_repetition), linewidth=0.5, color='tab:green', label='EC approach')
        
        plt.xlabel('Threshold (= 0.05 + x*0.05)', fontsize=12)
        plt.ylabel('Acceptance rate', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)
        
        plt.savefig('results/score-rejection-curves/{}_SM_acceptance_rate-threshold.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
#        plt.savefig('results/[10 x 10 cross-validation with train = 9 folds] RF accuracy-rejection curve/{}_accuracy-rejection.png'.format(data_names[d]), bbox_inches='tight', dpi=600)

        plt.show()
        plt.close()   
        

