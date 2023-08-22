# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:29:42 2023

@author: zhanghai
"""

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances


class ProbabilitySetsModel:
    def __init__(self, n_trees=100, alpha=0, p_dist_type='SE', min_samples_leaf=3, random_state=42):
        self.n_trees = n_trees
        self.alpha = alpha
        self.p_dist_type = p_dist_type
        self.min_samples_leaf = min_samples_leaf
        self.classes = None
        self.n_class = None
        self.rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=random_state)
        
        
    def fit(self, train_x, train_y):
        self.rf.fit(train_x, train_y)
        
        self.classes = self.rf.classes_
        self.n_class = len(self.rf.classes_)
    
    
    def predict(self, X, prediction_type='Maximality'):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_instance = X.shape[0]
        all_probability_sets = np.zeros((n_instance, self.n_trees, self.n_class))
        for t in range(self.n_trees):
            tree = self.rf.estimators_[t]
            all_probability_sets[:,t,:] = tree.predict_proba(X)
            
        predictions = []
        n_equal = 0
        class2check = 0
        for i in range(n_instance):
            probability_set = all_probability_sets[i,:,:]
            
            if self.p_dist_type == 'L1':
                def L1_distance(p_star, probability_set):
                    total_distance = 0
                    total_distance = np.sum(np.abs(probability_set - p_star))
#                     for p_point in probability_set:
#                         dist_distance = np.sum(np.abs(p_point - p_star))
#                         total_distance += dist_distance
                    return total_distance

                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                               {'type': 'ineq', 'fun': lambda x: x}]

                result = minimize(L1_distance, np.ones(self.n_class)/self.n_class, 
                                  args=(probability_set,), constraints=constraints)
               
#                 p_star = result.x.round(8)
                p_star = result.x
                probability_set = np.concatenate([p_star.reshape((1,-1)), probability_set])
                dists_to_p_star = abs(probability_set - p_star.reshape((1,-1))).sum(axis=1)
                    
            elif self.p_dist_type == 'KL':
                def KL_divergence(p_star, probability_set):
                    epsilon = 1e-10
                    total_distance = 0
                    p_star = np.where(p_star < epsilon, epsilon, p_star)
                    probability_set = np.where(probability_set < epsilon, epsilon, probability_set)
                    total_distance = np.sum(probability_set * np.log(probability_set/p_star))
#                     for p_point in probability_set:
#                         p_point = np.where(p_point < epsilon, epsilon, p_point)
#                         dist_distance = np.sum(p_point * np.log(p_point/p_star))
#                         total_distance += dist_distance
                    return total_distance
                
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                               {'type': 'ineq', 'fun': lambda x: x}]

                result = minimize(KL_divergence, np.ones(self.n_class)/self.n_class, 
                                  args=(probability_set,), constraints=constraints)
                
#                 p_star = result.x.round(5)
                p_star = result.x
                epsilon = 1e-10
                p_star = np.where(p_star < epsilon, epsilon, p_star)
                probability_set = np.where(probability_set < epsilon, epsilon, probability_set)
                probability_set = np.concatenate([p_star.reshape((1,-1)), probability_set])
                dists_to_p_star = (probability_set*np.log(probability_set/(p_star.reshape((1,-1))))).sum(axis=1)
            else: # SE
                p_star = probability_set.mean(axis=0)
                probability_set = np.concatenate([p_star.reshape((1,-1)), probability_set])
                dists_to_p_star = euclidean_distances(p_star.reshape((1,-1)), probability_set)[0]
            
            if prediction_type=='precise':
                predictions.append([self.classes[p_star.argmax()]])
            else: # imprecise predictions
                probability_set = probability_set[np.argsort(dists_to_p_star),:]
                rest_probability_set = probability_set[0:int((1-self.alpha)*self.n_trees)+1, :]
#                 rest_probability_set = np.vstack((rest_probability_set, p_star))
#                 print(rest_probability_set.shape)
#                 print(p_star)
#                 print(rest_probability_set)
                
                if prediction_type=='Maximality':
                    prediction_M_indicator = np.ones(self.n_class)
                    
                    for j in range(self.n_class):
                        compared_classes = np.setdiff1d(range(self.n_class), [j])
                        for k in compared_classes:
                            if np.all(rest_probability_set[:,j] <= rest_probability_set[:,k]):
                                prediction_M_indicator[j] = -1
                    prediction_M_index = np.where(prediction_M_indicator==1)[0]
                    predictions.append(self.classes[prediction_M_index])
                else: # E-Admissibility
                    prediction_E_indicator = np.zeros(self.n_class)
                    prediction_M_indicator = np.ones(self.n_class)
                    for j in range(self.n_class):
                        compared_classes = np.setdiff1d(range(self.n_class), [j])
                        for k in compared_classes:
                            if np.all(rest_probability_set[:,j] <= rest_probability_set[:,k]):
                                prediction_M_indicator[j] = -1
                                prediction_E_indicator[j] = -1
                                break
                        if prediction_E_indicator[j] == -1:
                            continue
                        if rest_probability_set[:,j].max() < 1/self.n_class:
                            prediction_E_indicator[j] = -1
                            continue
                        if rest_probability_set[:,j].max() > 0.5:
                            prediction_E_indicator[j] = 1
                            continue
                        if np.any(np.sum(rest_probability_set[:,j].reshape((-1,1)) > rest_probability_set, axis=1)==self.n_class-1):
                            prediction_E_indicator[j] = 1
                            
                    classes2check = np.where(prediction_E_indicator==0)[0]
                    if len(classes2check) == 0:
                        prediction_E_index = np.where(prediction_E_indicator==1)[0]
                    else:
                        for j in classes2check:
#                             k = np.setdiff1d(class2check, [j])[0]
                            def check_E_admissibility(distributions, class2check):
                                n_distributions = len(distributions)  # number of distributions
                                K = len(distributions[0])  # number of class

                                # to maximize the probability of the class to check
                                c = -distributions[:, class2check]

                                # inequality constraint matrix
                                A_ub = np.zeros((K, n_distributions))
                                for k in range(K):
                                    A_ub[k, :] = distributions[:, k] - distributions[:, class2check]
                                b_ub = np.zeros(K)

                                # Equality constraints (all weights sum to 1)
                                A_eq = np.ones(n_distributions)
                                b_eq = 1

                                # weight bounds
                                bounds = [(0, 1)] * n_distributions

                                # Solve linear programming problems
                                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=[A_eq], b_eq=[b_eq], bounds=bounds)

                                if result.success:
                                    weighted_average = result.x
                                    return weighted_average
                                else:
                                    return None
                            
                            result = check_E_admissibility(rest_probability_set, j)

                            if result is not None:
                                prediction_E_indicator[j] = 1
                            else:
                                prediction_E_indicator[j] = -1

                    if np.all(prediction_E_indicator == prediction_M_indicator):
                        n_equal += 1
                    class2check += np.sum(prediction_E_indicator==0)
                    # here for prediction_E_indicator==0, have to sovle the LPs
                    prediction_E_index = np.where(prediction_E_indicator==1)[0]
                    predictions.append(self.classes[prediction_E_index])

#         print(n_instance, n_equal, class2check)                
        return predictions
    
        
        
    def evaluate(self, X_test, y_test, prediction_type='Maximality'):
        predictions = self.predict(X_test, prediction_type)
        precise_predictions = self.rf.predict(X_test)
        rf_acc = sum(precise_predictions==y_test)/len(y_test)
        determinacy = 0
        single_set_accuracy = 0
        set_accuracy = 0
        set_size = 0
        u65 = 0
        u80 = 0
        rf_abstention_acc = 0
        for i in range(len(y_test)):
            prediction = predictions[i]
            precise_prediction = precise_predictions[i]
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
                if y_test[i] == precise_prediction:
                    rf_abstention_acc += 1
                    
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
            rf_abstention_acc = None
        else:
            set_accuracy /= n_indeterminate
            set_size /= n_indeterminate
            rf_abstention_acc /= n_indeterminate
            
        u65 /= len(y_test)
        u80 /= len(y_test)
                
        return {'rf accuracy': rf_acc,
                 'determinacy': determinacy,
                 'single accuracy': single_set_accuracy,
                 'set accuracy': set_accuracy,
                 'set size': set_size,
                 'u65 score': u65, 
                 'u80 score': u80,
                 'rf abstention accuracy': rf_abstention_acc}