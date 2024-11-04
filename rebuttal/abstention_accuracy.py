# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:26:16 2024

@author: zhanghai
"""

import numpy as np
import matplotlib.pyplot as plt



# without noise
evaluations = np.load("./imprecise prediction/min_samples_leaf=5/total_evaluations.npy")
print(evaluations)
recall = evaluations[3, :, :]
abstention_acc = evaluations[-2, :, :]
n_data = recall.shape[0]
n_model = recall.shape[1]
data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
label_list = ['a','b','c','d','e','f','g','h','i','j','k','l']
model_names = ['alpha=0', 'KL-Ead']

# by model
for m in range(n_model):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    bottom, top = plt.ylim()
    left, right =  plt.xlim()
    # ax.spines['bottom'].set_position(('data', bottom))
    # ax.spines['left'].set_position(('data', left))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0,1,11))
    plt.xlabel('RF', fontsize=12, loc='center')
    plt.ylabel(model_names[m], fontsize=12, loc='center')

    plt.plot([0,1],[0,1], linewidth=1, linestyle = '--', color='black')
    for d in range(n_data):
        plt.text(abstention_acc[d, m], recall[d, m], s=label_list[d], fontdict={'family':'serif', 'size': 16, 'color':'tab:blue'})
    plt.savefig('abestention accuracy no noise {} vs RF.png'.format(model_names[m]), bbox_inches='tight', dpi=600)
    plt.savefig('abestention accuracy no noise {} vs RF.pdf'.format(model_names[m]), bbox_inches='tight', dpi=600)
    plt.show()
    
    
    
    
# with noise    
evaluations = np.load("./noisy data imprecise prediction/min_samples_leaf=5/total_evaluations.npy")
recall = evaluations[3, :, :]
abstention_acc = evaluations[-2, :, :]
n_data = recall.shape[0]
n_model = recall.shape[1]
data_names = ['wine', 'seeds', 'glass', 'ecoli', 'dermatology', 'libras',
               'forest', 'balance_scale', 'vehicle', 'vowel', 'wine_quality', 'segment']
label_list = ['a','b','c','d','e','f','g','h','i','j','k','l']
model_names = ['alpha=0', 'KL-Ead']

# by model
for m in range(n_model):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    bottom, top = plt.ylim()
    left, right =  plt.xlim()
    # ax.spines['bottom'].set_position(('data', bottom))
    # ax.spines['left'].set_position(('data', left))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0,1,11))
    plt.xlabel('RF', fontsize=12, loc='center')
    plt.ylabel(model_names[m], fontsize=12, loc='center')

    plt.plot([0,1],[0,1], linewidth=1, linestyle = '--', color='black')
    for d in range(n_data):
        plt.text(abstention_acc[d, m], recall[d, m], s=label_list[d], fontdict={'family':'serif', 'size': 16, 'color':'tab:blue'})
    plt.savefig('abestention accuracy/with noise {} vs RF.png'.format(model_names[m]), bbox_inches='tight', dpi=600)
    # plt.savefig('abestention accuracy/with noise {} vs RF.pdf'.format(model_names[m]), bbox_inches='tight', dpi=600)
    plt.show()