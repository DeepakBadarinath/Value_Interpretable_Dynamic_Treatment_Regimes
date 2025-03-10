#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 02:25:04 2024

@author: badarinath
"""

import numpy as np
import Decision_tree_set_functions                                             
import pandas as pd                                                            
from Decision_tree_set_functions import DecisionTreeSet                          
import disjoint_box_union                                                      
from disjoint_box_union import DisjointBoxUnion as DBU                          
import math
import importlib
import inspect
import Trees
from Trees import DecisionTree
import matplotlib.pyplot as plt


importlib.reload(Trees)                                                         
importlib.reload(Decision_tree_set_functions)                                        
                                                                                
#%%                                                                             

#Dynamic programming deterministic tree policy algorithm

class DPTPA:
    
    def __init__(self, time_horizon, gamma,
                 decision_tree_sets, transition_kernels):
        
        '''
        The Dynamic Programming Deterministic Tree Policy Algorithm from the   
        paper by 'Interpretable Policies for Ventilator Resource Allocation'   
        by Julien Grand-Clement.                                               
        
        We compute the optimal interpretable policy using a backward induction 
        on the Bellman Equations and evaluation over all possible trees in decision_tree_sets.
        
        Parameters:
        -----------------------------------------------------------------------
        time_horizon : int
                       The time horizon of the total MDP
                                                                                   
        gamma : float
                discount_factor
                                                                                
   decision_tree_sets : list[time_horizon]
                        list of length time_horizon which consists of sets from
                        Decision_tree_set_functions
                                                                                               
   transition_kernels : list[function(s' \in state_spaces[t], s \in state_spaces[t],
                                      a \in action_spaces[], state_space, action_space) \to [0,1]]
   
                        List of length T which consists of probability transition maps.
                        Here the sum of the transition_kernels(s',s,a) for all s' in states = 1
                        
        '''
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.decision_tree_sets = decision_tree_sets
        self.transition_kernels = transition_kernels
    
    
    def compute_optimal_trees(self, tree_count = 1000):
        
        '''
        Computation of the optimal trees for the different timesteps; compute   
        the return over the different timesteps; we assume a uniform distribution
        on the predecessor distributions for all the timesteps.                
        
        -------------------------------------------------------------------------
        
        '''
        
        best_score = -np.inf
        best_tree = None
        
        # Compute optimal tree in the last timestep
        #print(f'We have the following number of trees : {len(self.decision_tree_sets[-1].trees)}')
        
        for i,tree in enumerate(self.decision_tree_sets[-1].trees[:tree_count]):
            
            score = 0.0
            #print(f'Root of tree is {tree.root}')                               
            tree_function = tree.generate_tree_function(tree)
            #print('We get the following after traversing the tree in order')                          
            #print(tree.traverse_in_order())
            #print(f'We are at the {i}th tree')                     
            
            for i,r in enumerate(self.decision_tree_sets[-1].cost_functions):   
                                                                                
                dbu_iter_class = disjoint_box_union.DBUIterator(self.decision_tree_sets[-1].states)
                dbu_iterator = iter(dbu_iter_class)                             
                                                                                
                for s in dbu_iterator:                                          
                                                                                
                    #print(f'State is {s}')
                    
                    #print(f'Tree function value is {tree_function(s)}')                       
                    #print(f'Tree function evaluated on {s} is {tree_function(s)}') 
                    score += r(s, tree_function(s))                                
                    #print(f'To the score, we add {r(s, tree_function(s))}')          
                                                                                                                                                                        
            if score > best_score:                                              
                best_tree = tree                                                
                best_tree_function = tree_function                                              
                
        best_tree.save_tree_as_png('Last_timestep_tree')                        
                                                                                        
        # Update the value function for the last timestep
        def v_T(s):                                                             
            no_of_rewards = len(self.decision_tree_sets[-1].cost_functions)     
            value = 0.0                                                           
            
            for r in self.decision_tree_sets[-1].cost_functions:                
                value += r(s, best_tree_function(s))                             
            
            value = value / no_of_rewards                                       
            return value                                                        
        
        optimal_values = [v_T]                                                  
        optimal_trees = [best_tree]                                             
        
        # Compute optimal trees for the previous timesteps
        for t in reversed(range(self.time_horizon - 1)):
                                                                                
            best_score = -np.inf
            best_tree = None                                                    
            
            for tree in (self.decision_tree_sets[t].trees)[:tree_count]:
            
                tree_function = DecisionTree.generate_tree_function(tree)
            
                for r in self.decision_tree_sets[t].cost_functions:             
                    
                    dbu_iter_class_prev = disjoint_box_union.DBUIterator(self.decision_tree_sets[t].states)
                    dbu_iterator_prev = iter(dbu_iter_class_prev)
                    
                    for s_old in dbu_iterator_prev:                             
                        
                        dbu_iter_class_next = disjoint_box_union.DBUIterator(self.decision_tree_sets[t+1].states)
                        dbu_iterator_next = iter(dbu_iter_class_next)
                                                                                
                        score = 0.0
                        
                        for s_new in dbu_iterator_next:
                        
                            #print(s_old, s_new, best_tree_function(s_old))
                            score += r(s_old, best_tree_function(s_old)) + (self.gamma * 
                                                                   self.transition_kernels[t](s_new, s_old,
                                                                                              best_tree_function(s_old)) *
                                                                   optimal_values[-1](s_new))
                        
                if score > best_score:
                    best_tree = tree
                    best_tree_function = tree_function
                
            
            # Update the value function for the previous timesteps             
            def v(s):                                                           
                no_of_rewards = len(self.decision_tree_sets)                
                value = 0                                                       
                
                for r in self.decision_tree_sets[t].cost_functions:             
                                                                                
                    dbu_iter_class_next = disjoint_box_union.DBUIterator(self.decision_tree_sets[t+1].states)
                    dbu_iterator_next = iter(dbu_iter_class_next)               
                    
                    for s_new in dbu_iterator_next:
                        value += r(s, best_tree_function(s)) + (self.gamma *  self.transition_kernels[t](s_new, s_old,best_tree_function(s_old)) * optimal_values[-1](s_new)) 
                                                                                
                value = value / no_of_rewards                                  
                return value
                    
            optimal_values = [v] + optimal_values                              
            optimal_trees = [best_tree] + optimal_trees                        
                                                                                
                                                                                
            best_tree.save_tree_as_png(f'{t}th_timestep_best_tree')             
                                                                                        
        self.optimal_values = optimal_values                                   
        self.optimal_trees = optimal_trees                                     
                
        return {'Optimal_values' : self.optimal_values,                        
                'Optimal_trees' : self.optimal_trees}                                       
    
    def plot_values(self,
                    error_bounds = [],
                    bellman_error_bounds = []):                                                     
        '''
        Plot the errors obtained after we perform the VIDTR algorithm           
        
        Parameters:
        -----------------------------------------------------------------------
        error_bounds : list
                       The bounds for the errors
        
 bellman_error_bounds : list
                        The bounds for the Bellman errors
 
        '''
        '''
        for t in range(self.optimal_values):
            plt.plot(self.decision_tree_sets[t].states, self.optimal_values[t])
        '''
        
        '''
        plt.plot(np.arange(len(self.optimal_values)), self.optimal_values)
        plt.title('Values with time')
        plt.xlabel('Time')
        plt.ylabel('Optimal values')
        
        plt.legend()
        plt.show()
        '''
        for t in range(len(self.optimal_values)):
            
            print(f'The state space at time {t} is given by:')
            print(self.decision_tree_sets[t].states)
        
        for t, tree1 in enumerate(self.optimal_trees):
            tree1.save_tree_as_png(f'Tree_at_timestep_{t}')
        
#%%