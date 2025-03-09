#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:32:00 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import inspect
from itertools import combinations, product

#%%

class DecisionTreeSet:
                                                                                    
    def __init__(self, dimension, state_bounds, actions, cost_functions,
                 trees = [], max_depth = 0, max_complexity = 0,
                 state_differences = []):
        
        '''
        The set of decision trees are characterized by possible states, actions,
        and cost functions.
                                                                                
        States here are a subset of \mathbb{R}^d. Actions we assume can be floats
        or a subset of \mathbb{R}^d. 
        
        Cost functions are functions from the state space,
        actions to the real line.
        
        Parameters:
        -----------------------------------------------------------------------
        dimension : int
                    The dimension of the state space
        
        state_bounds : np.array[d,2]
                       The bounds for the state space in the different dimensions
        
        actions : list
                  List of possible actions 
        
        cost_functions : list[T]
                         Function from the states times actions to reals
        
        trees : list
                Each tree is a list with dictonaries ({dims : [d1,d2,...,dk],
                                                       vals : [a1,a2,...,ak],
                                                       signs : [+/-1, +/-1, ..., +/-1],
                                                       is_leaf : True/False,
                                                       action : Element from action})
        '''
        
        self.dimension = dimension
        self.state_bounds = state_bounds
        self.actions = actions
        self.cost_functions = cost_functions
        
        
        if len(trees) != 0:
            self.trees = trees
        
        else:
            
            self.trees = self.generate_tree_functions(max_depth, max_complexity,
                                                      state_differences)
        
    
    def generate_trees(self, max_depth, max_complexity,
                       state_differences, state_bounds, existing_trees = []):
        
        '''
        Generate all possible trees which have maximum possible depth as max_depth,
        maximum possible complexity as max_complexity and spaced over 
        state_differences with respect to the state_bounds variable

        Parameters:
        -----------------------------------------------------------------------
        max_depth : int                                                        
                    Maximum allowable depth for a tree                         
        
        max_complexity : int
                         Maximum allowed complexity per node in a tree          
        
        state_differences : np.array
                            The differences per dimension of the state space we
                            split in

        Returns:
        -----------------------------------------------------------------------
        trees : list of dictonaries                
                Each dictonary looks like {dims : [d1,d2,...,dk],
                                           vals : [a1,a2,...,ak],
                                           signs : [+/-1, +/-1, ..., +/-1],
                                           is_leaf : True/False,
                                           action : float/np.array}
        '''
        
        
        
        k_tuples = DecisionTreeSet.generate_k_tuples(self.dimension,
                                                     max_complexity)
        
        for k_tuple in k_tuples:
            list_of_lists = []
            
            for dim in k_tuple:
                list_of_lists.append(np.arange(self.state_bounds[dim, 0],
                                               self.state_bounds[dim, 1],
                                               self.state_differences[dim]))
                
            possible_vals = list(product(list_of_lists))
            possible_two_tuples = list(combinations([-1,1]), len(k_tuple))
            
            for possible_val in possible_vals:
                for possible_signs in possible_two_tuples:
                    
                    existing_trees.append({'vals':possible_val,
                                           'dims':k_tuple,
                                           'signs':possible_signs})
                    
            
    
    @staticmethod
    def generate_k_tuples(d, j):
        """
        Generates all possible k-tuples for k <= j from the list [1, 2, ..., d].

        Parameters:
        -----------------------------------------------------------------------
         d: Integer, the length of the list [1, 2, ..., d].
         j: Integer, the upper limit for the tuple size (k).

        Returns:
        -----------------------------------------------------------------------
         List of tuples containing all possible k-tuples for k <= j.
        
        """
        elements = list(range(1, d + 1))
        k_tuples = []

        # Loop over all possible values of k up to j
        for k in range(1, j + 1):
            k_tuples.extend(combinations(elements, k))

        return k_tuples
    
    
    def generate_tree_function(self, tree):
        
        '''
        Given a tree generate its corresponding function using the heap structure
        of the tree
        
        Parameters:
        -----------------------------------------------------------------------
        tree : Heap structure of the tree
               List of dicts which denotes the different nodes in the tree
        
        '''
        
        def tree_function(x, tree):
            
            '''
            Parameters:
            -------------------------------------------------------------------
            x : np.array[d]
                The value at each we compute the tree function
            
            '''
            return x
        
        
    def visualize_tree(self, tree):
        '''                                                                     
        Given a tree function, print out the visualization of the tree.          
                                                                                        
        Parameters:
        -----------------------------------------------------------------------
        
        '''
        
                
        return 0

