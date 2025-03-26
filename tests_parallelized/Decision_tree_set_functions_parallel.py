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
import Trees_parallelized as Trees
import importlib
import itertools
import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU                         


importlib.reload(Trees)
importlib.reload(disjoint_box_union)


#%%                                                                            

class DecisionTreeSet:                                                          
                                                                                    
    def __init__(self, dimension, states, actions, cost_functions, trees = []):
        
        '''
        The set of decision trees are characterized by possible states, actions,
        and cost functions.                                                    
        
        States here are a subset of \mathbb{R}^d. Actions we assume can be floats
        or a subset of \mathbb{R}^d.                                            
                                                                                
        Cost functions are functions from the state space, actions to the real line.
                                                                                
        Parameters:
        -----------------------------------------------------------------------
        dimension : int
                    The dimension of the state space
        
        states : DisjointBoxUnion
                 The states which is of the type DisjointBoxUnion;
                 comes equipped with state bounds and differences
        
        actions : list
                  List of possible actions that can be taken
        
  cost_functions : list
                   The list of functions from S \times A to \mathbb{R} which   
                   indicates the different cost functions
        
        trees : list
                List of tree functions
        '''
        
        self.dimension = dimension
        self.states = states
        self.actions = actions
        self.cost_functions = cost_functions
        self.trees = trees
    
    
    def generate_trees(self, max_depth, max_complexity, state_bounds,
                       state_differences):
        
        '''
        Generate all possible trees with maximum depth max_depth where the maximum
        and conditions employed at each node condition is max_complexity.

        Parameters:
        -----------------------------------------------------------------------
        max_depth : int
                    The maximum depth for the tree 
            
        max_complexity : int
                         The maximum complexity per splitting condition of the
                         node
        
        state_bounds : np.array[d,2]                                           
                       The state bounds for the dimensions                     
        
    state_differences : np.array[d]
                        The differences we iterate through for each dimension
        
        Returns:
        -----------------------------------------------------------------------
        trees : list
                All possible trees with maximum complexity max_complexity,
                depth max_depth generated over state_bounds with spacing as in 
                state_differences

        '''
        trees = []
        self._generate_tree_recursive(trees, depth=1, max_depth=max_depth,
                                      max_complexity=max_complexity, 
                                      state_bounds=state_bounds,
                                      state_differences=state_differences,
                                      parent=None)
        return trees
    
    
    def _generate_tree_recursive(self, trees, depth, max_depth, max_complexity, 
                                 state_bounds, state_differences, parent):
        """                                                                       
        Recursively generate the trees up to max_depth and store complete trees.
        """
        if depth > max_depth:
            return  

        if depth == max_depth:
            # Create a leaf node with an action
            action_node = Trees.TreeNode(action=f"Action at depth {depth}")
            if parent:
                if parent.left is None:
                    parent.attach_left(action_node)
                elif parent.right is None:
                    parent.attach_right(action_node)
            trees.append(self.clone_tree())  # Store the tree
            return
    
        # Generate conditions for possible splits
        for complexity in range(1, max_complexity + 1):
            dimension_combinations = itertools.combinations(range(state_bounds.shape[0]), complexity)
            
            for dimensions in dimension_combinations:
                for dim in dimensions:
                    lower_bound, upper_bound = state_bounds[dim]
                    values = np.arange(lower_bound, upper_bound + state_differences[dim], state_differences[dim])
                    
                    for val in values:
                        for greater_than in [True, False]:  
                            # Create a condition-based node
                            new_node = Trees.TreeNode(dimensions=[dim], values=[val], greater_than=[greater_than])
                            
                            if parent is None:
                                self.root = new_node  # Set root node
                            else:
                                if parent.left is None:
                                    parent.attach_left(new_node)
                                elif parent.right is None:
                                    parent.attach_right(new_node)
                                else:
                                    continue  # Skip if both child slots are filled
    
                            # Recur for left and right children
                            self._generate_tree_recursive(trees, depth + 1, max_depth, max_complexity, 
                                                          state_bounds, state_differences, new_node)
    
                            # Backtrack: Remove the added node before trying a new one
                            if parent:
                                if parent.left == new_node:
                                    parent.left = None
                                elif parent.right == new_node:
                                    parent.right = None                        
                            else:
                                self.root = None

                                                                                      
    def clone_tree(self):
        """Deep copy the tree to save its current structure."""
        return Trees.DecisionTree._clone_node(self.root)
    
    @staticmethod
    def _clone_node(node):
        """Recursively clone nodes for deep copy of tree."""
        if node is None:
            return None
        new_node = Trees.TreeNode(dimensions=node.dimensions[:],
                                  values=node.values[:],
                                  greater_than=node.greater_than[:],
                                  action=node.action)
        
        new_node.left = Trees.DecisionTree._clone_node(node.left)
        new_node.right = Trees.DecisionTree._clone_node(node.right)
        return new_node
    
#%%