#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 06:46:37 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import Decision_tree_set_functions                                             
from Decision_tree_set_functions import DecisionTreeSet as DTS                 
import importlib
import disjoint_box_union
import Trees
from Trees import DecisionTree

importlib.reload(Decision_tree_set_functions)
importlib.reload(Trees)

#%%

'''                                                                             
We are getting None as the answer on the first tree when we generate the tree                   
function. Why is this? Is there an error with the generate_tree_function method?                      
                                                                                                
                                                                            
                                                                                
state_bounds = np.array([[-2,2], [-2,2]])                                       
stepsizes = np.array([1.0, 1.0])                                                     
max_depth = 2                                                                  
max_complexity = 2                                                              

tree = DecisionTree()

all_trees = tree.generate_all_trees(state_bounds,                               
                                    stepsizes,                                   
                                    max_depth,                                      
                                    max_complexity,
                                    action_set = [np.array([0,1]),
                                                  np.array([-1,0])])                                 

first_tree = all_trees[0].root
first_right = first_tree.right
print(first_right)

#second_right = first_right.right
#print(second_right)

'''

