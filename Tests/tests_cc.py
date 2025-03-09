#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:41:58 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU
import constraint_conditions as cc
import itertools
from importlib import reload

cc = reload(cc)
disjoint_box_union = reload(disjoint_box_union)

#%%
constraint_bounds = np.array([[0,1]])
state_bounds = np.array([[-1,1], [-1,1]])


constraint = cc.ConstraintConditions(dimension = 2,                              
                                     non_zero_indices = np.array([0]),         
                                     bounds = constraint_bounds,               
                                     state_bounds = state_bounds)              

print('The first constraint is given by')
print(constraint)
#%%

dbu = DBU.condition_to_DBU(constraint,
                           0.5)
print('DBU is')
print(dbu)

constraint.plot_2D_constraints()

points = [np.array([0,0]), np.array([0,1]),
          np.array([4,5]), np.array([-5,-10])]

for i, point in enumerate(points):
    print(f'Point {point} is contained in?')
    print(constraint.contains_point(point))