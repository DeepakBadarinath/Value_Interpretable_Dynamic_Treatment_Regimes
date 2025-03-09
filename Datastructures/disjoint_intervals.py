#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:58:42 2024
    
@author: badarinath
"""
 
import numpy as np
import matplotlib.pyplot as plt

# Disjoint sets are a list of tuples which denote the endpoints of the different sets



class DisjointSets:
    
    def __init__(self, end_points):
        
        # end_points are a list of tuples which denote the start and end points of
        # the intervals 
        self.end_points = end_points
    
    
    def evaluate_integral(self, function):
        
        '''
        Given a 
        '''
        
        integral  = 0
        for start, end in range(len(self.end_points)):
            for j in range(start, end+1):
                integral += function(j)
                    
        return integral
    
  