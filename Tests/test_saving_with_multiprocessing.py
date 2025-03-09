#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:58:00 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import disjoint_box_union                                                      
import constraint_conditions as cc                                             
import icecream
import random                                                                   
import pickle                                                                  
import multiprocessing as mp                                                   
import os                                                                      
import subprocess                                                              
import pandas as pd                                                             
import time

from icecream import ic

from importlib import reload                                                   
from itertools import combinations, product
                                    
#%%                                                                            
                                                                                    
def add_numbers(number_list, row_counter = [0]):
    
    total = 0
    for i, number in enumerate(number_list):
    	total += number
    
    keys = [i for i in range(len(number_list))]
    print(keys)
    values = [[number] for number in number_list]
    print(values)
    
    key_value_pairs = zip(keys, values)                                        
    
    row_dict = dict(key_value_pairs)
    																		   				
    print(row_dict)
    row_df = pd.DataFrame(row_dict)
    print(row_df)
    
    row_df.to_csv(f'Brow-{row_counter[-1]}.csv')
    
    row_counter[-1] = row_counter[-1] + 1                                                                 
    
    return {'List': number_list, 'Total': total}


#%%

row_df = pd.DataFrame()

def logger_row(row_dict):                                                    
    row_df = pd.DataFrame(row_dict)             


#%%

pool = mp.Pool()
                                                                                    
for start in range(5):                                                         
    for nos in range(10):                                                       
        
        number_list = [start + i for i in range(nos)]
        pool.apply_async(add_numbers, args = (number_list, ),
                         callback = logger_row)
        time.sleep(0.1)
        
pool.close()
pool.join()


#%%
a = []
def update_array(x):
    a.append(x)

update_array(2.0)
update_array(1.0)
update_array(-1.0)
update_array(-1.0)
update_array(2.5)
update_array(-10.0)
update_array(-15.0)
update_array(20.1)
print(f'Array is {a}')