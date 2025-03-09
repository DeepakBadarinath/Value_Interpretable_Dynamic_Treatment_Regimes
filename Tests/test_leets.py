#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 01:51:39 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

list_of_lists = [2,2,2,3]

go_on = False

for i, l in enumerate(list_of_lists):
    
    print(f'i is {i}')
    print(f'Element is {l}')
    
    if l == 2:
        
        list_of_lists.pop(i)                                                    
        i = i -1                                                                            
        
    else:
        go_on = True

print(list_of_lists)

#%%