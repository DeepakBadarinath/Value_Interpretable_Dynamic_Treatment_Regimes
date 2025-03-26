#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 01:01:59 2025                                            

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt                                                
import importlib
import Q_learning_convex_approximation_parallel
import pandas as pd                                                            
import warnings                                                                
warnings.simplefilter(action='ignore', category=FutureWarning)                 
                                                                               
#%%

from Q_learning_convex_approximation_parallel import ConvexEnvelope                     
importlib.reload(Q_learning_convex_approximation_parallel)                              


#%%

delta = 0.5                                                                    
M = 5                                                                                   
D = 1                                                                           
f = lambda x : (np.sum(x))**2 * np.sin(np.sum(np.abs(x)))
N = 1
#domain = np.linspace(0,1,num=N)                                                 

# Domain is [0,1]

convex_envelope = ConvexEnvelope(delta, M, D, f)                               
domain = []                                                                         
for x in convex_envelope.grid_points(D, M, delta):                             
    domain.append(x)                                                          

#print('Domain is')
#print(domain)

#%%

L = 2                                                             
                                                                                    
a = lambda n : 1 / int(1 + n / 1000)                                                            
column_names = ['location', 'action', 'binary', 'value']                            
Q_0 = pd.DataFrame(columns=column_names)                                                    
                                                                                    
count = 0                                                                      
                                                                                    
for x_val in convex_envelope.grid_points(D, M, delta):                          
    for u in convex_envelope.action_space:                                      
        for z in [0,1]:                                                        
            
            valz = (np.array(x_val) - np.array(u)*z)**2
                                                                                
            Q_0 = Q_0.append({"location": x_val, "action": u,                  
                              "binary": z, "value": valz}, ignore_index=True)   
                                                                                
#%%                                                                             
                                                                                                                                                                    
y_vals = [f(x) for x in domain]                                                 
y_approx = []                                                                   
                                                                                
Q_dict = convex_envelope.compute_Q_values(Q_0, N, a)                             
#print(Q_dict)                                                                 


#%%

print('Y vals are')                                                             
print(y_vals)                                                                  
                                                                                
for x in domain:                                                                    
    print(f'Domain point is {x}')                                              
                                                                               
for p in convex_envelope.grid_points(D, M, delta):                                           
    print(f'P is {p}')                                                         

Q_new = Q_dict['Q_final']                                                        
                                                                                                                                         
f_min = convex_envelope.find_minimizing_function(Q_new)

for x in domain:
    y_approx.append(f_min(x))

                                                                                                    
plt.plot(domain, y_vals, label='Original_function')                                                 
plt.plot(domain, y_approx, label = 'Approximated function')
                                                                   

plt.legend()                                                                    
plt.show()                                                                          
                                                                                
#%%

print(y_vals)
print(y_approx)
