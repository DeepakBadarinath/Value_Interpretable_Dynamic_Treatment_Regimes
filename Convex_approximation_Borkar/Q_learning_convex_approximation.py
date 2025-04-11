#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 23:34:31 2025

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm


#%%

# Given a function f, can you find f_c a convex function that approximates f?  
'''
Compute u as the solution to the following ODE, where \lambda_1[u](x) is the   
first Eigenvalue of D2[u](x)                                                   
-------------------------------------------------------------------------------
max(u(x) - g(x), \lambda_1[u](x)) = 0.                                         
                                                                                
We can compute u as a solution to a stochastic differential equation which is given by:
    dX_t = \sqrt(2) U(t) dW_t                                                  
    X_0 = x                                                                     
                                                                                
Let $\tau$ be the time at which it hits the boundary of \Omega.                 

Here we aim to minimize the following objective over U(.):                     
AA    J(x,U(.),\tau) := E[f(X_{\tau}) | X(0) = x]                                
                                                                                

To the above SDE, we construct a discrete time Markov chain equivalent          
with {X_n}_{n \geq 0} as described in the Approximation of Convex Envelope using
Reinforcement Learning.                                                         

'''                                                                            

#%%

class ConvexEnvelope:
    
    '''
    Class to find the convex envelope for a function u in D space.             
    
    '''                                                                        
    
    def __init__(self, delta, M, D, f):
                                                                                            
        '''
        Parameters:                                                            
        -----------------------------------------------------------------------
        delta : float
                Stepsize
        
        M : int                                                                
            The state space is [-M, M] \times [-M, M] \times ... [-M,M] D times
                                                                                    
        D : int                                                                
            Dimension of the space                                             
                                                                                
        f : f : R^D \to R                                                      
            The function we wish to approximate                                 
        
 action_space : list
                List of actions we can perform                                  
 
        '''
        
        self.delta = delta                                                             
        self.M = M                                                             
        self.D = D                                                             
        self.f = f                                                             
        self.action_space = ([tuple(np.eye(D)[i]) for i in range(self.D)] +    
                             [tuple(-np.eye(D)[i]) for i in range(self.D)])    
   
##########################################################################################################################################################                                                                        
                                                                                
    def compute_U_function(self, Q):
                                                                                
        '''                                                                     
        Given the Q function, compute the corresponding U function.            
        
        Parameters:
        -----------------------------------------------------------------------
        Q : Dataframe with columns given by columm names:                       
                                                                                
            X \times \mathcal{U} \times {0,1} \to \mathbb{R}                   
            
            The Q value function we wish to compute the U of                   
                                                                                
        Returns:
        -----------------------------------------------------------------------
        U : U : X \to \mathcal{U}                                              
            The optimal action function                                        
        
        '''
                                                                                
        def U(x):
                                                                                      
            min_val = np.inf                                                   
            min_action = None
            
            for u in self.action_space:                                         
                for z in [0,1]:                                                 
                    
                    if isinstance(x, tuple):
                        #print(f'{x} is a tuple')                              
                        q_val_list = Q.loc[(Q['location'] == x) & (Q['binary'] == z) &
                                           (Q['action'] == u),'value'].tolist() 
                                                                                
                    else:
                        q_val_list = Q.loc[(Q['location'] == (x,)) & (Q['binary'] == z) &
                                           (Q['action'] == u[0]),'value'].tolist() 
                                                                               
                    #print(q_val_list)                                          
                                                                                
                    if len(q_val_list) > 0:                                                                
                        q_val = q_val_list[0]                                      
                        #print(f'Q_value is given by {q_val}')                       
                                                                                
                        if (q_val < min_val):                                  
                            min_val = min(q_val, min_val)                      
                            min_action = u                                                
            
            #print(f'Min action is {min_action}, min_val is {min_val}')        
            if min_action is None:                                             
                print(f'Min action not found, error in location = {x}')        
                print('Q is')                                                  
                print(Q)                                                       
                
            return min_action
        
        return U
    
    ######################################################################################################################################################
        
    def scroll_through_grid(self, D, M, delta):
                                                                                    
        """
        Generator function to iterate through all points in a D-dimensional grid
        within the domain [-M, M]^D with step size delta in each dimension.    
                                                                                
        Parameters:
        -----------------------------------------------------------------------
            D : int                                                            
                Number of dimensions                                            
                                                                                                    
            M : float                                                          
                Half the side length of the domain                             
                
            delta :  float                                                     
                     Step size in each dimension.                              
                                                                                 
        Yields:                                                                
        -----------------------------------------------------------------------
            tuple: A D-dimensional point in the grid.
        """
        
        grid_ranges = [np.arange(-M, M + delta, delta) for _ in range(D)]      
                                                                               
        for point in np.ndindex(*[len(r) for r in grid_ranges]):               
            yield tuple(grid_ranges[d][i] for d, i in enumerate(point))        

    ######################################################################################################################################################    
    
    def grid_points(self, D, M, delta):
        """
        Generator to iterate through all points in a D-dimensional grid        
        within the domain [-M, M]^D with step size delta in each dimension.    
        
        Parameters:                                                            
-------------------------------------------------------------------------------
        D : int                                                                
            Number of dimensions                                                
        
        M : float
            Half-width of the domain in each dimension
            
        delta : float 
                Step size in each dimension                                        
                                                                                    
        Yields:
-------------------------------------------------------------------------------
        tuple: A point in the D-dimensional grid                               
        """                                                                    
        grid_ranges = [np.arange(-M, M + delta, delta) for _ in range(D)]      
        for point in itertools.product(*grid_ranges):                          
            yield point                                                        

##############################################################################################################################################################
        
    def compute_Q_values(self, Q_0, N, a,
                         column_names = ['location', 'action', 'binary', 'value'],
                         rounding_number = 3): 
        
        '''                                                                      
        Given the number N for which we can run this algorithm, compute the corr. 
        Q-values.
        
        Parameters:
        -----------------------------------------------------------------------
        Q_0 : X \times \mathcal{U} \times [0,1] \to \mathbb{R}                 
              The Q value function at the 0th iteration                        
                                                                                                                          
        N : int
            Number of times we run this algorithm for convergence              
        
        a : a : N \to [0,1]
            Fractional sampling constant
        
column_names : list
               The names of the columns for the Q dataframe
        
rounding_number : int
                  The number of places we round each val to while storing etc
                                                                                
        Returns:                                                               
        ----------------------------------------------------------------------- 
        Q_N : Q_N : X \times \mathcal{U} \times {0,1} \to \mathbb{R}           
              The Q value function at the Nth iteration                                             
                                                                                                  
        U_N : U_N : C \to \mathbb{U}                                           
              The optimal actions to be chosen at the end of N iterations
        '''
                                                                                    
        self.N = N
        self.Q_0 = Q_0                                                         
        self.a = a                                                             
        
        # Store Q as a dict instead of a function                              
                                                                                
        Q_prev = Q_0
                                                                                
        for n in range(1, self.N + 1):
            
            Q = pd.DataFrame(columns=['location', 'action','binary', 'value'])                       
            
            Q = Q.set_index(['location', 'action', 'binary'])                  
            
            for x_val in self.grid_points(self.D, self.M, self.delta):         
                for u in self.action_space:                                     
                    for z in [0,1]:                                             
                        
                                                                                
                        bernoulli_sample = np.random.binomial(1, 0.5)           
                                                                                
                        x_new = (bernoulli_sample * (x_val + self.delta * np.array(u)) + 
                                 (1 - bernoulli_sample) * (x_val - self.delta * np.array(u)))
                        
                        #Add functionality to check if x_new is in the grid, if not make have x_new := point closest to the current point in the grid
                        print(f'X_new is {x_new}') 
                        
                        for act_ind, act in enumerate(u):
                            if np.abs(act) > 0:
                                check_index = act_ind
                                break
                        
                        if x_new[check_index] > self.M:
                            x_new[check_index] = self.M
                        
                        elif x_new[check_index] < -self.M:
                            x_new[check_index] = -self.M
                        
                        min_Q_val = np.inf                                      
                        if type(x_new) == type(np.array([1,1])):
                            x_new = tuple(x_new)
                        
                        print(f'X_new preprocessed is {x_new}')
                        
                        for act in self.action_space:                                        
                            for binary_val in [0,1]:                                     
                                
                                if isinstance(x_new, tuple):
                                    
                                    # Compute Euclidean distances - make the below two steps faster
                                    distances = Q_prev['location'].apply(lambda loc: np.linalg.norm(np.array(loc) - np.array(x_new)))

                                    # Find the closest location
                                                                                
                                    closest_location = Q_prev.loc[distances.idxmin(), 'location']
                                    
                                                                        
                                    q_val_list = Q_prev.loc[(Q_prev['location'] == closest_location) & (Q_prev['binary'] == binary_val) &
                                                            (Q_prev['action'] == act),'value'].tolist() 
                                                                                
                                else:
                                    
                                    distances = Q_prev['location'].apply(lambda loc: np.linalg.norm(np.array(loc) - np.array([x_new])))

                                    # Find the closest location
                                        
                                    closest_location = Q_prev.loc[distances.idxmin(), 'location']
                                    
                                    q_val_list = Q_prev.loc[(Q_prev['location'] == closest_location) & (Q_prev['binary'] == binary_val) &
                                                            (Q_prev['action'] == act[0]),'value'].tolist()
                                
                                
                                print(f'Length of q_val_list is {len(q_val_list)}')
                                if (len(q_val_list) == 0):
                                    print('Q val list has zero length')         
                                    print(f'Q_prev does not have values corresponding to (x={x_new}, binary={binary_val}, action={act})')
                                    print('Q previous is')
                                    print(Q_prev)
                                
                                minimizing_val = q_val_list[0]
                                print(minimizing_val, min_Q_val)
                                min_Q_val = min(minimizing_val, min_Q_val)     
                        
                        if isinstance(x_val, tuple):
                            
                            #Compute Eucledian distances - make the below two steps faster - have everything be rounded up to a certain grid size?
                            distances = Q_prev['location'].apply(lambda loc: np.linalg.norm(np.array(loc) - np.array(x_val)))

                            # Find the closest location
                                
                            closest_location = Q_prev.loc[distances.idxmin(), 'location']
                            
                            
                            old_q_val_list = Q_prev.loc[(Q_prev['location'] == closest_location) & (Q_prev['binary'] == z) &
                                                        (Q_prev['action'] == u),'value'].tolist()
                            
                        else:
                            
                            #Compute Eucledian distances 
                            distances = Q_prev['location'].apply(lambda loc: np.linalg.norm(np.array(loc) - np.array([x_val])))

                            # Find the closest location
                                
                            closest_location = Q_prev.loc[distances.idxmin(), 'location']
                            
                            
                            old_q_val_list = Q_prev.loc[(Q_prev['location'] == (closest_location,)) & (Q_prev['binary'] == z) &
                                                        (Q_prev['action'] == u[0]),'value'].tolist()    
                        
                        
                        
                        old_q_val = old_q_val_list[0]
                        q_val =  old_q_val + (self.a(n) * (z * min_Q_val + (1-z)*self.f(x_val) - old_q_val))
                        
                        new_row = pd.DataFrame([[x_val, u, z, q_val]], columns=column_names)
                        print(f'We append the following row to Q : [{x_val}, {u}, {z}, {q_val}]')
                        Q = pd.concat([Q, new_row], ignore_index=True)          
                       
                        
        Q_prev = Q                                                              
        U = self.compute_U_function(Q)                                         
                                  
        return {'Q_final': Q,                                                  
                'U_final': U}
    
    ################################################################################################
    
    def find_minimizing_function(self, Q):
        
        '''                                                                      
        Compute x \to min_{u,z} Q(x,u,z)
        
        Parameters:
        -----------------------------------------------------------------------
        Q : X \times \mathcal{U} \times [0,1] \times \mathbb{R} \to \mathbb{R}                 
            The Q value function at the 0th iteration                        
                                                                                
        Returns:                                                               
        ----------------------------------------------------------------------- 
        f_min : X \to \mathbb{R}
                The minimizing function at each state space point.
        '''
        
        def f_min(x):                                                           
            
            min_val = np.inf                                                   
                                                                                
            #print(f'X is {x}, type of x is {type(x)}')
            #print(f'Action 1 is {self.action_space[0]} and type of action1 is {type(self.action_space[0])}')
            
            
            for u in self.action_space:                                         
                for z in [0,1]:                                                 
                    
                    
                    if isinstance(x, tuple):                                    
                        
                        #print(f'Type of x is {type(x)}, type of z is {type(z)}, type of u is {type(u)}')
                        all_tuples = Q['location'].apply(lambda x: isinstance(x, tuple)).all()
                        #print(f'All entries are tuples is {all_tuples}')  # True if all values are tuples, False otherwise

                        q_val = Q.loc[(Q['location'] == x) & (Q['binary'] == z) & (Q['action'] == u),'value']

                        #q_val_list = non_unique_list[0]
                                                                                
                    else:                                                       
                        q_val = Q.loc[(Q['location'] == (x,)) & (Q['binary'] == z) & (Q['action'] == u[0]),'value'][0]
                                                                                
                    
                    #print(f'Final Q_val is given by {q_val} and its type is {type(q_val)}')
                    if type(q_val) == list:
                        q_val = q_val[0]
                    elif type(q_val) == pd.Series:
                        q_val = q_val.iloc[0]
                    
                    if (q_val < min_val):                                  
                        min_val = min(q_val, min_val)
                    
                    else:
    
                        print(f'For x = {x}, u = {u}, and z = {z} we have that no Q value is found')
                        #print(Q[])
                        
            #print(f'Minimum value is {min_val}')
            return min_val
        
        return f_min    
    
#####################################################################################################    
