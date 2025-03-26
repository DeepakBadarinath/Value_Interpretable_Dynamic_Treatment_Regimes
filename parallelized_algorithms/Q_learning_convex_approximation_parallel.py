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


from concurrent.futures import ProcessPoolExecutor, as_completed

#%%

def process_grid_point(args):
    x_val, Q_prev, n, a_func, self_obj, column_names = args
    rows = []

    for u in self_obj.action_space:
        for z in [0, 1]:
            bernoulli_sample = np.random.binomial(1, 0.5)
            x_new = (bernoulli_sample * (x_val + self_obj.delta * np.array(u)) + 
                     (1 - bernoulli_sample) * (x_val - self_obj.delta * np.array(u)))

            check_index = next((i for i, act in enumerate(u) if abs(act) > 0), 0)
            x_new[check_index] = np.clip(x_new[check_index], -self_obj.M, self_obj.M)

            min_Q_val = np.inf
            x_new = tuple(x_new)

            for act in self_obj.action_space:
                for binary_val in [0, 1]:
                    distances = Q_prev['location'].apply(lambda loc: np.linalg.norm(np.array(loc) - np.array(x_new)))
                    closest_location = Q_prev.loc[distances.idxmin(), 'location']
                    q_val_list = Q_prev.loc[(Q_prev['location'] == closest_location) &
                                            (Q_prev['binary'] == binary_val) &
                                            (Q_prev['action'] == act), 'value'].tolist()
                    if q_val_list:
                        min_Q_val = min(min(q_val_list), min_Q_val)

            distances = Q_prev['location'].apply(lambda loc: np.linalg.norm(np.array(loc) - np.array(x_val)))
            closest_location = Q_prev.loc[distances.idxmin(), 'location']
            old_q_val_list = Q_prev.loc[(Q_prev['location'] == closest_location) &
                                        (Q_prev['binary'] == z) &
                                        (Q_prev['action'] == u), 'value'].tolist()
            old_q_val = old_q_val_list[0] if old_q_val_list else 0.0

            q_val = old_q_val + (a_func(n) * (z * min_Q_val + (1 - z) * self_obj.f(x_val) - old_q_val))
            rows.append([x_val, u, z, q_val])

    return rows

def grid_worker(x_val, Q_data, n, a_n, delta, M, action_space,
                f_x_val, column_names, rounding_number):
    """
    Worker function to compute Q rows for a single x_val.
    """
    Q_rows = []
    
    for u in action_space:
        for z in [0, 1]:
            bernoulli_sample = np.random.binomial(1, 0.5)
            x_new = (bernoulli_sample * (np.array(x_val) + delta * np.array(u)) +
                     (1 - bernoulli_sample) * (np.array(x_val) - delta * np.array(u)))

            # Clip x_new to within bounds
            x_new = np.clip(x_new, -M, M)

            # Find closest location in Q_data
            min_dist = float('inf')
            min_Q_val = np.inf

            for loc, act, binary, val in Q_data:
                dist = np.linalg.norm(np.array(loc) - x_new)
                if dist < min_dist:
                    min_dist = dist
                    closest_q = (loc, act, binary, val)

            # Compute min_Q_val across actions and binary
            for loc, act, binary, val in Q_data:
                if np.linalg.norm(np.array(loc) - x_new) == min_dist:
                    min_Q_val = min(min_Q_val, val)

            # Get old Q value for (x_val, u, z)
            min_dist_old = float('inf')
            old_q_val = 0
            for loc, act, binary, val in Q_data:
                dist = np.linalg.norm(np.array(loc) - np.array(x_val))
                if dist < min_dist_old and (np.allclose(act, u) and binary == z):
                    min_dist_old = dist
                    old_q_val = val

            # Update Q value
            q_val = old_q_val + a_n * (z * min_Q_val + (1 - z) * f_x_val - old_q_val)
            q_val = np.round(q_val, rounding_number)

            Q_rows.append([tuple(x_val), tuple(u), z, q_val])
    
    return Q_rows


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
                                                                                
    def compute_U_function(self, Q_df):
        
        """
        Compute the optimal policy U(x) from the final Q-values.
    
        Parameters:
        -----------------------------------------------------------------------
        Q_df : pd.DataFrame
               DataFrame with columns ['location', 'action', 'binary', 'value']
               representing Q(x, u, z)
    
        Returns:
        -----------------------------------------------------------------------
        U_final : dict
                  Mapping from state (location) to optimal action
        """
    
        # Initialize policy dictionary
        U_final = {}
    
        # Group Q values by 'location' (state) and 'binary'
        grouped = Q_df.groupby(['location', 'binary'])
    
        for (location, binary), group in grouped:
            # For each (location, binary), find action u that minimizes Q
            min_idx = group['value'].idxmin()
            optimal_action = Q_df.loc[min_idx, 'action']
    
            # You can store (location, binary) -> action or location -> action depending on use-case
            U_final[(location, binary)] = optimal_action
    
        return U_final

    
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
                         column_names=['location', 'action', 'binary', 'value'],
                         rounding_number=3):
        '''
        Compute Q values over N iterations using parallelized grid loop and tqdm.
        '''
        self.N = N
        self.Q_0 = Q_0
        self.a = a
    
        Q_prev = Q_0  # Initial Q
        for n in range(1, self.N + 1):
            Q_rows = []
            Q = pd.DataFrame(columns=column_names)
        
            Q_data = list(zip(Q_prev['location'], Q_prev['action'], Q_prev['binary'], Q_prev['value']))
            grid_points = list(self.grid_points(self.D, self.M, self.delta))  
            a_n = self.a(n)
        
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(grid_worker, x_val, Q_data, n, a_n, self.delta, self.M, self.action_space,
                                    self.f(x_val), column_names, rounding_number)
                    for x_val in grid_points
                ]
        
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Iteration {n}/{self.N}"):
                    Q_rows.extend(future.result())
        
            # Combine all rows into DataFrame
            Q = pd.DataFrame(Q_rows, columns=column_names)
            Q_prev = Q  # Prepare for next iteration
        
        # Compute final policy from last Q
        U_final = self.compute_U_function(Q_prev)
        
        return {'Q_final': Q_prev,
                'U_final': U_final}


#######################################################################################################
    
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
