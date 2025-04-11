#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 03:45:27 2025

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU
import random
import constraint_conditions as cc
import VIDTR_algo
from VIDTR_algo import VIDTR

import VIDTR_envs
from VIDTR_envs import GridEnv

from importlib import reload
import markov_decision_processes as mdp_module
from mpl_toolkits.mplot3d import Axes3D

                                                                               
#%%

mdp_module = reload(mdp_module)
disjoint_box_union = reload(disjoint_box_union)
cc = reload(cc)

from markov_decision_processes import MarkovDecisionProcess as MDP

#%%

# Test if Psi is convex, we wish to test:
# Psi(tx + (1-t)y) \leq t Psi(x) + (1-t) Psi(y) for different values of t, x, and, y

def Psi(condition, action, mdp, time,
        time_horizon, eta, rho, gamma,
        max_lengths, max_complexity, stepsizes):
    
    # Compute the value of Psi_t(R,a | r,P) for different values of condition, action,
    
    condition_DBU = DBU.condition_to_DBU(condition, stepsizes)                  
    error = 0   
    
    algo_VIDTR = VIDTR(mdp, max_lengths, eta, rho, max_complexity,
                      stepsizes)
    
    #print(f'The action we have is : {a}')
    bellman_function = lambda s: algo_VIDTR.maximum_over_actions(algo_VIDTR.bellman_function(time),time)(s) -VIDTR.fix_a(algo_VIDTR.bellman_function(time), a=action)(s)
    constant_function = lambda s: -algo_VIDTR.constant_eta_function()(s,action)
                                                                               
    bellman_error = condition_DBU.integrate(bellman_function)    
    #print(f'Bellman error is {bellman_error})
    const_error = condition_DBU.integrate(constant_function)     
    #print(f'Constant eta_error is {const_error)
    complexity_error = rho * condition.complexity 
    #print(f'Complexity error is {complexity_error}')      
        
    error += bellman_error + const_error + complexity_error
    
    return error
    
#%%

class VIDTR_grid:
                                                                                
    '''
    Build the algorithm environment for the VIDTR on a grid
    
    '''
    
    def __init__(self, dimensions, center, side_lengths, stepsizes, max_lengths,
                 max_complexity, goal, time_horizon, gamma, eta, rho):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        dimensions : int
                     Dimension of the grid 
        
        center : np.array
                 Center of the grid
                                                                                 
        side_lengths : np.array
                       Side lengths of the grid
                                                                                               
        stepsizes : np.array
                    Stepsizes for the grid
        
        max_lengths : np.array 
                      Maximum lengths for the grid
        
        max_complexity : int
                         Maximum complexity for the tree 
        
        goal : np.array
               Location of the goal for the 2D grid problem
               
        time_horizon : int
                       Time horizon for the VIDTR problem
                       
        gamma : float
                Discount factor
        
        eta : float
              Splitting promotion constant    
        
        rho : float
              Condition promotion constant                                      

        Stores:
        -----------------------------------------------------------------------
        envs : list[GridEnv]
               The 2D environments for the grid for the different timesteps
        
        VIDTR_MDP : markov_decision_processes
                    The Markov Decision Process represented in the algorithm
        
        algo : VIDTR_algo
               The algorithm representing VIDTR
        '''
        self.dimensions = dimensions
        self.center = center
        self.side_lengths = side_lengths
        self.stepsizes = stepsizes
        self.max_lengths = max_lengths
        self.max_complexity = max_complexity
        self.goal = goal
        self.time_horizon = time_horizon                                       
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
                                                                                            
        self.env = GridEnv(dimensions, center, side_lengths, goal, stepsizes)              
        self.transitions = [self.env.transition for t in range(time_horizon)]   
        self.rewards = [self.env.reward for t in range(time_horizon)]           
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        self.VIDTR_MDP = MDP(dimensions, self.states, self.actions, time_horizon, gamma,
                             self.transitions, self.rewards)                    
        
        self.algo = VIDTR(self.VIDTR_MDP, max_lengths, eta, rho, max_complexity,
                          stepsizes)


#%%

# Test demo for /Psi function

dimension = 2
center = np.array([0,0])
side_lengths = np.array([4,4])
time_horizon = 5
stepsizes = np.array([0.5,0.5])
max_lengths = np.array([5,5])
gamma = 0.9
max_lengths = 3
max_complexity = 3
goal = np.array([1,1])
eta = 0.9
rho = 0.5
                                                                                
vidtr_grid = VIDTR_grid(dimension, center, side_lengths, stepsizes,            
                        max_lengths, max_complexity, goal, time_horizon,       
                        gamma, eta, rho)                                       

state_spaces = vidtr_grid.states
action_spaces = vidtr_grid.actions

mdp = MDP(dimension, state_spaces, action_spaces, time_horizon, gamma,         
          vidtr_grid.transitions, vidtr_grid.rewards)                          


constraint_bounds = np.array([[0,1]])
state_bounds = np.array([[-2,2], [-2,2]])


constraint = cc.ConstraintConditions(dimension = 2,                              
                                     non_zero_indices = np.array([1]),         
                                     bounds = constraint_bounds,               
                                     state_bounds = state_bounds)              

action = np.array([1.0])

time = 1.0

print(Psi(constraint, action, mdp, time, time_horizon, eta, rho,
          gamma, max_lengths, max_complexity, stepsizes))

#%%
# Plot \Psi

dimension = 2
center = np.array([0,0])
side_lengths = np.array([4,4])
time_horizon = 5
stepsizes = np.array([0.5,0.5])
max_lengths = np.array([5,5])
gamma = 0.9
max_lengths = 3
max_complexity = 3
goal = np.array([1,1])
eta = 3.2
rho = 1.7
                                                                                
vidtr_grid = VIDTR_grid(dimension, center, side_lengths, stepsizes,            
                        max_lengths, max_complexity, goal, time_horizon,       
                        gamma, eta, rho)                                       

state_spaces = vidtr_grid.states
action_spaces = vidtr_grid.actions

mdp = MDP(dimension, state_spaces, action_spaces, time_horizon, gamma,         
          vidtr_grid.transitions, vidtr_grid.rewards)                          


constraint_bounds = np.array([[-1.5,1.5]])
state_bounds = np.array([[-2,2], [-2,2]])


constraint = cc.ConstraintConditions(dimension = 2,                              
                                     non_zero_indices = np.array([1]),         
                                     bounds = constraint_bounds,               
                                     state_bounds = state_bounds)              

action = np.array([-1.0])

time = 2.0

print(Psi(constraint, action, mdp, time, time_horizon, eta, rho,
          gamma, max_lengths, max_complexity, stepsizes))


#%%

def f(a,b):
    
    print(f'A is {a}, B is {b}')
    constraint = cc.ConstraintConditions(dimension = 2,
                                         non_zero_indices = np.array([0]),
                                         bounds = np.array([[a,b]]),
                                         state_bounds = np.array([[-2,2],
                                                                  [-2,2]]))
    print('Constraint is')
    print(constraint)
    
    return Psi(constraint, action, mdp, time, time_horizon, eta, rho,
              gamma, max_lengths, max_complexity, stepsizes)
    
# Define the ranges and stepsizes
a_values = np.arange(-2, 2.5, 0.25)  # Stepsize 0.5                             
b_values = np.arange(-2, 2.5, 0.25)  # Stepsize 0.5                             

# Initialize arrays for storing computed values
A, B = np.meshgrid(a_values, b_values)
Z = np.zeros_like(A)                                                           
                                                                                
# Compute function values point-by-point
for i in range(A.shape[0]):                                                    
    for j in range(A.shape[1]):                                                
        Z[i, j] = f(A[i, j], B[i, j])

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))                                                
ax = fig.add_subplot(111, projection='3d')                                          

# Plot the surface
surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# Add labels and title
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('f(a, b)')
ax.set_title('Surface Plot of f(a, b)')

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.show()

#%%

for a in range(-2.0, 2.0, 0.5):
    print(a)
    for b in range(-2.0, 2.0, 0.5):

        constraint = cc.ConstraintConditions(dimension = 2,                     
                                             non_zero_indices = np.array([0,1]),
                                             bounds = np.array([a,b]),          
                                             state_bounds = np.array([[-2,2], [-2,2]]))
        
        

#%%

# Test convexity for /Psi function

def check_inequality(Psi, t, constraint_bounds_1, constraint_bounds_2,
                     non_zero_indices_1, non_zero_indices_2,
                     action, mdp, time, time_horizon, eta, rho, gamma, max_lengths,
                     stepsizes, state_bounds):
    
    #print(constraint_bounds_1)
    
    constraint_1 = cc.ConstraintConditions(dimension = 2,                              
                                           non_zero_indices = non_zero_indices_1,         
                                           bounds = constraint_bounds_1,               
                                           state_bounds = state_bounds)
    
    #print(constraint_bounds_2)
    
    constraint_2 = cc.ConstraintConditions(dimension = 2,                              
                                           non_zero_indices = non_zero_indices_2,         
                                           bounds = constraint_bounds_2,               
                                           state_bounds = state_bounds)
    
    
    convex_constraint_bounds = t*constraint_bounds_1 + (1-t)*constraint_bounds_2
    non_zero_indices_new = []
    
    for i,index in enumerate(non_zero_indices_1):
        if index not in non_zero_indices_new:
            non_zero_indices_new.append(index)
        
    for i, index in enumerate(non_zero_indices_2):
        if index not in non_zero_indices_new:
            non_zero_indices_new.append(index)
    
    convex_constraint = cc.ConstraintConditions(dimension = 2,
                                                non_zero_indices = non_zero_indices_new,
                                                bounds = convex_constraint_bounds,
                                                state_bounds = state_bounds)
    
    convex_Psi = Psi(convex_constraint, action, mdp, time, time_horizon,
                     eta, rho, gamma, max_lengths, max_complexity, stepsizes)
    
    rhs = t * Psi(constraint_1, action, mdp, time, time_horizon,
                  eta, rho, gamma, max_lengths, max_complexity, stepsizes)
    
    rhs += (1-t) * Psi(constraint_2, action, mdp, time, time_horizon,
                       eta, rho, gamma, max_lengths, max_complexity, stepsizes)
    
    if convex_Psi <= rhs:                                                       
        return True                                                                 
                                                                                
    else:
        return False
    
#%%

# Plot 1D errors in \Psi







#%%
t = 0.25

constraint_bounds_1 = np.array([[-1,2.0], [-3,3]])
non_zero_indices_1 = np.array([0,1])

constraint_bounds_2 = np.array([[0.25, 0.75], [-1,-0.3]])
non_zero_indices_2 = np.array([0,1])

state_bounds = (vidtr_grid.states[0].bounds)[0]
print(np.array(state_bounds))

for d in range(dimension):
    print(f'Dimension is {d}')
    print(f'State bounds at {(d,0)} is {state_bounds[d][0]} and at {(d,1)} it is {state_bounds[d][1]}')

check_inequality(Psi, t, constraint_bounds_1, constraint_bounds_2,
                 non_zero_indices_1, non_zero_indices_2, action, mdp, time,
                 time_horizon, eta, rho, gamma, max_lengths, stepsizes,
                 np.array(state_bounds))

#%%
t = 0.75

constraint_bounds_1 = np.array([[-1,2.0], [-3,3]])
non_zero_indices_1 = np.array([0,1])

constraint_bounds_2 = np.array([[0.25, 0.75], [-1,-0.3]])
non_zero_indices_2 = np.array([0,1])

state_bounds = (vidtr_grid.states[0].bounds)[0]
print(np.array(state_bounds))

for d in range(dimension):
    print(f'Dimension is {d}')                                                 
    print(f'State bounds at {(d,0)} is {state_bounds[d][0]} and at {(d,1)} it is {state_bounds[d][1]}')

check_inequality(Psi, t, constraint_bounds_1, constraint_bounds_2,
                 non_zero_indices_1, non_zero_indices_2, action, mdp, time,
                 time_horizon, eta, rho, gamma, max_lengths, stepsizes,
                 np.array(state_bounds))
                                                                                
#%%                                                                            
                                                                               
t = 0.8
constraint_bounds_1 = np.array([[-1.2, 1.5], [-4,3]])
non_zero_indices_1 = np.array([0,1])

constraint_bounds_2 = np.array([[0.25, 0.75], [-1,-0.3]])
non_zero_indices_2 = np.array([0,1])

state_bounds = (vidtr_grid.states[0].bounds)[0]
print(np.array(state_bounds))

for d in range(dimension):
    print(f'Dimension is {d}')
    print(f'State bounds at {(d,0)} is {state_bounds[d][0]} and at {(d,1)} it is {state_bounds[d][1]}')

check_inequality(Psi, t, constraint_bounds_1, constraint_bounds_2,
                 non_zero_indices_1, non_zero_indices_2, action, mdp, time,
                 time_horizon, eta, rho, gamma, max_lengths, stepsizes,
                 np.array(state_bounds))


#%%
