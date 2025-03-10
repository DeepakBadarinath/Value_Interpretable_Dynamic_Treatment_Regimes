#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:00:10 2024                                             

@author: badarinath                                                              

"""
import numpy as np
import random                                                                  
import matplotlib.pyplot as plt
import itertools                                                               
import importlib                                                               
from itertools import product, combinations                                    
import VIDTR_envs                                                              
from VIDTR_envs import GridEnv                                                 
                                                                                
#%%

import Juliens_paper as JP
import markov_decision_processes as mdp_module
import Decision_tree_set_functions
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc
import Trees                                                                   
from mpl_toolkits.mplot3d import Axes3D


#%%

importlib.reload(constraint_conditions)
importlib.reload(disjoint_box_union)
importlib.reload(JP)
importlib.reload(mdp_module)
importlib.reload(VIDTR_envs)
importlib.reload(Decision_tree_set_functions)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from Juliens_paper import DPTPA
from Decision_tree_set_functions import DecisionTreeSet
from Trees import DecisionTree

#%%

class DPTPA_grid:
    
    '''
    Build the algorithm environment for the VIDTR on a grid
    
    '''
    
    def __init__(self, dimensions, center, side_lengths, stepsizes, max_lengths,
                 max_complexity, max_depth, goal, time_horizon, gamma, eta, rho):
                                                                                
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
        
        
        self.env = GridEnv(dimensions, center, side_lengths, goal)              
        self.transitions = [self.env.transition for t in range(time_horizon)]   
        self.rewards = [self.env.reward for t in range(time_horizon)]           
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        
        def cost_function(s, a):                                                
                                                                                
            #print(f'State is {s}, action is {a}')                    
            
            #print(f'Function value is {np.exp((np.array(s)) - a)**2/2}')            
                                                                                                                                                                                                                
            return np.exp(np.sum((np.array(s) - a)**2/ 2 * (0.5)*len(s)))                                                    
        
        
        tree = DecisionTree()
                              
        print('We wish to examine the behaviour of generate_all_trees on the following:')        
        print(f'State bounds is : {np.array(self.env.state_space.bounds[0])}')
        print(f'Stepsizes is : {np.array(self.env.state_space.stepsizes[0])}')
        print(f'Max depth is: {max_depth}')
        print(f'Max complexity is: {max_complexity}')
        
        action_set = [np.array([-1,0]), np.array([1,0]),
                      np.array([0,1]), np.array([0,-1])]
                       
        all_trees = tree.generate_all_trees(max_depth,                           
                                            max_complexity,
                                            np.array(self.env.state_space.bounds[0]),         
                                            np.array(self.env.state_space.stepsizes[0]),
                                            action_set)                        
        
        self.decision_tree_sets = [DecisionTreeSet(dimension = dimensions,               
                                                                                                                   
                                                   states = self.env.state_space,
                                                                               
                                                   actions = [np.array([-1,0]),
                                                              np.array([1,0]), 
                                                              np.array([0,1]),                     
                                                              np.array([0,-1])],            
                                                   
                                                   cost_functions = [cost_function],
                                                                               
                                                   trees = all_trees) for t in range(time_horizon)]
                                                                                

        self.algo = DPTPA(time_horizon, gamma, self.decision_tree_sets,        
                          self.transitions)                                     
        
    
    def generate_random_trajectories(self, N):                                  
        '''
        Generate N trajectories from the VIDTR grid setup where we take a       
        random action at each timestep and we choose a random initial state    
        
        Returns:                                                               
        -----------------------------------------------------------------------
           obs_states : list[list]
                        N trajectories of the states observed
        
           obs_actions : list[list]
                         N trajectories of the actions observed
           
           obs_rewards : list[list]
                         N trajectories of rewards obtained                    
           
        '''
        
        obs_states = []                                                         
        obs_actions = []                                                                
        obs_rewards = []                                                        
                                                                                    
        for traj_no in range(N):
            obs_states.append([])                                                   
            obs_actions.append([])                                              
            obs_rewards.append([])                                                  
            s = np.squeeze(self.states[0].pick_random_point())                  
            obs_states[-1].append(s)                                             
            
            for t in range(self.time_horizon):                                  
                                                                                    
                a = random.sample(self.actions[t], 1)[0]                        
                r = self.rewards[t](s,a)                                                
                
                s = self.env.move(s,a)                                              
                obs_states[-1].append(s)                                        
                obs_actions[-1].append(a)                                       
                obs_rewards[-1].append(r)                                       
                
                                                                                                    
        return obs_states, obs_actions, obs_rewards
                                                                                
#%%
'''                                                                            
Tests GridEnv                                                                   
'''                                                                              
                                                                                                
if __name__ == '__main__':                                                      
                                                                                
    dimensions = 2                                                              
    center = np.array([0, 0])
    side_lengths = np.array([4, 4])                                             
    goal = np.array([-1, 0])                                                    
    time_horizon = 3                                                            
    gamma = 0.9                                                                                 
    max_lengths = [3 for t in range(time_horizon)]                              
    stepsizes = 0.1                                                                 
    max_complexity = 2                                                              
    max_depth = 2                                                               
    eta = -100/120                                                             
    rho = 10.2                                                                  
    
    grid_class = DPTPA_grid(dimensions, center, side_lengths,                     
                            stepsizes, max_lengths, max_complexity, max_depth,  
                            goal, time_horizon, gamma, eta, rho)                

    #%%                                                                          
    N = 3                                                                       
    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)         
                                                                                 
    #%%
    '''
    Tests VIDTR Bellman function                                               
    '''                                                                        
    # For t = 3 is the bellman map correct                                          
    s = np.array([2,2])                                                         
    max_val = -np.inf
    best_action = grid_class.actions[0]
                                                                               
    for a in grid_class.actions[0]:
        bellman_val = grid_class.rewards[2](s,a)                               
        if bellman_val > max_val:
            best_action = a                                                    
    
    best_action
    
    #%%                                                                        
    '''                                                                        
    Tests for compute_interpretable_policies                                   
    '''                                                                        
    return_dict = grid_class.algo.compute_optimal_trees()                                             
                                                                                
    print(return_dict['Optimal_values'])                                        
                                                                                
    for i,state_space in enumerate(grid_class.algo.decision_tree_sets):         
                                                                                
        state_iterator = disjoint_box_union.DBUIterator(state_space.states)     
        optimal_value_function = return_dict['Optimal_values'][i]               
        
        for s in state_iterator:                                                            
            print(f'{i}th value function evaluated on {s}: {optimal_value_function(s)}')    
                                                                                
#%%                                                                             
    for i,t in enumerate(return_dict['Optimal_trees']):                        
                                                                                
        t.save_tree_as_png(f'{i}th_tree')                                      
                                                                                    
    #%%                                                                        
    '''                                                                        
    Juliens paper - plot errors                                                        
    '''                                                                        
    grid_class.algo.plot_values()                                              
                                                                                                                                                              
    #%%
    '''
    Juliens paper - get interpretable policy
    '''
    for t in range(grid_class.time_horizon-1):
        
        val_tree_dict = grid_class.algo.compute_optimal_trees()
            
        print('Optimal values are')
        print(val_tree_dict['Optimal_values'])                                  

#%%%
    '''
    Juliens paper- plot the 2D grid
    '''
    for t in range(time_horizon):

        t_tree = val_tree_dict['Optimal_trees'][t]
        t_tree_function = DecisionTree.generate_tree_function(t_tree)
        
        grid_class.env.plot_policy_2D(t_tree_function,
                                      title = f'Actions at time {t}')
    

#%%%
    '''
    Juliens paper - plot values
    '''
    
    # Assuming dbu_iterator provides (x, y) pairs
    def plot_dbu_values_points(dbu_iterator, plotting_function,
                               x_label = 'X_axis', y_label = 'Y_axis', z_label='Z_axis',
                               plot_title = '3D_scatter_plot'):
        
        x_vals, y_vals, z_vals = [], [], []
    
        for s in dbu_iterator:
            x, y = s  # Extract x, y values from dbu_iterator
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(plotting_function(np.array([x,y])))  # Example function for z
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis')
    
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.title(plot_title)
        plt.show()
    
#%%%
    '''
    Juliens paper- plot surface
    '''
    
    # Assuming dbu_iterator provides (x, y) pairs
    def plot_dbu_values_surface(dbu_iterator, plotting_function,
                                x_label = 'X_axis', y_label = 'Y_axis', z_label='Z_axis',
                                plot_title = '3D_surface_plot'):
        
        x_vals, y_vals, z_vals = zip(*[(x, y, plotting_function(np.array([x,y])))
                                       for x, y in dbu_iterator])
    
        X = np.array(x_vals).reshape(int(np.sqrt(len(x_vals))), -1)
        Y = np.array(y_vals).reshape(int(np.sqrt(len(y_vals))), -1)
        Z = np.array(z_vals).reshape(int(np.sqrt(len(z_vals))), -1)
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='coolwarm')
                                                                                        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.title(plot_title)
        plt.show()

#%%%
                                                                                
    for t in range(grid_class.time_horizon-1):
        
        curr_dbu = grid_class.decision_tree_sets[t].states
        dbu_iter_class = disjoint_box_union.DBUIterator(curr_dbu)
        dbu_iterator = iter(dbu_iter_class)
        
        plotting_function = val_tree_dict['Optimal_values'][t]
        plot_dbu_values_surface(dbu_iterator, plotting_function,
                                plot_title = f'Optimal_values_at_time={t}')
        
        dbu_iter_class = disjoint_box_union.DBUIterator(curr_dbu)
        dbu_iterator = iter(dbu_iter_class)
        plot_dbu_values_surface(dbu_iterator, plotting_function,
                                plot_title = f'Optimal_surface_at_time={t}')        
        