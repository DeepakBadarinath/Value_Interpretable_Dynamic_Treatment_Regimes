#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:27:22 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import disjoint_box_union 
import constraint_conditions as cc
import icecream
import random
import pickle

from icecream import ic

from importlib import reload
from itertools import combinations, product


disjoint_box_union = reload(disjoint_box_union)
from disjoint_box_union import DisjointBoxUnion as DBU                         
                                                                                
import IVIDTR_algo_dict_to_func as VIDTR_module                                                                    
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc
from VIDTR_envs import GridEnv

#%%

reload(constraint_conditions)
reload(disjoint_box_union)
reload(VIDTR_module)
reload(mdp_module)


from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_algo_dict_to_func import VIDTR


cc = reload(cc)

ic.enable()

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
        
        
        self.env = GridEnv(dimensions, center, side_lengths, goal)
        self.transitions = [self.env.transition for t in range(time_horizon)]
        self.rewards = [self.env.reward for t in range(time_horizon)]
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        self.VIDTR_MDP = MDP(dimensions, self.states, self.actions, time_horizon, gamma,
                             self.transitions, self.rewards)                    
        
        self.algo = VIDTR(self.VIDTR_MDP, max_lengths, eta, rho, max_complexity,
                          stepsizes)
        
    
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
            s = np.squeeze(self.VIDTR_MDP.state_spaces[0].pick_random_point())  
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
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])
    time_horizon = 4
    gamma = 0.9
    max_lengths = [5 for t in range(time_horizon)]
    stepsizes = 0.1
    max_complexity = 2

    
    #%%
    '''
    VIDTR - tests for different \rho and \eta values
    '''
    etas = [-2, -1, -0.5, -0.2, -0.1, 0.1, 0.5]
    
    rhos = [-5.0, -1.0, -0.1, 0.2, 1.0, 5.0, 10.0]
    
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])
    time_horizon = 2                                                                                        
    gamma = 0.9                                                                                          
    max_lengths = [10 for t in range(time_horizon)]                                                          
    stepsizes = 0.1                                                                                         
    max_complexity = 4
    bellman_errors_rho_eta_dict = {}
    total_errors_rho_eta_dict = {}
    
    for i, eta in enumerate(etas):
        for j, rho in enumerate(rhos):                                                                                          
                                                                        
            grid_class = VIDTR_grid(dimensions, center, side_lengths,
                                       stepsizes, max_lengths, max_complexity, goal,
                                       time_horizon, gamma, eta, rho)
            
            int_policies = grid_class.algo.compute_interpretable_policies()
            
            for t in range(grid_class.time_horizon-1):
                
                int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                            grid_class.algo.optimal_actions[t])
                    
                grid_class.env.plot_policy_2D(int_policy,
                                              title=f'Time={t}, eta = {eta}, rho = {rho}',
                                              saved_fig_name = f'Intepretable_policy_for_eta={eta}_rho = {rho}.png')
                
                bellman_errors_rho_eta_dict[(t, eta, rho)] = grid_class.algo.total_bellman_error
                total_errors_rho_eta_dict[(t, eta, rho)] = grid_class.algo.total_error
                
                with open('bellman_errors_rho_eta_dict.pkl', 'wb') as file:
                    pickle.dump(bellman_errors_rho_eta_dict, file)
                
                with open('total_errors_rho_eta_dict.pkl', 'wb') as file:
                    pickle.dump(total_errors_rho_eta_dict, file)
                
                
    #%%
    '''
    Code to test how the errors scale with the length of the grid
    '''
    errors_per_length = []
    bellman_errors_per_length = []
    
    vars_per_length = []
    bellman_vars_per_length = []
    
    error_dict_length = {}
    bellman_error_dict_length = {}
    
    vars_dict_length = {}
    bellman_vars_dict_length = {}
    
    possible_lengths = [3, 4, 5, 6]
    no_of_trials = 10
    
    eta = -2
    rho = -5
    
    for i, l in enumerate(possible_lengths):
    
        avg_error = 0
        avg_bellman_error = 0
        
        avg_error_variance = 0
        bellman_error_variance = 0
    	
        for trial_no in range(no_of_trials): 
            side_lengths = np.array([l,l])
            
            grid_class = VIDTR_grid(dimensions, center, side_lengths,
                                    stepsizes, max_lengths, max_complexity, goal,
                                    time_horizon, gamma, eta, rho)
            
            int_policies = grid_class.algo.compute_interpretable_policies()
            
            for t in range(grid_class.time_horizon-1):
                
                int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                        grid_class.algo.optimal_actions[t])
        
        
            new_avg_error = (avg_error * trial_no + grid_class.algo.total_error) / (trial_no + 1)
            new_avg_bellman_error = (avg_bellman_error * trial_no + grid_class.algo.total_bellman_error) /(trial_no + 1)        
        
            avg_error_variance = avg_error_variance ** 2 + ((grid_class.algo.total_error - avg_error) * 
                                            (grid_class.algo.total_error - new_avg_error) / 
                                            (trial_no + 1))
        
            bellman_error_variance = bellman_error_variance ** 2 + ((grid_class.algo.total_error - avg_error) * 
                                            (grid_class.algo.total_error - new_avg_error) / 
                                            (trial_no + 1))
        
            avg_error = new_avg_error
            avg_bellman_error = new_avg_bellman_error
        
        
        errors_per_length.append(avg_error)
        bellman_errors_per_length.append(avg_bellman_error)
        
        vars_per_length.append(avg_error_variance)
        bellman_vars_per_length.append(bellman_error_variance)
    
    
        print('Bellman errors per length is')
        print(bellman_errors_per_length)
    
    for i, b_error in enumerate(bellman_errors_per_length):
        print(f'For i={i}, possible_lengths = {possible_lengths[i]}, we have the bellman error = {b_error}')
        bellman_error_dict_length[possible_lengths[i]] = b_error
        bellman_vars_dict_length[possible_lengths[i]] = bellman_vars_per_length[i]
            
        error_dict_length[possible_lengths[i]] = errors_per_length[i]
        vars_dict_length[possible_lengths[i]] = vars_per_length[i]
    
    with open('bellman_error_dict.pkl', 'wb') as file:
    	pickle.dump(bellman_error_dict_length, file)
    
    with open('bellman_vars_dict.pkl', 'wb') as file:
        pickle.dump(bellman_vars_dict_length, file)
    
    with open('error_dict.pkl', 'wb') as file:
        pickle.dump(error_dict_length, file)
    
    with open('vars_dict.pkl', 'wb') as file:
        pickle.dump(vars_dict_length, file)
    
    # Calculate lower and upper bounds of the confidence interval
    lower_bound = np.array(bellman_errors_per_length) - np.array(bellman_vars_per_length)
    upper_bound = np.array(bellman_errors_per_length) + np.array(bellman_vars_per_length)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(possible_lengths), np.array(bellman_errors_per_length),
             label='Bellman Errors per length', color='b', marker='o')
    plt.fill_between(np.array(possible_lengths), lower_bound, upper_bound, color='b',
                     alpha=0.2, label='Errors per length confidence interval')

    # Add labels and title
    plt.xlabel('Lengths')
    plt.ylabel('Bellman errors')
    plt.title('Bellman errors per length')
    plt.legend()
    
    
    plt.savefig('BellmanErrorsPerLength.png')
    # Show the plot
    plt.show()
    
    # Calculate lower and upper bounds of the confidence interval
    lower_bound = np.array(errors_per_length) - np.array(vars_per_length)      
    upper_bound = np.array(errors_per_length) + np.array(vars_per_length)                       

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(possible_lengths), np.array(errors_per_length),
             label='Average Errors per length', color='b', marker='o')
    plt.fill_between(np.array(possible_lengths), lower_bound, upper_bound, color='b',
                     alpha=0.2, label='Errors per length confidence interval')

    
    # Add labels and title
    plt.xlabel('Lengths')
    plt.ylabel('Average errors')
    plt.title('Average errors per length')
    plt.legend()
    
    plt.savefig('ErrorsPerLength.png')
    # Show the plot
    plt.show()
    
    print('Errors per length tests')
    
    #%%
    
    '''
    Code to test how the errors scale with dimension 
    '''
    errors_per_dimension = []
    bellman_errors_per_dimension = []
    
    vars_per_dimension = []
    bellman_vars_per_dimension = []
    
    error_dict_dimension = {}
    bellman_error_dict_dimension = {}
    
    vars_dict_dimension = {}
    bellman_vars_dict_dimension = {}
    
    possible_dimensions = [2, 3]
    no_of_trials = 1
    
    time_horizon = 3
    gamma = 0.9
    max_lengths = [5 for t in range(time_horizon)]
    stepsizes = 0.2
    max_complexity = 2
    eta = -2
    rho = -5
    
    l = 5
    
    for i,d in enumerate(possible_dimensions):
        
        avg_error = 0                                                           
        avg_bellman_error = 0                                                   
                                                                                        
        avg_error_variance = 0                                                  
        bellman_error_variance = 0                                              
                                                                                
        for trial_no in range(no_of_trials):                             
            
            side_lengths = np.array([l for i in range(d)])                                      
            center = np.zeros(d)
            goal = np.array([-1] + list(np.zeros(d-1)))
            
            
            grid_class = VIDTR_grid(d, center, side_lengths,                        
                                    stepsizes, max_lengths, max_complexity, goal,
                                    time_horizon, gamma, eta, rho)              
            
            int_policies = grid_class.algo.compute_interpretable_policies()     
            
            for t in range(grid_class.time_horizon-1):
                
                int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                            grid_class.algo.optimal_actions[t])
        
        
        new_avg_error = (avg_error * trial_no + grid_class.algo.total_error) / (trial_no + 1)
        new_avg_bellman_error = (avg_bellman_error * trial_no + grid_class.algo.total_bellman_error) /(trial_no + 1)        
        
        avg_error_variance = avg_error_variance ** 2 + ((grid_class.algo.total_error - avg_error) * 
                                            (grid_class.algo.total_error - new_avg_error) / 
                                            (trial_no + 1))                     
        
        bellman_error_variance = bellman_error_variance ** 2 + ((grid_class.algo.total_error - avg_error) * 
                                            (grid_class.algo.total_error - new_avg_error) / 
                                            (trial_no + 1))
        
        errors_per_dimension.append(avg_error)
        bellman_errors_per_dimension.append(avg_bellman_error)
        
        vars_per_dimension.append(avg_error_variance)
        bellman_vars_per_dimension.append(bellman_error_variance)
    
    print('Storing of errors begins now')
    ic(bellman_error_dict_dimension)
    ic(bellman_vars_dict_dimension)
    ic(error_dict_dimension)
    ic(vars_dict_dimension)
    
    for i, b_error in enumerate(bellman_errors_per_dimension):                  
        bellman_error_dict_dimension[possible_dimensions[i]] = b_error
        bellman_vars_dict_dimension[possible_dimensions[i]] = bellman_vars_per_dimension[i]
            
        error_dict_dimension[possible_dimensions[i]] = errors_per_dimension[i]  
        vars_dict_dimension[possible_dimensions[i]] = vars_per_dimension[i]     
    
    with open('bellman_error_dict_dimension.pkl', 'wb') as file:			
        pickle.dump(bellman_error_dict_dimension, file)																
    
    with open('bellman_vars_dict_dimension.pkl', 'wb') as file:
        pickle.dump(bellman_vars_dict_dimension, file)										
    
    with open('error_dict_dimension.pkl', 'wb') as file:
        pickle.dump(error_dict_dimension, file)
    
    with open('vars_dict_dimension.pkl', 'wb') as file:
        pickle.dump(vars_dict_dimension, file)
    
    # Calculate lower and upper bounds of the confidence interval
    lower_bound = np.array(bellman_errors_per_dimension) - np.array(bellman_vars_per_dimension)
    upper_bound = np.array(bellman_errors_per_dimension) + np.array(bellman_vars_per_dimension)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(possible_dimensions), np.array(bellman_errors_per_dimension),
             label='Bellman Errors per dimension', color='b', marker='o')
    plt.fill_between(np.array(possible_dimensions), lower_bound, upper_bound, color='b',
                     alpha=0.2, label='Errors per dimension confidence interval')

    # Add labels and title
    plt.xlabel('Dimensions')
    plt.ylabel('Bellman errors')
    plt.title('Bellman errors per Dimension')
    plt.legend()
    
    
    plt.savefig('BellmanErrorsPerDimension.png')
    # Show the plot
    plt.show()
    
    # Calculate lower and upper bounds of the confidence interval
    lower_bound = np.array(errors_per_dimension) - np.array(vars_per_dimension)
    upper_bound = np.array(errors_per_dimension) + np.array(vars_per_dimension)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(possible_dimensions), np.array(errors_per_dimension),
             label='Average Errors per dimension', color='b', marker='o')
    plt.fill_between(np.array(possible_dimensions), lower_bound, upper_bound, color='b',
                     alpha=0.2, label='Errors per dimension confidence interval')

    
    # Add labels and title
    plt.xlabel('Dimensions')
    plt.ylabel('Average errors')
    plt.title('Average errors per dimension')
    plt.legend()
    
    plt.savefig('ErrorsPerDimension.png')
    # Show the plot
    plt.show()
    
    print('Errors per dimension tests')
    
#%%
