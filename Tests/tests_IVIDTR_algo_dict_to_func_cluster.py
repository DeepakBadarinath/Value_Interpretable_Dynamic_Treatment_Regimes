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
import time
import pickle

#%%
import IVIDTR_algo_dict_to_func as VIDTR_module
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc

#%%

importlib.reload(constraint_conditions)
importlib.reload(disjoint_box_union)
importlib.reload(VIDTR_module)
importlib.reload(mdp_module)
importlib.reload(VIDTR_envs)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_algo_dict_to_func import VIDTR

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
    max_lengths = [3 for t in range(time_horizon)]
    stepsizes = 0.1
    max_complexity = 2
    eta = -100/120
    rho = 10.2
    
    grid_class = VIDTR_grid(dimensions, center, side_lengths,
                            stepsizes, max_lengths, max_complexity, goal,
                            time_horizon, gamma, eta, rho)
    #%%
    N = 5
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
        bellman_val = grid_class.VIDTR_MDP.reward_functions[2](s,a) 
        if bellman_val > max_val:
            best_action = a
    
    best_action


#%%                                                                               
    '''
    Tests for compute_optimal_policies                                          
    '''
    points = []                                                                  
    values = []                                                                                  
                                                                                
    optimal_policies, optimal_values = grid_class.algo.compute_optimal_policies()
    env = GridEnv(dimensions, center, side_lengths, goal)                        
                                                                                                                    
    print(grid_class.algo.MDP.state_spaces[0])                                  
                                                                                                                                                                                                        
    for t in range(time_horizon):                                                   
        print(optimal_policies[t](np.array([0,2])))                              
                                                                                
    for t in range(time_horizon):                                               
        env.plot_policy_2D(optimal_policies[t], title = f'Actions at time {t}')
        
#%%                                                                            
    '''                                                                        
    Tests for printing optimal values and actions                              
    '''
                                                                                
    for t in range(grid_class.time_horizon):                                                
        states = disjoint_box_union.DBUIterator(grid_class.algo.MDP.state_spaces[t]) 
        iter_state = iter(states)                                               
    
        print('Optimal actions are')                                            
        for s in iter_state:                                                    
            print(optimal_policies[0](np.array(s)))                                 
    
        print('Optimal values are')                                            
        states = disjoint_box_union.DBUIterator(grid_class.algo.MDP.state_spaces[t])
        iter_state = iter(states)
        for s in iter_state:                                                    
            print(f'The value at {s} is {optimal_values[0](np.array(s))}')      
                                                                                
    #%%
    '''
    Tests for compute_interpretable_policies                                    
    '''
    optimal_conditions, optimal_actions = grid_class.algo.compute_interpretable_policies(integration_method=DBU.integrate_static) 
    
    #%%
    '''
    VIDTR - plot errors
    '''                                                                        
    grid_class.algo.plot_errors()                                                       
                                                                                
    #%%                                                                        
    print(grid_class.algo.optimal_actions[0])                                   
    
    
    #%%
    '''
    VIDTR - get interpretable policy
    '''
    for t in range(grid_class.time_horizon-1):
        
        int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                    grid_class.algo.optimal_actions[t])
            
        grid_class.env.plot_policy_2D(int_policy, title=f'Int. policy at time {t}')
        
        
    #%%
    for i,c in enumerate(grid_class.algo.optimal_conditions[0]):
        print(f'The {i} lengthstep condition is given by')
        print(c)
        print(f'The {i}th action is given by')
        print(grid_class.algo.optimal_actions[0][i])
    

    #%%
    
    def plot_confidence_intervals(errors_list, title, labels, figure_title):
        
        num_methods = len(errors_list)
        means = []
        half_std_devs = []

        for method_errors in errors_list:
            method_errors = np.array(method_errors)
            mean = np.mean(method_errors)
            std_dev = np.std(method_errors)
            means.append(mean)
            half_std_devs.append(std_dev / 2)  # Half of the standard deviation
    
        # Create a plot with error bars (mean ± half std_dev)
        plt.figure(figsize=(8, 5))
        x = np.arange(num_methods)  # X-axis: Method indices
        plt.errorbar(x, means, yerr=half_std_devs, fmt='o', capsize=5)
        
        plt.xticks(x, labels)  # Custom labels for methods
        plt.xlabel('Integration Method')
        plt.ylabel('Error')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(figure_title)
        plt.show()
    
#%%

MAXLENGTHS = [3, 4, 5]
DIMENSIONS = [2]

times_per_integ_method = {}


for dimensions in DIMENSIONS:                                                                                                   
    for max_length in MAXLENGTHS:
        start_time = time.time()
        center = np.array([0 for d in range(dimensions)])
        side_lengths = np.array([6 for d in range(dimensions)])
        goal = np.array([-1] + [0 for d in range(dimensions-1)])
        time_horizon = 3
        gamma = 0.9
        max_lengths = [max_length for t in range(time_horizon)]
        stepsizes = 0.1
        max_complexity = 2
        eta = -100/120
        rho = 10.2
        
        grid_class = VIDTR_grid(dimensions, center, side_lengths,
                                stepsizes, max_lengths, max_complexity, goal,
                                time_horizon, gamma, eta, rho)    
        
        # Given a function and a parameter, run different trials of the function
        percentages = [0.2, 0.8]
        trial_number = 3
        total_errors = []
        bellman_total_errors = []
        
        labels = [f'S-integral={p}' for p in percentages]
        labels.append('Full integral')
        labels.append('Easy Integral')
        
        errors_per_integ_method = []
        bellman_errors_per_integ_method = []
        for i, p in enumerate(percentages):
            
            error_per_trial = []
            bellman_error_per_trial = []
            
            for t in range(trial_number):
                grid_class.algo.compute_interpretable_policies(integration_method =
                                                               DBU.sampling_integrate_static,
                                                               integral_percent=
                                                               p)               
                
                error_per_trial.append(grid_class.algo.total_error)             
                bellman_error_per_trial.append(grid_class.algo.total_bellman_error)
            
            errors_per_integ_method.append(error_per_trial)
            bellman_errors_per_integ_method.append(bellman_error_per_trial)
            
            print(type(dimensions))
            print(type(max_length))
            
            times_per_integ_method[(dimensions, max_length, p, 0)] = time.time() - start_time
            filename = "dim_{dimensions}_ml_{max_length}_intm_s_integral_per_{p}.txt"
            outfile = open(filename, 'w')
            
            outfile.write(str(times_per_integ_method[(dimensions, max_length, p, 0)]))

            outfile.close() #Close the file when we’re done!                   
            
            
        error_per_trial = []                                                    
        bellman_error_per_trial = []
        
        start_time = time.time()
        for t in range(trial_number):
            
            grid_class.algo.compute_interpretable_policies(integration_method = DBU.integrate_static)
    
            error_per_trial.append(grid_class.algo.total_error)
            bellman_error_per_trial.append(grid_class.algo.total_bellman_error)
        
        errors_per_integ_method.append(error_per_trial)    
        bellman_errors_per_integ_method.append(bellman_error_per_trial)
        
        print(type(dimensions))
        print(type(max_length))
        
        times_per_integ_method[(dimensions, max_length, 1)] = time.time() - start_time
        
        filename = "dim_{dimensions}_ml_{max_length}_intm_s_integral_static.txt"
        outfile = open(filename, 'w')                                          
            
        outfile.write(str(times_per_integ_method[(dimensions, max_length, 1)]))

        outfile.close() #Close the file when we’re done!                       

        error_per_trial = []
        bellman_error_per_trial = []
        
        start_time = time.time()
        for t in range(trial_number):                                                                    
            
            grid_class.algo.compute_interpretable_policies(integration_method = DBU.easy_integral)
    
            error_per_trial.append(grid_class.algo.total_error)                 
            bellman_error_per_trial.append(grid_class.algo.total_bellman_error)
        
        errors_per_integ_method.append(error_per_trial)    
        bellman_errors_per_integ_method.append(bellman_error_per_trial)
        
        times_per_integ_method[(dimensions, max_length, 2)] = time.time() - start_time
        
        filename = "dim_{dimensions}_ml_{max_length}_int_easy.txt"
        outfile = open(filename, 'w')
            
        outfile.write(str(times_per_integ_method[(dimensions, max_length, 2)]))

        outfile.close() #Close the file when we’re done!
        
        
        plot_confidence_intervals(errors_per_integ_method, 'Total Errors Per Method',
                                  labels, f'epm_dim_{dimensions}_maxl_{max_length}')
        
        plot_confidence_intervals(bellman_errors_per_integ_method,
                                  'Total Bellman Errors Per Method', labels,
                                  f'bpm_dim_{dimensions}_maxl_{max_length}')
    
f = open("times_per_integ_method.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(times_per_integ_method, f)

# close file
f.close()
