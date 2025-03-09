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

import IVIDTR_algo_dict_to_func_parallelized_debugged as VIDTR_module
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc

#%%

#importlib.reload(constraint_conditions)
#importlib.reload(disjoint_box_union)
#importlib.reload(VIDTR_module)
#importlib.reload(mdp_module)
#importlib.reload(VIDTR_envs)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_algo_dict_to_func_parallelized_debugged import VIDTR

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
DIMENSIONS = [2, 3, 4]

for dimensions in DIMENSIONS:
    for max_lengths in MAXLENGTHS:
        center = np.array([0 for d in range(dimensions)])
        side_lengths = np.array([6 for d in range(dimensions)])
        goal = np.array([-1] + [0 for d in range(dimensions-1)])
        time_horizon = 3
        gamma = 0.9
        max_lengths = [max_lengths for t in range(time_horizon)]
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
        
        error_per_trial = []
        bellman_error_per_trial = []
        
        for t in range(trial_number):
            
            grid_class.algo.compute_interpretable_policies(integration_method = DBU.integrate_static)
    
            error_per_trial.append(grid_class.algo.total_error)
            bellman_error_per_trial.append(grid_class.algo.total_bellman_error)
        
        errors_per_integ_method.append(error_per_trial)    
        bellman_errors_per_integ_method.append(bellman_error_per_trial)
        
        error_per_trial = []                                                        
        bellman_error_per_trial = []                                                
        
        for t in range(trial_number):
            
            grid_class.algo.compute_interpretable_policies(integration_method = DBU.easy_integral)
    
            error_per_trial.append(grid_class.algo.total_error)
            bellman_error_per_trial.append(grid_class.algo.total_bellman_error)
        
        errors_per_integ_method.append(error_per_trial)    
        bellman_errors_per_integ_method.append(bellman_error_per_trial)
        
        
        plot_confidence_intervals(errors_per_integ_method, 'Total Errors Per Method',
                                  labels, f'epm_dim_{dimensions}_maxl_{max_lengths}')
        
        plot_confidence_intervals(bellman_errors_per_integ_method,
                                  'Total Bellman Errors Per Method', labels,
                                  f'bpm_dim_{dimensions}_maxl_{max_lengths}')
    
    #%%
    
    '''
    VIDTR - tests for different \rho and \eta values
    '''
    etas = [0.1, 0.5, 1.0, 2.0]
                                                                                
    rhos = [-0.1, -0.5, -1.0, -5.0]                                             
                                                                                
    dimensions = 2                                                                  
    center = np.array([0, 0])                                                           
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])
    time_horizon = 2                                                                                        
    gamma = 0.9                                                                                          
    max_lengths = [5 for t in range(time_horizon)]                                                          
    stepsizes = 0.1                                                                                         
    max_complexity = 2
    
    for i, eta in enumerate(etas):
        for j, rho in enumerate(rhos):                                                                                          
                                                                        
            grid_class = VIDTR_grid(dimensions, center, side_lengths,
                                       stepsizes, max_lengths, max_complexity, goal,
                                       time_horizon, gamma, eta, rho)
            
            int_policies = grid_class.algo.compute_interpretable_policies()
            
            for t in range(grid_class.time_horizon-1):
                
                int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                            grid_class.algo.optimal_actions[t])
                    
                grid_class.env.plot_policy_2D(int_policy, title=f'Time{t}, eta = {eta}, rho = {rho}')

    
    #%%
    
    '''
    VIDTR - tests for different max_lengths
    '''
    
    maxi_lengths = [2,3,4,5]
    eta = -0.0001
    rho = 0.0001
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])                                                                                                
    time_horizon = 4
    gamma = 0.9
    stepsizes = 0.1
    max_complexity = 2
    
    for i, l in enumerate(maxi_lengths):
        
        max_lengths = [l for t in range(time_horizon)]
        grid_class = VIDTR_grid(dimensions, center, side_lengths, stepsizes, max_lengths,
                                   max_complexity, goal, time_horizon, gamma, eta, rho)
        
        int_policies = grid_class.algo.compute_interpretable_policies()
        
        for t in range(grid_class.time_horizon-1):
            int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],                      
                                                        grid_class.algo.optimal_actions[t])
                
            grid_class.env.plot_policy_2D(int_policy, title=f'Time{t}, max_length = {l}')

    
    #%%                                                                        
    '''
    VIDTR - tests for different sidelengths
    '''
    
    eta = -0.1
    rho = 0.3
    dimensions = 2
    center = np.array([0, 0])
    goal = np.array([-1, 0])
    time_horizon = 4
    gamma = 0.9
    stepsizes = 0.1
    max_complexity = 2
    eta = -3/80
    rho = 1/50
    
    max_lengths = [5 for t in range(time_horizon)]
    possible_lengths = [4,5,6,7,8]
    
    errors_per_length = []
    bellman_errors_per_length = []
    
    for i,s in enumerate(possible_lengths):
        
        side_lengths = np.array([s,s])
        grid_class = VIDTR_grid(dimensions, center, side_lengths, stepsizes, max_lengths,
                                   max_complexity, goal, time_horizon, gamma, eta, rho)
        
        int_policies = grid_class.algo.compute_interpretable_policies()
        
        for t in range(grid_class.time_horizon):
            int_policy = VIDTR.get_interpretable_policy(grid_class.algo.optimal_conditions[t],
                                                        grid_class.algo.optimal_actions[t])
                
            grid_class.env.plot_policy_2D(int_policy, title=f'Time{t}, max_length = {s}')
        
        errors_per_length.append(grid_class.algo.total_error)
        bellman_errors_per_length.append(grid_class.algo.total_bellman_error)
        
    
    plt.plot(possible_lengths, errors_per_length, label='Errors per lengthstep')
    plt.plot(possible_lengths, bellman_errors_per_length, label = 'Bellman Errors per lengthstep')
    plt.title('Errors per lengthstep')
    plt.legend()
    plt.show()
    
    #%%
    '''
    VIDTR - tests for different dimensions
    '''
    
    eta = -0.1                                                                                      
    rho = 0.3
    center = np.array([0, 0])
    goal = np.array([-1, 0])                                                    
    time_horizon = 4                                                            
    gamma = 0.9                                                                 
    stepsizes = 0.1                                                             
    max_complexity = 2                                                          
    eta = -3/80                                                                     
    rho = 1/50
    side_lengths = np.array([6,6])
    
    max_lengths = [5 for t in range(time_horizon)]                                                      
    possible_dims = [3,4,5,6,7]                                                 
    
    errors_per_dim = []                                                         
    bellman_errors_per_dim = []                                                 
    
    for i,dim in enumerate(possible_dims):
        
        grid_class = VIDTR_grid(dim, center, side_lengths, stepsizes, max_lengths,
                                   max_complexity, goal, time_horizon, gamma, eta,            
                                   rho)                                             
        
        int_policies = grid_class.algo.compute_interpretable_policies()
        
        errors_per_dim.append(grid_class.total_error)
        bellman_errors_per_dim.append(grid_class.total_bellman_error)
    
    plt.plot(possible_dims, errors_per_dim, label = 'Errors_per_dimension')
    plt.plot(possible_dims, bellman_errors_per_dim, label = 'Bellman_errors_per_dimension')
    plt.title('Errors per dimension')
    plt.legend()
    plt.show()
    #%%
    # Tests VIDTR - Gaussian Kernel
    
    # Create a class to represent these elements
    
    state_space = DBU(1, 2, np.array([[3,3]]), np.array([[0,0]]),
                      stepsizes = 0.5)
    
    print(f'The stepsizes of the state space is {state_space.stepsizes}')
    
    action_space = np.array([0,1,2])
    
    time_horizon = 3
    
    gamma = 0.9
    
    variance = 1.0
    
    # We look at the performance on the random walk kernel P(.|s,a) ~ N(s+a, tuning_parameter)
    
    def gaussian_kernel_constant(state_space, action,
                                 initial_state, variance):
        
        const = 0
        state_iter_class = disjoint_box_union.DBUIterator(state_space)
        state_iterator = iter(state_iter_class)
        for s_new in state_iterator:
            #print(-np.sum((np.array(initial_state) + action - np.array(s_new))**2) / variance)
            const += np.exp(-np.sum((np.array(initial_state) + action - np.array(s_new)**2) / variance))
        
        return const
            
    
    def gaussian_kernel(state1, state2, action, state_space,
                        action_space, integrating_const = None):
        
        if integrating_const == None:
            integrating_const = gaussian_kernel_constant(state_space, action,
                                                         state1, variance)
        
        
        return np.exp(-np.sum((np.array(state1) + action - np.array(state2)**2))) / integrating_const
        
    def reward(state, action, state_space, action_space):                           
        
        if state_space.point_count == None:
            state_space.point_count = state_space.no_of_points()
        
        return np.linalg.norm(2 * (np.array(state) / state_space.point_count + action))
    
    #%%
    state_iter_class = disjoint_box_union.DBUIterator(state_space)
    state_iterator = iter(state_iter_class)
    
    state_list = []
    
    for i,s in enumerate(state_iterator):
        #print(f'The {i}th state is given by {s}')
        state_list.append(s)
    
    print(f'The kernel constant is given by {gaussian_kernel_constant(state_space, 0, np.array(state_list[0]), 0.5)}')
    print(f'The reward is given by {reward(np.array([0,1]), 1.2, state_space, action_space)}')
    print(f'The kernel is given by {gaussian_kernel(np.array([0,1]), np.array([0.5,0.5]), 0, state_space, 2)}')
    #%%
    transitions = [gaussian_kernel for t in range(time_horizon)]
    rewards = [reward for t in range(time_horizon)]
    max_lengths = [5 for t in range(time_horizon)]
    eta = 0.5
    rho = 0.3
    stepsizes = 0.1
    max_complexity = 2
    
    centering_MDP = MDP(state_space, action_space, time_horizon, gamma, transitions, rewards)
    #%%
    algo = VIDTR(centering_MDP, max_lengths, eta, rho, max_complexity,
                 stepsizes = 0.5, max_conditions=50)
    #%%
    '''
    Tests for maximum_over_actions
    '''
    transitions = [gaussian_kernel for t in range(time_horizon)]
    rewards = [reward for t in range(time_horizon)]
    max_lengths = [4 for t in range(time_horizon)]
    eta = 0.5
    rho = 0.3
    stepsizes = 0.1
    max_complexity = 2
    
    MDP_1 = MDP(state_space, action_space, time_horizon, gamma, transitions, rewards)
    algo = VIDTR(MDP_1, max_lengths, eta, rho, max_complexity, stepsizes = 0.5, max_conditions = 10)
    function = lambda s,a : np.sum((s-a)**2)
    maxi_function = lambda s : algo.maximum_over_actions(function)(s)
    
    points = []
    values = []
    
    actions_to_symbols = ['+', 'x', '*']
    
    for x in np.arange(-10, 10.5, 0.5):
        for y in np.arange(-10, 10.5, 0.5):
            point = np.array([x,y])
            points.append(point)
            values.append(maxi_function(point))
    
    for i,p in enumerate(points):
        plt.scatter(p[0], p[1], marker = actions_to_symbols[values[i]])
    
    print(values)
    #%%
    '''
    Tests for bellman_function(2)
    '''
    points = []
    for x in np.arange(-2, 2.5, 0.5):
        for y in np.arange(-2, 2.5, 0.5):
            point = np.array([x,y])
            print(f'{algo.bellman_function(2)(point, 1)}')
    
    
    #%%
    '''
    Tests for fix_a
    '''
    f = lambda s, a: np.sum((s-a)**2)
    VIDTR.fix_a(f, a=1)(np.array([4,4]))
    
    #Toy example where we know what happens
    #IDTR
    #Dataset from basal-bolus
    
    #%%
    '''
    Tests for redefine_function
    '''
    h = lambda s : np.sum(s) % 3
    g = VIDTR.redefine_function(h, np.array([0,0]), 1.0)
    print(g(np.array([0,0])))
    print(g(np.array([0,1])))
    print(g(np.array([1.3,2.5])))
    print(g(np.array([-1,2])))
    #%%
    '''
    Tests for compute_optimal_policies
    '''
    points = []
    values = []
    
    algo.compute_optimal_policies()
    
    actions_to_symbols = ['+', 'x', '*']
    actions_to_colors = ['blue', 'green', 'red']
    
    for t in range(3):
        for x in np.arange(-1, 1.5, 0.5):
            for y in np.arange(-1, 1.5, 0.5):
                
                point = np.array([x,y])
                points.append(point)
                
                values.append(algo.optimal_values[t](point))
        
        
        for i,point in enumerate(points):
            plt.scatter(point[0], point[1], 
                        marker = actions_to_symbols[int(algo.optimal_policies[t](point))],
                        c = actions_to_colors[int(algo.optimal_policies[t](point))])
        
        plt.title(f'Optimal policy at time {t}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.show()
        
    #%%
    
    for t in range(3):
        for x in np.arange(-2, 2.5, 0.5):
            for y in np.arange(-2, 2.5, 0.5):
                point = np.array([x,y])
                print(algo.optimal_values[t](point))
                print(algo.optimal_policies[t](point))
    
    
    #%%
    '''
    Tests for constant_eta_function
    '''
    algo.constant_eta_function()(np.array([5,0]), 1)
    
    #%%
    '''
    Tests for compute_interpretable_policies
    '''
    int_policies = algo.compute_interpretable_policies()
    
    #%%
    '''
    Reproduce the max over actions term from the VIDTR_algorithm
    '''
    
    int_function = lambda s : algo.maximum_over_actions(algo.bellman_function(2))(s)
    int_function(np.array([0,0]))

#%%
    '''
    Tests for tuplify_2D_array    
    '''
    A = np.array([[1,3], [0, 5], [-1, 3]])
    print(VIDTR.tuplify_2D_array(A))
    