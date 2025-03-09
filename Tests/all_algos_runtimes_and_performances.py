#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:10:30 2024

@author: badarinath
"""

# Library imports
import random
import numpy as np
import matplotlib.pyplot as plt                                                

# Trees and decision sets import
import Decision_tree_set
import Trees
from Trees import DecisionTree, TreeNode
from Decision_tree_set import DecisionTreeSet

#Algorithm imports
import IDTR_implementation
from IDTR_implementation import IDTR                                           

import IVIDTR_algo_dict_to_func                                                
from IVIDTR_algo_dict_to_func import VIDTR
                                                                                    
import Juliens_paper                                                           
from Juliens_paper import DPTPA                                                                   
                                                                               
import VIDTR_envs
from VIDTR_envs import GridEnv

import markov_decision_processes
from markov_decision_processes import MarkovDecisionProcess as MDP

#%%
import IDTR_implementation
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions as cc

import importlib
from importlib import reload

#%%

importlib.reload(cc)
importlib.reload(disjoint_box_union)
importlib.reload(IDTR_implementation)

importlib.reload(mdp_module)
importlib.reload(VIDTR_envs)


from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IDTR_implementation import IDTR

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
                                                                                
class algo_environment:
    
    '''
    Build the algorithm environment for the IDTR on a grid
    
    '''                                                                         
    
    def __init__(self, time_horizon, dimensions, center, side_lengths, max_lengths,
                 max_complexity, goal, gamma, zeta, rho, stepsizes, algo=None):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        time_horizon : int
                       The time horizon for the MDP         
        
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
                       
        gamma : float
                Discount factor
        
        zeta : float
               Volume promotion constant    
        
        rho : float
              Complexity constant
        
        
        stepsizes : float
                    The stepsizes for the DBU at the different time and lengthsteps
        
        algo : Algorithm
               The algorithm we wish to have as input in our method

        Stores:
        -----------------------------------------------------------------------
        envs : list[GridEnv]
               The 2D environments for the grid for the different timesteps
        
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
        
        self.zeta = zeta
        self.rho = rho
        self.max_complexity = max_complexity
        
        self.env = GridEnv(dimensions, center, side_lengths, goal)
        
        self.action_spaces = [self.env.actions for t in range(time_horizon)]
        self.state_spaces = [self.env.state_space for t in range(time_horizon)]
        
        transitions = [self.env.transition for t in range(time_horizon)]
        rewards = [self.env.reward for t in range(time_horizon)]
        
        self.MDP = MDP(self.dimensions, self.state_spaces, self.action_spaces,
                       time_horizon, gamma, transitions, rewards)
        
        self.rewards = [self.env.reward for t in range(time_horizon)]
        
        self.algo = algo
        
                
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
            s = np.squeeze(self.state_spaces[0].pick_random_point())
            obs_states[-1].append(s)
            
            for t in range(self.time_horizon):
                
                a = random.sample(self.action_spaces[t], 1)[0]
                r = self.rewards[t](s,a)
                
                s = self.env.move(s,a)
                obs_states[-1].append(s)
                obs_actions[-1].append(a)
                obs_rewards[-1].append(r)
                
            
        return obs_states, obs_actions, obs_rewards
    
#%%
                                                                                
if __name__ == '__main__':                                                      
   
    
    # Set environmental constants
    
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])                                                    
    time_horizon = 3                                                           
    gamma = 0.9
    max_lengths = [5 for t in range(time_horizon)]
    stepsizes = 0.1
    max_complexity = 2
    
    # Set IDTR constants
    IDTR_zeta = -100/600
    IDTR_rho = 6.4                                                                                                                                                    
    IDTR_lambdas = 5


    # Set VIDTR constants
    VIDTR_zeta = -100/600                                                      
    VIDTR_rho = 7.0                                                            
    VIDTR_lambdas = 5                                                           
    

    #%%

    #Set algorithm environments on grid based data for IDTR and VIDTR
    
    grid_class = algo_environment(time_horizon, dimensions, center,             
                                  side_lengths, max_lengths,                    
                                  max_complexity, goal, gamma, IDTR_zeta,            
                                  IDTR_rho, stepsizes)                               
    
    N = 50
    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
    
    
    IDTR_env = algo_environment(time_horizon, dimensions, center, side_lengths,
                                max_lengths, max_complexity, goal, gamma,      
                                IDTR_zeta, IDTR_rho, stepsizes, algo = IDTR)                       
    
    IDTR_env.algo = IDTR(time_horizon, dimensions, obs_states, obs_actions,    
                         obs_rewards, grid_class.state_spaces, max_lengths,    
                         IDTR_zeta, IDTR_rho, gamma, IDTR_lambdas,             
                         max_complexity, grid_class.env.actions, stepsizes)        
    
    VIDTR_env = algo_environment(time_horizon, dimensions, center, side_lengths,
                                 max_lengths, max_complexity, goal, gamma,     
                                 VIDTR_zeta, VIDTR_rho, stepsizes, algo = VIDTR)
    
    VIDTR_env.algo = VIDTR(VIDTR_env.MDP, max_lengths, VIDTR_zeta, VIDTR_rho, max_complexity,
                           stepsizes)                                          

    #%%
    '''
    Tests GridEnv
    '''

    if __name__ == '__main__':
        
        dimensions = 2
        center = np.array([0, 0])
        side_lengths = np.array([6, 6])
        goal = np.array([-1, 0])
        time_horizon = 3                                                            
        gamma = 0.9                                                             
        max_lengths = [5 for t in range(time_horizon)]                          
        stepsizes = 0.1                                                         
        max_complexity = 2                                                      
        zeta = -100/600                                                               
        rho = 6.4                                                                  
        gamma = 0.9                                                                
        lambdas = 5                                                                 
                                                                                
        grid_class = algo_environment(time_horizon, dimensions, center, side_lengths, max_lengths,
                                      max_complexity, goal, gamma, zeta, rho, stepsizes)
                                                                                        
        N = 50                                                                 
        obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
                                                                                    
        
        algo_IDTR = IDTR(time_horizon, dimensions, obs_states, obs_actions, obs_rewards,
                         grid_class.state_spaces, max_lengths, zeta, rho, gamma, lambdas, max_complexity,
                         grid_class.action_spaces, stepsizes)                   
        
        algo_VIDTR = VIDTR(grid_class.MDP, max_lengths, zeta, rho, max_complexity,
                           stepsizes)                                                               
                                                                                
    #%%                                                                         
        '''                                                                       
        Tests for compute interpretable policies                                                              
        '''                                                                                     
        optimal_conditions_IDTR, optimal_actions_IDTR = algo_IDTR.compute_interpretable_policies()        
        optimal_conditions_VIDTR, optimal_actions_VIDTR = algo_VIDTR.compute_interpretable_policies()
                                                                                
    #%%                                                                        
        '''                                                                       
        Policy comparisons : VIDTR vs IDTR                                                
        
        '''
    
        for t in range(len(algo_IDTR.max_lengths)):                                  
            
            plt.plot(np.arange(len(algo_IDTR.optimal_q_scores_per_time[t])),    
                     algo_IDTR.optimal_q_scores_per_time[t],                    
                     label='IDTR Q-scores')                                     
            
            plt.plot()
            
            plt.title(f'Q Scores at time {t}: IDTR VS VIDTR')                                   
            plt.xlabel('Time')                                               
            plt.ylabel('Q-scores')                                                
                                                                                
            if len(algo_IDTR.optimal_q_scores_per_time[t]) > 0:                                                 
                
                plt.plot(np.arange(algo_IDTR.optimal_q_scores_per_time[t]),
                         algo_IDTR.optimal_q_scores_per_time[t],                
                         label = 'Q-score-bounds')                             
                
                
                
            plt.legend()
            plt.show()                                                         
            
            plt.plot(np.arange(len(self.optimal_scores_per_time[t])),
                     self.optimal_scores_per_time[t])
            
            plt.title(f'Optimal scores at time {t}')                           
            plt.xlabel('Time')                                                 
            plt.ylabel('Optimal score')
            
            if len(score_bounds) > 0:
                plt.plot(np.arange(len(score_bounds[t])), score_bounds[t], label = 'Score bounds')
                                                   
            plt.show()
    
        
    #%%
        '''
        Runtime comparisons
        
        '''
        
        
        
        
        