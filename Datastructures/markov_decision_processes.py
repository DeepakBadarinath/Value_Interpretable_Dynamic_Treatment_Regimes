#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 09:53:59 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

#%%

class MarkovDecisionProcess:
    
    '''
    Class that represents a Markov Decision Process, with the states, actions,
    transitions and rewards
    '''
    
    def __init__(self, dimensions, state_spaces, action_spaces, time_horizon,
                 gamma, transition_kernels, reward_functions):
        
        '''
        Parameters:                                                            
        -----------------------------------------------------------------------
        dimensions : list[int]
                     The dimensions of the state spaces of the MDP at the different timesteps
        
        state_spaces : list[DBU]
                       The state spaces of the MDP at the different timesteps                                   
         
        action_spaces : list[list]  
                        The action spaces for the MDP at the different timesteps                                      
                  
        time_horizon : int                                                     
                       The time horizon for the MDP                            
                                                                               
        gamma : float                                                          
                The discount factor for the MDP
        
        transition_kernels : list[function(s' \in state_spaces[t], s \in state_spaces[t], a \in action_spaces[], state_space, action_space) \to [0,1]]
                             List of length T which consists of probability
                             transition maps.
                             Here the sum of the transition_kernels(s',s,a) for
                             all s' in states = 1
        
        reward_functions : list[function(state, action, state_space, action_space) \to \mathbb{R}]
                           List of length T which consists of reward_functions 
        
        '''
        
        self.dimensions = dimensions
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.time_horizon = time_horizon
        self.gamma = gamma                                                     
        self.transition_kernels = transition_kernels                           
        self.reward_functions = reward_functions     


