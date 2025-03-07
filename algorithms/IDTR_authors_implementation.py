#.................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................#!/usr/bin/env python3
# ---------------------------------------------------------*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:08:33 2024                                            

@author: badarinath                                                            
"""

import numpy as np                                                             
import matplotlib.pyplot as plt                                                  
import itertools
import disjoint_box_union
import constraint_conditions as cc

from importlib import reload
from itertools import combinations, product

disjoint_box_union = reload(disjoint_box_union)
from disjoint_box_union import DisjointBoxUnion as DBU

cc = reload(cc)


#%%

class IDTR:                                                                     
    '''                                                                        
    We code up the IDTR approach from the paper where we use ridge regression to
    estimate the Q functions.                                                       
                                                                                
    This differs from our approach in how we compute the optimal regions and actions 
    in the final optimization problem. We follow the same approach as the authors                                           

    '''
    
    
    def __init__(self, time_horizon, dimensions, obs_states, obs_actions, obs_rewards,
                 state_spaces, max_lengths, zetas, rhos, gammas, lambdas, complexities, actions,
                 stepsizes):                                                    
        '''
        
        Parameters:
        -----------------------------------------------------------------------
        time_horizon : int
                       The time horizon of the MDP                             
                       
        dimensions : list[T]
                     The dimensions of the space at the various timesteps             
        
        obs_states : list[list]
                     The observed states in the trajectories                   
        
        obs_actions : list[list]
                      The observed actions in the trajectories                 
        
        obs_rewards : list[list]
                      The observed rewards in the trajectories                 
        
        state_spaces : list[DBU][T]
                       The state spaces for the trajectories for the different timesteps
        
        max_lengths : list[T]
                      The maximum allowed lengths for the decision tree at each
                      timestep                                                 
        
        zetas : list[T]
                The volume promotion constants at the different timesteps 
        
        rhos : list[T]
               The complexity constants at the different timesteps 
        
        gammas : list[T]
                 The constant that shows up in the Gaussian Kernel estimation
        
        lambdas : list[T]
                  The regularization constants for the different timesteps
        
        complexities : list[T]
                       The maximum allowable complexity for the conditions at each 
                       timestep of the DBU
        
        actions : list[T]
                  The list of actions we can take in the MDP;
                  We assume the actions are hashable.
        
        stepsizes : float
                    The stepsizes for the DBU at the different time and lengthsteps
            
        '''
        
        self.dimensions = dimensions
        self.obs_states = obs_states
        self.obs_actions = obs_actions
        self.obs_rewards = obs_rewards
        self.state_spaces = state_spaces
        
        if type(max_lengths) == int or type(max_lengths) == float:
            max_lengths = [max_lengths for t in range(time_horizon)]
        
        if type(zetas) == int or type(zetas) == float:
            zetas = [zetas for t in range(time_horizon)]
        
        if type(max_lengths) == int or type(max_lengths) == float:
            max_lengths = [max_lengths for t in range(time_horizon)]
        
        if type(rhos) == int or type(rhos) == float:
            rhos = [rhos for t in range(time_horizon)]
        
        if type(complexities) == int or type(complexities) == float:
            complexities = [complexities for t in range(time_horizon)]
        
        if type(gammas) == float or type(gammas) == int:
            gammas = [gammas for t in range(time_horizon)]
        
        if type(lambdas) == float or type(lambdas) == int:
            lambdas = [lambdas for t in range(time_horizon)] 
        
        self.max_lengths = max_lengths
        self.zetas = zetas
        self.rhos = rhos
        self.complexities = complexities
        self.time_horizon = time_horizon
        self.actions = actions
        self.stepsizes = stepsizes
        self.gammas = gammas
        self.lambdas = lambdas
        
    @staticmethod
    def gaussian_kernel(x, y, gamma=[]):                                        
        '''
        The Gaussian kernel evaluated at (x,y) and weighted by gamma.          
        
        Parameters:
        -----------------------------------------------------------------------
        x : np.array[d] or np.array[n,d]
            d dimensional vector
        
        y : np.array[d] or np.array[m,d]
            d dimensional vector                                               
        
        gamma : np.array[d]
                d dimensional weighting vector
        
        Returns:
        -----------------------------------------------------------------------
        K : np.array[n,m] or float                                             
            (n,m) matrix when x and y are matrices                             
            else a float which represents the kernel evaluation of the two vectors
        
        '''
        if type(gamma) == list:
            if len(gamma) == 0:
                gamma = np.ones(x.shape[0])
        
        if len(x.shape) > 1 and len(y.shape) > 1:
            n = x.shape[0]                                                     
            m = y.shape[0]                                                      
                
            K = np.zeros((n,m))
            
            for i in range(n):
               for j in range(m):
                   K[i,j] = np.exp(-np.sum(gamma * (x-y)**2))
        
            return K
           
        else:
            
            return np.exp(- np.sum(gamma * (x-y)**2))
  
    
    def compute_interpretable_policies(self):
        
        '''
        We estimate the Q values at the different timesteps.
        
        Stores:
        -----------------------------------------------------------------------
        rewards_per_time : list[list]
                           rewards_per_time[l][t] 
        
        states_per_time : list[list]
        
        
        '''
        # 1 / (n_Ta lambda_T) * (K^T K)^{-1} K^T y
        
        rewards_per_time = []
        states_per_time = []
        Q_values_per_time = [lambda x,a : 0]
        pi_hat_per_time = [lambda x : 0]
        
        optimal_scores_per_time = []                                                                                
        optimal_actions_per_time = []                                           
        optimal_q_scores_per_time = []
        optimal_zeta_scores_per_time = []
        optimal_complexity_scores_per_time = []
        optimal_cond_DBUs_per_time = []                        
        
        
        for t in reversed(range(self.time_horizon)):
            
            all_conditions = []                                                                 
            all_condition_DBUs = []                                                         
            
            optimal_scores = []                                                     
            stored_DBUs = []                                                            
            optimal_conditions = []                                                  
            optimal_actions = []                                                            
            optimal_q_scores = []
            optimal_zeta_scores = []
            optimal_complexity_scores = []                                                                                                          
            optimal_cond_DBUs = []
            
            cond_iter_class = disjoint_box_union.ConditionsIterator(self.state_spaces[t],
                                                                    self.complexities[t])
            cond_iterator = iter(cond_iter_class)
            
            remaining_space = self.state_spaces[t]
            
            total_score = 0
            total_q_score = 0
            
            print(f'The state spaces at time {t} is {self.state_spaces[t]}')
            
            null_conditions = 0
            for i,c in enumerate(cond_iterator):
                
                
                if c != None:
                    all_conditions.append(c)
                    
                    print(f'The condition is {c}')
                    con_DBU = c.create_a_DBU(self.stepsizes)
                    if con_DBU.no_of_boxes != 0:                                            
                        all_condition_DBUs.append(con_DBU)
                    
            
            print('Conditions have finished storing')
            condition_DBUs = all_condition_DBUs
            betas = {}
            rewards_dict = {}
            states_dict = {}                                                   
            Q_values = {}                                                      
            current_policy = {}

            #for act_index, action in enumerate(list(self.obs_actions[:,t])):  
            for j in range(len(self.obs_actions)):
                action = tuple(self.obs_actions[j][t])
                print(f'Action is {action}')
                
                if action in rewards_dict.keys():
                    rewards_dict[action].append(self.obs_rewards[j][t])
                    states_dict[action].append(self.obs_states[j][t])
                
                else:
                    rewards_dict[action] = [self.obs_rewards[j][t]]             
                    states_dict[action] = [self.obs_states[j][t]]               
            
            print(f'We have rewards = {rewards_dict[action]}')
            print(f'We have states_per_time = {states_dict[action]}')
            rewards_per_time.insert(0, rewards_dict[action])                    
            states_per_time.insert(0, states_dict[action])                              
            
            
            for action in rewards_dict.keys():                                 
                                                                               
                if type(action) == 'numpy.ndarray' or type(action) == 'list':  
                    action = tuple(action)                                     
                
                print(f'Action is {action}')
                
                X = np.array(states_dict[action])
                print(f'The state matrix is {X}')                              
                
                K = IDTR.gaussian_kernel(X,X, gamma = self.gammas[t])          
                print(f'The Gaussian Kernel is {K}')
                
                n = len(rewards_dict[action])
                print(f'The number of trajectories {n}')
                
                y  = rewards_dict[action]                               
                print(f'The rewards are {y}')
                
                
                Q_value = np.array([ Q_values_per_time[-1](states_dict[action][i],  pi_hat_per_time[-1](states_dict[action][i])) for i in range(len(states_dict[action]))])
                
                betas[(t, action)] = (np.linalg.inv(K + n * self.lambdas[t] * np.eye(K.shape[0])) @ (y + Q_value))
                beta = betas[(t, action)]
                
                def Q_values_functor(x, action=action, beta=beta):
                    print(f'Size of states dict is {len(states_dict[action])}')
                    print(f'Size of Beta is {len(beta)}')
                    print(f'Action is {action}')
                    total = 0
                    for i,s in enumerate(states_dict[action]):
                        total = total + IDTR.gaussian_kernel(x, s) * beta[i]
                    
                    return total
            
                
                Q_values[action] = lambda x : Q_values_functor(x, action = action, beta = betas[(t,action)])
                
                print('States we scan through are')
                print(states_dict[action])
                print(f'Length of the states we scan through {len(states_dict[action])}')
                
                
            Q_values_per_time.insert(0, Q_values)                               
                                                                                
            remaining_space = self.state_spaces[t]                              
            
            print(f'Remaining space is {remaining_space}')
            
            for l in range(self.max_lengths[t] - 1):                                
                                                                                
                max_score_dbu_action = -np.inf                                            
                for cond_index, cond_DBU in enumerate(condition_DBUs):     
                    
                    for act_index, action in enumerate(self.actions[t]):                  
                        score_dbu_action = 0.0                                 
                        
                        for i in range(len(self.obs_states)):               
                            
                            def pi_hat(x, t=t):
                                max_val = -np.inf
                                best_action = None                                  
                                for a in self.actions[t]:
                                    value = Q_values[action](x)
                                    if value > max_val:
                                        max_val = value
                                        best_action = a
                                return best_action
                                    
                            action = tuple(action)
                            
                            print(f'The state is {self.obs_states[i][t]}')
                            
                            q_score = 1 / n * (Q_values[action](self.obs_states[i][t]) *
                                               remaining_space.is_point_in_DBU(self.obs_states[i][t])*cond_DBU.is_point_in_DBU(self.obs_states[i][t]) + 
                                               remaining_space.is_point_in_DBU(self.obs_states[i][t]) * (1 - cond_DBU.is_point_in_DBU(self.obs_states[i][t])) * 
                                               Q_values[tuple(pi_hat(self.obs_states[i][t], t))](self.obs_states[i][t])) 
                            
                            zeta_score = 1/n * self.zetas[t] * remaining_space.is_point_in_DBU(self.obs_states[i][t])*cond_DBU.is_point_in_DBU(self.obs_states[i][t])
                    
                            
                            print(f'Q score is {q_score}')
                            print(f'Zeta score is {zeta_score}')                
                            
                            complexity_score = self.rhos[t] * (self.complexities[t] - cond_DBU.complexity)
                            
                            print(f'Complexity score is {complexity_score}')
                            
                            score_dbu_action += q_score + zeta_score + complexity_score
                            print(f'Total score is {score_dbu_action}')
                            
                        if score_dbu_action > max_score_dbu_action:
                            
                            max_score_dbu_action = score_dbu_action
                            optimal_action = action
                            optimal_condition = all_conditions[cond_index]
                            optimal_cond_DBU = cond_DBU
                            optimal_q_score = q_score
                            optimal_zeta_score = zeta_score
                            optimal_complexity_score = complexity_score
                            optimal_score = score_dbu_action
                
                new_condition_DBUs = []                                          
                
                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                
                total_score += max_score_dbu_action                                       
                
                no_of_null_DBUs = 0
                for i, cond_dbu in enumerate(condition_DBUs):                        
                    sub_DBU = cond_dbu.subtract_DBUs(optimal_cond_DBU)           
                                                                                
                    if sub_DBU.no_of_boxes != 0:                                
                        new_condition_DBUs.append(sub_DBU)                      
                    else:                                                           
                        no_of_null_DBUs = no_of_null_DBUs + 1                   
                
                                                                                
                print(f'Timestep {t} and lengthstep {l}:')                      
                print('--------------------------------------------------------')
                print(f'Optimal condition at timestep {t} and lengthstep {l} is {optimal_condition}')
                print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
                print(f'Optimal conditional DBU at timestep {t} and lengthstep {l} is {optimal_cond_DBU}')
                print(f'Optimal score is {optimal_score}')      
                print(f'Non null DBUs = {len(condition_DBUs)} - {no_of_null_DBUs}')
                print(f'Rho is {self.rhos[t]}, zeta is {self.zetas[t]}')
                
                
                optimal_actions.append(optimal_action)                          
                optimal_q_scores.append(q_score)
                optimal_zeta_scores.append(zeta_score)
                optimal_complexity_scores.append(complexity_score)
                optimal_scores.append(optimal_score)
                optimal_actions.append(optimal_action)
                optimal_cond_DBUs.append(optimal_cond_DBU)
                
            
            max_score = -np.inf
            for a in self.actions[t]:
                
                for cond_index, cond_DBU in enumerate(condition_DBUs):
                    
                    score_dbu_action = 0
                    
                    for i in range(len(self.obs_states)):
                        q_score = 1 / n *  (Q_values[action](self.obs_states[i][t]) * remaining_space.is_point_in_DBU(self.obs_states[i][t]))
                        
                        complexity_score = self.rhos[t] * (self.complexities[t] - all_conditions[cond_index].complexity)
                        
                        score_dbu_action = score_dbu_action + q_score + complexity_score
                        
                        if score_dbu_action > max_score:
                            
                            max_score = score_dbu_action
                            optimal_action = a
                            optimal_cond_DBU = cond_DBU
                            
            
            optimal_actions.append(optimal_action)
            optimal_q_scores.append(q_score)
            optimal_zeta_scores.append(zeta_score)
            optimal_complexity_scores.append(complexity_score)
            optimal_scores.append(optimal_score)
            optimal_actions.append(optimal_action)
            optimal_cond_DBUs.append(optimal_cond_DBU)
            
        
            optimal_actions_per_time.append(optimal_actions)
            optimal_q_scores_per_time.append(optimal_q_scores)
            optimal_zeta_scores_per_time.append(optimal_zeta_scores)
            optimal_complexity_scores_per_time.append(optimal_complexity_scores)
            optimal_scores_per_time.append(optimal_scores)
            optimal_cond_DBUs_per_time.append(optimal_cond_DBUs)
        
        
        self.optimal_actions_per_time = optimal_actions_per_time        
        self.optimal_q_scores_per_time = optimal_q_scores_per_time
        self.optimal_zeta_scores_per_time = optimal_zeta_scores_per_time
        self.optimal_complexity_scores_per_time = optimal_complexity_scores_per_time
        self.optimal_scores_per_time = optimal_scores_per_time
        self.optimal_cond_DBUs_per_time = optimal_cond_DBUs_per_time
        
        return optimal_cond_DBUs_per_time, optimal_actions_per_time
    
    
    @staticmethod
    def get_interpretable_policy(cond_DBUs, actions):
        '''
        Given the conditions defining the policy, obtain the interpretable policy
        implied by the conditions.                                             
                                                                               
        Parameters:                                                            
        -----------------------------------------------------------------------
        conditions : np.array[l]                                               
                     The conditions we want to represent in the int. policy              
                                                                               
        actions : np.array[l]                                                  
                  The actions represented in the int. policy                   
                  
        '''
        def policy(state):                                                     
            
            for i, cond_DBU in enumerate(cond_DBUs):                                
                                                                                 
                if cond_DBU.is_point_in_DBU(state):                                  
                                                                                
                    return actions[i]                                           
            
                                                                                
            return actions[-1]                                                  
        
        return policy                                                           
    
                                                                                
    def plot_scores(self, score_bounds = [], q_score_bounds = [],              
                    zeta_score_bounds = [], complexity_score_bounds = []):                                                     
        '''                                                                    
        Plot the errors obtained after we perform the VIDTR algorithm           
                                                                                
        '''                                                                    
        for t in range(len(self.max_lengths)):                                  
            plt.plot(np.arange(len(self.optimal_q_scores_per_time[t])), self.optimal_q_scores_per_time[t])
            plt.title(f'Q Scores at time {t}')                                   
            plt.xlabel('Time')                                                  
            plt.ylabel('Q-scores')                                                
            
            if len(q_score_bounds) > 0:                                             
                plt.plot(np.arange(len(q_score_bounds[t])), q_score_bounds[t], label = 'Q-score-bounds')
            
            plt.legend()                                                        
            plt.show()                                                          
            
            plt.plot(np.arange(len(self.optimal_scores_per_time[t])), self.optimal_scores_per_time[t])
            plt.title(f'Optimal scores at time {t}')                           
            plt.xlabel('Time')                                                 
            plt.ylabel('Optimal score')
            
            if len(score_bounds) > 0:
                plt.plot(np.arange(len(score_bounds[t])), score_bounds[t], label = 'Score bounds')
                                                   
            plt.show()                                                               
