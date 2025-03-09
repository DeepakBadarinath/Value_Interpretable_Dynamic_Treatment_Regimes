#............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................#!/usr/bin/env python3
#----------------------------------------------------------*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:08:33 2024                                            

@author: badarinath                                                            
"""

import numpy as np                                                             
import matplotlib.pyplot as plt                                                  
import itertools
import disjoint_box_union
import constraint_conditions as cc
import icecream

from icecream import ic

from importlib import reload
from itertools import combinations, product


disjoint_box_union = reload(disjoint_box_union)
from disjoint_box_union import DisjointBoxUnion as DBU

cc = reload(cc)
                                                                                                
ic.disable()
#%%

class IDTR:                                                                     
    '''
    We code up the IDTR approach from the paper where we use ridge regression to
    estimate the Q functions.                                                  

    '''
    
    
    def __init__(self, time_horizon, dimensions, obs_states, obs_actions, obs_rewards,
                 state_spaces, max_lengths, zetas, rhos, gammas, lambdas,
                 complexities, action_spaces, stepsizes):                       
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
        
        action_spaces : list[list]
                      The list of actions we can take in the MDP on the different timesteps;
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
        self.action_spaces = action_spaces
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
        rewards_per_time : list[dicts]
                           rewards_per_time[t]['action'] = Estimate of rewards at 
                           time t and action = r_t(s,a) is the estimated rewards
                          
                           rewards_per_time[t] is a list of dictonarys for eg.
                           rewards[action] = {}
        
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
            
            ic(self.state_spaces[t])
            
            for i,c in enumerate(cond_iterator):
                
                if c != None:
                    all_conditions.append(c)
                    
                    ic(c)
                    con_DBU = c.create_a_DBU(self.stepsizes)
                    if con_DBU.no_of_boxes != 0:                                            
                        all_condition_DBUs.append(con_DBU)
                
                else:
                    break
                    
            print('Conditions have finished storing')
            condition_DBUs = all_condition_DBUs
            betas = {}
            rewards_dict = {}
            states_dict = {}                                                   
            Q_values = {}                                                      
            current_policy = {}

            #for act_index, action in enumerate(list(self.obs_actions[:,t])):  
            for j in range(len(self.obs_actions)):
            
                action = self.obs_actions[j][t]
                if type(action) == list or type(action) == type(np.array([0,0])):
                    action = tuple(self.obs_actions[j][t])
                    print('Action got tuplized')
                else:
                    action = action
                
                if action in rewards_dict.keys():
                    rewards_dict[action].append(self.obs_rewards[j][t])
                    states_dict[action].append(self.obs_states[j][t])
                
                else:
                    rewards_dict[action] = [self.obs_rewards[j][t]]             
                    states_dict[action] = [self.obs_states[j][t]]               
            
            ic(rewards_dict[action])
            ic(states_dict[action])
            rewards_per_time.insert(0, rewards_dict[action])                    
            states_per_time.insert(0, states_dict[action])                              
            
            for action in rewards_dict.keys():                                 
                                                                               
                if type(action) == 'numpy.ndarray' or type(action) == 'list':  
                    action = tuple(action)
                    print(f'Action is {action}')                              
                
                ic(action)
                
                X = np.array(states_dict[action])
                ic(X)                           
                
                K = IDTR.gaussian_kernel(X,X, gamma = self.gammas[t])          
                ic(K)
                
                if K.shape != (len(states_dict[action]), len(states_dict[action])):
                    raise ValueError(f'Shape of Kernel is not ({len(states_dict[action])}, {len(states_dict[action])})')
                
                n = len(rewards_dict[action])
                ic(n)
                
                y  = np.array(rewards_dict[action])                               
                ic(y)
                
                
                Q_value = np.array([ Q_values_per_time[-1](states_dict[action][i],  pi_hat_per_time[-1](states_dict[action][i])) for i in range(len(states_dict[action]))])
                if len(Q_value.shape) == 1:
                    Q_value = np.expand_dims(Q_value, axis=-1)
                
                if Q_value.shape != (len(states_dict[action]), 1):
                    raise ValueError(f' Shape of Q value is {Q_value.shape} != ({len(states_dict[action])},1) ')
                
                if len(Q_value.shape) == 1:
                    Q_value = np.expand_dims(Q_value, axis=-1)
                
                if len(y.shape) == 1:
                    y = np.expand_dims(y, axis=-1)
                
                if y.shape != (len(states_dict[action]), 1):
                    raise ValueError(f' Shape of y is {y.shape} != ({len(states_dict[action])},1) ')
                
                
                
                betas[(t, action)] = np.linalg.inv(K + n * self.lambdas[t] * np.eye(K.shape[0])) @ (y + Q_value)
                beta = betas[(t, action)]
                ic(Q_value)
                
                
                def Q_values_functor(x, action=action, beta=beta):
                    ic(len(states_dict[action]))
                    ic(len(beta))
                    ic(action)
                    ic(beta)
                    total = 0
                    for i,s in enumerate(states_dict[action]):
                        ic(beta[i])
                        ic(x)
                        ic(i)
                        ic(s)
                        ic(IDTR.gaussian_kernel(x, s))
                        ic(beta)
                        ic(beta.shape)
                        
                        total = total + np.squeeze(IDTR.gaussian_kernel(x, s)) * np.squeeze(beta[i])
                    
                    return total
                
                
                Q_values[action] = lambda x : Q_values_functor(x, action = action, beta = betas[(t,action)])
                
                ic('States we scan through are')
                ic(states_dict[action])
                ic(len(states_dict[action]))
            '''
            print(states_dict.keys())
            for a in self.action_spaces[t]:
                if a not in list(states_dict.keys()):
                    Q_values[a] = lambda x : 0.0
            '''
            Q_values_per_time.insert(0, Q_values)                               
                                                                                
            remaining_space = self.state_spaces[t]                              
            
            ic(remaining_space)
            
            for l in range(self.max_lengths[t] - 1):                                
                                                                                
                max_score_dbu_action = -np.inf                                            
                for cond_index, cond_DBU in enumerate(condition_DBUs):     
                    
                    for act_index, action in enumerate(self.action_spaces[t]):                  
                        score_dbu_action = 0.0
                        
                        if type(action) == type([0,0]) or type(action) == type(np.array([0,0])):
                            action = tuple(action)                        
                        
                        for i in range(len(self.obs_states)):               
                            
                            def pi_hat(x, t=t):
                                max_val = -np.inf
                                best_action = None
                                for a in self.action_spaces[t]:
                                    value = Q_values[action](x)
                                    ic(value)
                                    
                                    if value > max_val:
                                        max_val = value
                                        best_action = a
                                return best_action
                            
                            ic(self.obs_states[i][t])
                            ic(action)
                            ic(Q_values.keys())
                            pi_action = (pi_hat(self.obs_states[i][t], t))
                            ic(pi_action)
                            if type(pi_action) == list or type(pi_action) == type(np.array([0,1])):
                                pi_action = tuple(pi_action)
                            
                            
                            q_score = 1 / n * (Q_values[action](self.obs_states[i][t]) *
                                               remaining_space.is_point_in_DBU(self.obs_states[i][t])*cond_DBU.is_point_in_DBU(self.obs_states[i][t]) + 
                                               remaining_space.is_point_in_DBU(self.obs_states[i][t]) * (1 - cond_DBU.is_point_in_DBU(self.obs_states[i][t])) * 
                                               Q_values[pi_action](self.obs_states[i][t])) 
                            
                            zeta_score = 1/n * self.zetas[t] * remaining_space.is_point_in_DBU(self.obs_states[i][t])*cond_DBU.is_point_in_DBU(self.obs_states[i][t])
                    
                            
                            ic(q_score)
                            ic(zeta_score)                
                            
                            complexity_score = self.rhos[t] * (self.complexities[t] - cond_DBU.complexity)
                            
                            ic(complexity_score)
                            
                            score_dbu_action += q_score + zeta_score + complexity_score
                            ic(score_dbu_action)
                            
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
            for a in self.action_spaces[t]:
                
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
    


#%%
a = (0,1)
states_dict = {0 : 'a', 1: 'b'}
print( a in (states_dict.keys()))
