#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:36:08 2024                                            
                                                                               
@author: badarinath                                                            
"""

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import inspect
import math

import markov_decision_processes as mdp_module
import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU
import constraint_conditions as cc

#%%

mdp_module = reload(mdp_module)
disjoint_box_union = reload(disjoint_box_union)
cc = reload(cc)

#%%                                                                             
class VIDTR:
    
    def __init__(self, MDP, max_lengths,
                 eta, rho, max_complexity,
                 stepsizes, max_conditions = math.inf):
        '''
        Value-based interpretable dynamic treatment regimes; Generate a tree based
        policy while solving a regularized interpretable form of the Bellmann  
        equation.                                                              
        
        Parameters:
        -----------------------------------------------------------------------
        MDP : MarkovDecisionProcess
              The underlying MDP from where we want to get interpretable policies
              
        max_lengths : list[T]
                      The max depth of the tree upto the T timesteps
        
        eta : float
              Volume promotion constant
              Higher this value, greater promotion in the splitting process    
                                                                               
        rho : float                                                            
              Complexity promotion constant                                    
              Higher this value, greater the penalization effect of the complexity 
              splitting process                                                
                                                                               
        max_complexity : int                                                   
                         The maximum complexity of the conditions; maximum number of 
                         and conditions present in any condition               
                                                                               
        stepsizes : np.array((1, MDP.states.dimension)) or float or int        
                    The stepsizes when we have to integrate over the DBU       
        
        max_conditions : int or None                                           
                         The maximum number of conditions per time and lengthstep
                         If None then all the conditions will be looked at     
        '''
        
        self.MDP = MDP
        self.max_lengths = max_lengths
        if type(stepsizes) == float or type(stepsizes) == int:
            stepsizes = np.ones((1, MDP.state_spaces[0].dimension))
            print(stepsizes)
                                                                                       
        self.stepsizes = stepsizes                                             
        self.eta = eta                                                          
        self.rho = rho
        self.max_complexity = max_complexity
        self.true_values = [lambda s: 0 for t in range(self.MDP.time_horizon+1)]
        self.max_conditions = max_conditions
        #print(f'The max conditions is {self.max_conditions}')
        
    def maximum_over_actions(self, function, t):
                                                                                
        '''
        Given a function over states and actions, find the function only over  
        states.
        
        Parameters:
        -----------------------------------------------------------------------
        function : function(s,a)
                   A function over states and actions for which we wish to get 
                   the map s \to max_A f(s,a)

        Returns:
        -----------------------------------------------------------------------
        max_function : function(s)
                       s \to \max_A f(s,a) is the function we wish to get
        
        '''
        def max_function(s):
            
            max_val = -np.inf
            
            #print(f't is {t}, int of t is {int(t)}')
            
            for a in self.MDP.action_spaces[int(t)]:
                if function(np.array(s),a) > max_val:
                    max_val = function(s,a)
            
            return max_val
                    
        return max_function


    def bellman_function(self, t):
        '''
        Return the Bellman equation for the Markov Decision Process.
        
        Assumes we know the true values from t+1 to T.
        
        Parameters:
        -----------------------------------------------------------------------
        t : float                                                              
            The time at which we wish to return the Bellman function for the MDP.

        Returns:
        -----------------------------------------------------------------------
        bellman_function : func
                           The Bellman function of the MDP for the t'th timestep.

        '''
        def bellman_map(s,a):
                                                   
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[int(t)])
            dbu_iterator = iter(dbu_iter_class)
            
            #print(f'State is {s}, action is {a}')
            
            return self.MDP.reward_functions[int(t)](s, a, self.MDP.state_spaces[int(t)], self.MDP.action_spaces[int(t)]) + self.MDP.gamma * (
                    np.sum([self.MDP.transition_kernels[int(t)](s_new, s, a, self.MDP.state_spaces[int(t)], self.MDP.action_spaces[int(t)]) * self.true_values[int(t)+1](s_new) 
                            for s_new in dbu_iterator]))
        
        return bellman_map

    
    @staticmethod
    def fix_a(f, a):
        '''
        Given a function f(s,a), get the function over S by fixing the action
                                                                                                                                                        
        Parameters:                                                                                                                                              
        -----------------------------------------------------------------------
        f : func
            The function we wish to get the projection over
            
        a : type(self.MDP.actions[0])
            The action that is fixed
        '''
        return lambda s : f(s,a)
    
    
    @staticmethod
    def redefine_function(f, s, a):
        '''
        Given a function f, redefine it such that f(s) is now a

        Parameters:
        -----------------------------------------------------------------------
        f : function
            Old function we wish to redefine
        s : type(domain(function))
            The point at which we wish to redefine f
        a : type(range(function))
            The value taken by f at s

        Returns:
        -----------------------------------------------------------------------
        g : function
            Redefined function

        '''
        def g(state):
            if np.sum((np.array(state)-np.array(s))**2) == 0:
                return a
            else:
                return f(state)
        return g
        
    
    def compute_optimal_policies(self):
        '''
        Compute the true value functions at the different timesteps.
        
        Stores:
        -----------------------------------------------------------------------
        optimal_values : list[function]
                         A list of length self.MDP.time_horizon which represents the 
                         true value functions at the different timesteps
        
        optimal_policies : list[function]
                         The list of optimal policies for the different timesteps 
                         for the MDP
        '''
        zero_value = lambda s : 0
        const_policy = lambda s: self.MDP.action_spaces[0][0]
        
        optimal_policies = [const_policy for t in range(self.MDP.time_horizon)]
        optimal_values = [zero_value for t in range(self.MDP.time_horizon)]
    
        for t in np.arange(self.MDP.time_horizon-1, -1, -1):
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)

            for s in state_iterator:
                print(f's is {s}')
                max_val = -np.inf
                for a in self.MDP.action_spaces[t]:

                    bellman_value = self.bellman_function(t)(s,a)

                    if bellman_value > max_val:
                        
                        max_val = bellman_value
                        best_action = a
                        optimal_policies[t] = VIDTR.redefine_function(optimal_policies[t], s, best_action)
                        optimal_values[t] = VIDTR.redefine_function(optimal_values[t], s, max_val)

                
                print(f'Time : {t}, State : {s}, Best Action : {best_action}') 
            
        self.optimal_policies = optimal_policies
        self.optimal_values = optimal_values
                                                                               
        return optimal_policies, optimal_values                                
    
    
    def constant_eta_function(self):
        
        f = lambda s,a: self.eta
        return f
    
    def compute_interpretable_policies(self):                                                               
        
        '''
        Compute the interpretable policies for the different length and        
        timesteps given a DBUMarkovDecisionProcess.                             
        
        Parameters:                                                            
        -----------------------------------------------------------------------
        stepsizes : int or float or np.array(self.DBU.dimension)               
                    The stepsizes over the conditional DBU                     
        
        Stores:
        -----------------------------------------------------------------------
        optimal_conditions :  list[list]
                              condition_space[t][l] gives the optimal condition at
                              timestep t and lengthstep l
        
        optimal_errors : list[list]
                         errors[t][l] represents the error obtained at timestep t and 
                         length step l
        
        optimal_actions : list[list]
                          optimal_intepretable_policies[t][l] denotes
                          the optimal interpretable policy obtained at 
                          time t and length l
        
        stored_DBUs :   list[list]
                        stored_DBUs[t][l] is the DBU stored at timestep t and 
                        lengthstep l for the final policy
        
        stepsizes :  np.array(self.DBU.dimension) or int or float
                     The length of the stepsizes in the different dimensions
        
        total_error : float
                      The total error in the splitting procedure
        
        total_bellman_error : float
                              The total Bellman error resulting out of the splitting 
                    
        Returns:
        -----------------------------------------------------------------------
        int_policies : list[list[function]]
                       optimal_intepretable_policies[t][l] denotes
                       the optimal interpretable policy obtained at 
                       time t and length l

        '''
        
        optimal_errors = []
        stored_DBUs = []
        optimal_conditions = []
        optimal_actions = []
        optimal_bellman_errors = []

        all_conditions = []
        all_condition_DBUs = []
        
        for t in np.arange(self.MDP.time_horizon-1, -1, -1):
            
            cond_iter_class = disjoint_box_union.ConditionsIterator(self.MDP.state_spaces[t],
                                                                    self.max_complexity)
            cond_iterator = iter(cond_iter_class)
            state_bounds = self.MDP.state_spaces[t].get_total_bounds()
            
            remaining_space = self.MDP.state_spaces[t]
            
            total_error = 0
            total_bellman_error = 0
            
            for i,c in enumerate(cond_iterator):
                all_conditions.append(c)
                c.state_bounds = state_bounds                                  
                con_DBU = DBU.condition_to_DBU(c, self.stepsizes)                        
                if con_DBU.no_of_boxes != 0:                                   
                    all_condition_DBUs.append(con_DBU)                         
            
            optimal_errors.append([])
            optimal_bellman_errors.append([])
            stored_DBUs.append([])
            optimal_conditions.append([])
            optimal_actions.append([])

            condition_DBUs = all_condition_DBUs
            conditions = all_conditions
            
            #print('We are at time {t}')
            #print('-----------------------------------------------------------')
            
            for l in range(self.max_lengths[t]-1):
                
                min_error = np.inf
                optimal_condition = None
                optimal_action = None
                no_of_null_DBUs = 0
                
                max_conditions = min(self.max_conditions, len(condition_DBUs))
                
                for i in range(max_conditions):
                    
                    cond_DBU = condition_DBUs[i]
                                                                                
                    error = 0                                                   
                    for a in self.MDP.action_spaces[t]:                                      
                        
                        #print(f'The action we have is : {a}')
                        bellman_function = lambda s: self.maximum_over_actions(self.bellman_function(t), t)(s) -VIDTR.fix_a(self.bellman_function(t), a=a)(s)
                        constant_function = lambda s: -self.constant_eta_function()(s,a)
                                                                                
                        bellman_error = cond_DBU.integrate(bellman_function)    
                        #print(f'Bellman error is {bellman_error
                        const_error = cond_DBU.integrate(constant_function)     
                        #print(f'Constant eta_error is {const_error)
                        complexity_error = self.rho * conditions[i].complexity 
                        #print(f'Complexity error is {complexity_error}')      
                        
                        error += bellman_error + const_error + complexity_error
                        
                        
                        if error < min_error:
                            optimal_condition = conditions[i]
                            optimal_cond_DBU = cond_DBU                        
                            optimal_action = a                                       
                            min_error = error                                  
                            optimal_bellman_error = bellman_error              
                
                new_condition_DBUs = []                                          
                
                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                
                total_error += min_error                                       
                total_bellman_error += optimal_bellman_error                     
                
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
                print(f'Optimal error is {min_error}')                          
                print(f'Non null DBUs = {len(condition_DBUs)} - {no_of_null_DBUs}')
                print(f'Eta is {self.eta}, Rho is {self.rho}')

                                                                                
                condition_DBUs = new_condition_DBUs                            
                optimal_errors[-1].append(min_error)                            
                stored_DBUs[-1].append(optimal_cond_DBU)                        
                optimal_conditions[-1].append(optimal_condition)                
                optimal_actions[-1].append(optimal_action)                      
                optimal_bellman_errors[-1].append(optimal_bellman_error)        
                
                if len(condition_DBUs) == 0:                                    
                    print('----------------------------------------------------')
                    print(f'For timestep {t} we end at lengthstep {l}')         
                    break                                                      
                
            
            #Final lengthstep - We can only choose the optimal action here and we work over S - \cap_{i=1}^K S_i
            min_error = np.inf                                                 
            for a in self.MDP.action_spaces[t]:                                          
                integrating_function = lambda s: self.maximum_over_actions(self.bellman_function(t), t)(s) -VIDTR.fix_a(self.bellman_function(t), a=a)(s) - self.constant_eta_function()(s,a)
                error = remaining_space.integrate(integrating_function)        
                
                
                if error<min_error:                                             
                    optimal_action = a                                         
                    min_error = error                                          
            
            total_error += min_error                                           
            total_bellman_error += min_error                                   
            print('--------------------------------------------------------')  
            print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
            optimal_actions[-1].append(optimal_action)                         
            print(f'Optimal conditional DBU at timestep {t} and lengthstep {l} is {optimal_cond_DBU}')
            stored_DBUs[-1].append(optimal_cond_DBU)                           
            print(f'Optimal error is {min_error}')                             
            optimal_errors[-1].append(min_error)                               
        
        self.optimal_conditions = optimal_conditions                           
        self.optimal_errors = optimal_errors                                   
        self.optimal_bellman_errors = optimal_bellman_errors                    
        self.optimal_actions = optimal_actions                                 
        self.stored_DBUs = stored_DBUs                                         
        self.total_bellman_error = total_bellman_error                         
        self.total_error = total_error                                         
        
        print(optimal_conditions, optimal_actions)                             
        
        return optimal_conditions, optimal_actions                             
    
    @staticmethod
    def get_interpretable_policy(conditions, actions):
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
            
            for i, cond in enumerate(conditions):                                
                                                                                 
                if cond.contains_point(state):                                  
                                                                                
                    return actions[i]                                           
            
                                                                                
            return actions[-1]                                                  
        
        return policy                                                           
    
    
    def plot_errors(self, error_bounds = [], bellman_error_bounds = []):                                                     
        '''
        Plot the errors obtained after we perform the VIDTR algorithm           
                                                                                    
        '''
        for t in range(len(self.max_lengths)):                                  
            plt.plot(np.arange(len(self.optimal_errors[t])), self.optimal_errors[t])
            plt.title(f'Errors at time {t}')                                   
            plt.xlabel('Lengths')                                               
            plt.ylabel('Errors')                                                
            
            if len(error_bounds) > 0:
                plt.plot(np.arange(len(error_bounds[t])), error_bounds[t], label = 'Error bounds')
            
            plt.legend()
            plt.show()     
            
            plt.plot(np.arange(len(self.optimal_bellman_errors[t])), self.optimal_bellman_errors[t])
            plt.title(f'Bellman Errors at time {t}')                           
            plt.xlabel('Time')                                                 
            plt.ylabel('Bellman Error')
            
            if len(bellman_error_bounds) > 0:
                plt.plot(np.arange(len(bellman_error_bounds[t])), bellman_error_bounds[t], label = 'Bellman error bounds')
            
            plt.legend()                                        
            plt.show()                                                           
    