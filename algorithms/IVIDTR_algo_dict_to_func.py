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
import icecream
from icecream import ic

import markov_decision_processes as mdp_module
import disjoint_box_union                                                      
from disjoint_box_union import DisjointBoxUnion as DBU                         
import constraint_conditions as cc                                              

import itertools
import sys
sys.setrecursionlimit(9000)  # Set a higher limit if needed
                                                                                            
                                                                                
ic.disable()
#%%

mdp_module = reload(mdp_module)
disjoint_box_union = reload(disjoint_box_union)
cc = reload(cc)


#%%
class VIDTR:
    
    def __init__(self, MDP, max_lengths,
                 etas, rhos, max_complexity,
                 stepsizes, max_conditions = math.inf):                        
        '''
        Value-based interpretable dynamic treatment regimes; Generate a tree based
        policy while solving a regularized interpretable form of the Bellmann 
        equation. In this module we assume that the state spaces are time dependent.
        
        Parameters:
        -----------------------------------------------------------------------
        MDP : MarkovDecisionProcess
              The underlying MDP from where we want to get interpretable policies
              
        max_lengths : list[T] or int
                      The max depth of the tree upto the T timesteps
        
        etas : list[T] or int
              Volume promotion constants
              Higher this value, greater promotion in the splitting process    
                                                                               
        rhos : list[T] or int                                                            
              Complexity promotion constants                                    
              Higher this value, greater the penalization effect of the complexity 
              splitting process                                                
                                                                               
        max_complexity : int or list                                                   
                         The maximum complexity of the conditions; maximum number of 
                         and conditions present in any condition               
                                                                               
        stepsizes : list[np.array((1, MDP.states.dimension[t])) for t in range(time_horizon)] or float or int        
                    The stepsizes when we have to integrate over the DBU       
        
        max_conditions : int or list                                           
                         The maximum number of conditions per time and lengthstep
                         If None then all the conditions will be looked at     
        '''
        
        self.MDP = MDP
        self.time_horizon = self.MDP.time_horizon
        
        if type(max_lengths) == float or type(max_lengths) == int:
            max_lengths = [max_lengths for t in range(self.MDP.time_horizon)]
        
        self.max_lengths = max_lengths
        
        if type(etas) == float or type(etas) == int:
            etas = [etas for t in range(self.MDP.time_horizon)]
        
        self.etas = etas
        
        if type(rhos) == float or type(rhos) == int:
            rhos = [rhos for t in range(self.MDP.time_horizon)]

        self.rhos = rhos
        
        if type(stepsizes) == float or type(stepsizes) == int:
            stepsizes = [np.ones((1, MDP.state_spaces[t].dimension)) for t in range(self.time_horizon)]
        
        self.stepsizes = stepsizes
        
        if type(max_complexity) == int:
            max_complexity = [max_complexity for t in range(self.MDP.time_horizon)]
        
        self.max_complexity = max_complexity
        
        self.true_values = [lambda s: 0 for t in range(self.MDP.time_horizon+1)]
        
        if type(max_conditions) == int:
            max_conditions = [max_conditions for t in range(max_conditions)]
        
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
            
            for a in self.MDP.action_spaces[t]:
                if function(np.array(s),a) > max_val:
                    max_val = function(s,a)
            
            return max_val
                    
        return max_function


    def bellman_equation(self, t):
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
            
            space = self.MDP.state_spaces[t]                                   
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            return self.MDP.reward_functions[t](np.array(s), a, space, action_space) + self.MDP.gamma * (
                    np.sum([self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space) * self.true_values[t+1](np.array(s_new)) 
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
        
    @staticmethod
    def convert_function_to_dict_s_a(f, S):
        '''
        Given a function f : S \times A \to \mathbb{R}                         
        Redefine it such that f is now represented by a dictonary              

        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            The function that is to be redefined to give a dictonary           
            
        S : iterable version of the state space                                
            iter(DisjointBoxUnionIterator)                                     

        Returns:
        -----------------------------------------------------------------------
        f_dict : dictionary
                The function which is now redefined to be a dictonary

        '''
        f_dict = {}
        for s in S:
            f_dict[tuple(s)] = f(s)
        
        return f_dict
    
    @staticmethod
    def convert_dict_to_function(f_dict, S, default_value=0):
        '''
            
        Given a dictonary f_dict, redefine it such that we get a function f from 
        S to A

        Parameters:
        -----------------------------------------------------------------------
        f_dict : dictionary
                 The dictonary form of the function
                 
        S : iterable version of the state space
            iter(DisjointBoxUnionIterator)

        Returns:
        -----------------------------------------------------------------------
        f : func
            The function version of the dictonary

        '''
            
        def f(s):
            
            if tuple(s) in f_dict.keys():
                
                return f_dict[tuple(s)]
            
            else:
                return default_value
        
        return f
    
    
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
        #zero_value = lambda s : 0
        zero_value_dicts = []
        const_action_dicts = []
        
        for t in range(self.time_horizon):
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            zero_dict = {}
            const_action_dict = {}
            
            for s in state_iterator:
                zero_dict[tuple(s)] = 0
                const_action_dict[tuple(s)] = self.MDP.action_spaces[t][0]
            
            zero_value_dicts.append(zero_dict)
            const_action_dicts.append(const_action_dict)
        
        
        optimal_policy_dicts = const_action_dicts
        
        optimal_value_dicts = zero_value_dicts
        
        #Start from T-1 or T-2?                                                                            
        for t in np.arange(self.time_horizon-1, -1, -1):                    
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)

            for s in state_iterator:
                max_val = -np.inf
                                
                for a in self.MDP.action_spaces[t]:
                                                    
                    bellman_value = self.bellman_equation(t)(s,a)
                    #ic(s)
                    #ic(a)
                    #ic(bellman_value)
                    
                    if bellman_value > max_val:                                 
                        
                        max_val = bellman_value
                        best_action = a
                                                                                
                        optimal_policy_dicts[t][tuple(s)] = best_action                       
                        optimal_value_dicts[t][tuple(s)] = max_val                        

                
                #ic(t)
                #ic(s)
                #ic(best_action)
        
        
        optimal_policy_funcs = []
        optimal_value_funcs = []
        
        for t in range(self.MDP.time_horizon):
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            optimal_policy_funcs.append(VIDTR.convert_dict_to_function(optimal_policy_dicts[t],
                                                                       state_iterator,
                                                                       default_value=self.MDP.action_spaces[t][0]))
            
            optimal_value_funcs.append(VIDTR.convert_dict_to_function(optimal_value_dicts[t],
                                                                      state_iterator,
                                                                      default_value=0.0))
        
        self.optimal_policy_funcs = optimal_policy_funcs
        self.optimal_value_funcs = optimal_value_funcs
                                                                               
        return optimal_policy_funcs, optimal_value_funcs                                
    
    
    def constant_eta_function(self, t):                                         
        '''
        Return the constant \eta function for time t

        Parameters:
        -----------------------------------------------------------------------
        t : int
            Time step

        Returns:
        -----------------------------------------------------------------------
        f : function
            Constant eta function at time t                                    
                                                                                 
        '''                                                                                               
        f = lambda s,a: self.etas[t]
        return f
    
    def compute_interpretable_policies(self, cond_stepsizes=0.0,
                                       integration_method = DBU.sampling_integrate_static,
                                       integral_percent = 0.5):                                                               
        
        '''                                                                    
        Compute the interpretable policies for the different length and        
        timesteps given a DBUMarkovDecisionProcess.                             
        
        Parameters:                                                            
        -----------------------------------------------------------------------
        cond_stepsizes : int or float or np.array(self.DBU.dimension)               
                         The stepsizes over the conditional DBU                     
        
        integration_method : function
                             The method of integration we wish to use
        
        integral_percent : float
                           The percent of points we wish to sample from
        
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

                
        for t in np.arange(self.MDP.time_horizon-1, -1, -1):
            
            all_conditions = []
            all_condition_DBUs = []
            
            if self.MDP.state_spaces[t].no_of_boxes == 0:
                print('Zero state space found')
            
            all_conditions = self.MDP.state_spaces[t].generate_all_conditions(self.max_complexity[t])
            all_condition_DBUs = []
            
            state_bounds = self.MDP.state_spaces[t].get_total_bounds()
            ic(state_bounds)
            
            remaining_space = self.MDP.state_spaces[t]
            
            total_error = 0
            total_bellman_error = 0
            
            all_condition_dicts = {}
            
            for i,c in enumerate(all_conditions):
                if c != None:
                    con_DBU = DBU.condition_to_DBU(c, self.stepsizes[t])
                    if con_DBU.no_of_boxes != 0:
                        all_condition_DBUs.append(con_DBU)
                        necc_tuple = con_DBU.dbu_to_tuple()
                        if necc_tuple not in all_condition_dicts:
                            all_condition_dicts[necc_tuple] = 1
            		
            optimal_errors.append([])
            optimal_bellman_errors.append([])
            stored_DBUs.append([])
            optimal_conditions.append([])
            optimal_actions.append([])
            
            maxim_bellman_error_dict = {}
            fixed_bellman_error_dict = {}
            
            condition_DBUs = all_condition_DBUs
            conditions = all_conditions
                
                #print('We are at time {t}')
                #print('-----------------------------------------------------------')
            
            for l in range(self.max_lengths[t]-1):
                
                min_error = np.inf
                optimal_condition = None
                optimal_action = None
                no_of_null_DBUs = 0
                
                print(f'We are at timestep {t} and lengthstep {l}')
                
                max_conditions = min(self.max_conditions, len(condition_DBUs)) 
                
                
                for i in range(max_conditions):                                                                  
                    
                    cond_DBU = condition_DBUs[i]                                
                    
                    #print('We iterate over the following cond dbu')				             
                    #print(cond_DBU)                                             	
                    
                    error = 0.0
                    
                    if cond_DBU.dbu_to_tuple() not in maxim_bellman_error_dict:
                        
                        maxim_bellman_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s)
                        
                        if integration_method == DBU.sampling_integrate_static:
                            maxim_bellman_error = integration_method(cond_DBU,
                                                                     maxim_bellman_function,
                                                                     integral_percent)
                            
                        else:
                            maxim_bellman_error = integration_method(cond_DBU,
                                                                     maxim_bellman_function)
                            
                        maxim_bellman_error_dict[cond_DBU.dbu_to_tuple()] = maxim_bellman_error
                                                                                
                    else:
                        maxim_bellman_error = maxim_bellman_error_dict[cond_DBU.dbu_to_tuple()]
                    
                    for a in self.MDP.action_spaces[t]:                         
                        
                        if type(a) == type([0,1]) or type(a) == type(np.array([0,0])):
                            a = list(a)                                                                                         
                            key = tuple(list(cond_DBU.dbu_to_tuple()) + a)      
                            #ic(key)
                        else:
                            key = tuple(list(cond_DBU.dbu_to_tuple()) + a)
                        
                        if key not in fixed_bellman_error_dict:
                            fixed_bellman_function = lambda s: -VIDTR.fix_a(self.bellman_equation(t), a=a)(s)            
                            #ic(a)
                            
                            if integration_method == DBU.sampling_integrate_static:
                                fixed_bellman_error = integration_method(cond_DBU,
                                                                         fixed_bellman_function,
                                                                         integral_percent)
                                
                            else:
                                fixed_bellman_error = integration_method(cond_DBU,
                                                                         fixed_bellman_function)
                            
                            fixed_bellman_error = integration_method(cond_DBU, fixed_bellman_function)
                            fixed_bellman_error_dict[key] = fixed_bellman_error
                        
                        else:
                            fixed_bellman_error = fixed_bellman_error_dict[key]
                        
                        bellman_error = maxim_bellman_error + fixed_bellman_error
                        #ic(bellman_error)
                        
                        #print(f'Etas at the {t}th timestep is {self.etas[t]}')                                        
                        const_error = -cond_DBU.no_of_points() * self.etas[t]   
                        #ic(const_error)                                         
                        complexity_error = self.rhos[t] * cond_DBU.complexity
                        #ic(complexity_error)
                                                                                
                        error += bellman_error + const_error + complexity_error
                        #ic(error)                                               
                                                                                    
                        if error < min_error:                                       
                            optimal_condition = conditions[i]                   
                            optimal_cond_DBU = cond_DBU                        
                            optimal_action = a                                      
                            min_error = error                                  
                            optimal_bellman_error = bellman_error              
            
                    new_condition_DBUs = []                                         
                    new_condition_dicts = {}                                     
                
                #print('Yo')
                #print(f'We subtract {optimal_cond_DBU} from {remaining_space}')
                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                #print('Subtraction done')
                total_error += min_error
                total_bellman_error += optimal_bellman_error
                
                for i, cond_dbu in enumerate(condition_DBUs):                       
                    sub_DBU = cond_dbu.subtract_DBUs(optimal_cond_DBU)          
                    #print(f'We subtract {optimal_cond_DBU} from {cond_dbu}')
                    
                    necc_tuple = sub_DBU.dbu_to_tuple()
                    if sub_DBU.no_of_boxes == 0:
                       	no_of_null_DBUs = no_of_null_DBUs + 1
                    elif necc_tuple not in new_condition_dicts:         
                        new_condition_dicts[necc_tuple] = 1             
                        new_condition_DBUs.append(sub_DBU)	         
				
                      
                print(f'Timestep {t} and lengthstep {l}:')                      
                print('----------------------------------------------------------------')
                print(f'Optimal condition at timestep {t} and lengthstep {l} is {optimal_condition}')
                print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
                print(f'Optimal conditional DBU at timestep {t} and lengthstep {l} is {optimal_cond_DBU}')
                print(f'Optimal error is {min_error}')                          
                print(f'Non null DBUs = {len(condition_DBUs)} - {no_of_null_DBUs}')
                print(f'Eta is {self.etas[t]}, Rho is {self.rhos[t]}')

                
                all_condition_dicts = new_condition_dicts                                                        
                condition_DBUs = new_condition_DBUs                            
                optimal_errors[-1].append(min_error)                            
                stored_DBUs[-1].append(optimal_cond_DBU)                        
                optimal_conditions[-1].append(optimal_condition)                
                optimal_actions[-1].append(optimal_action)                      
                optimal_bellman_errors[-1].append(optimal_bellman_error)        
        
                if len(condition_DBUs) == 0:                                    
                    print('--------------------------------------------------------------')
                    print(f'For timestep {t} we end at lengthstep {l}')
                    if l != self.max_lengths[t] - 2:
                        print('Early stopping detected')
                    break                                                      
        
                                                                                    
            #Final lengthstep - We can only choose the optimal action here and we work over S - \cap_{i=1}^K S_i
            min_error = np.inf
            for a in self.MDP.action_spaces[t]:
                integrating_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s) -VIDTR.fix_a(self.bellman_equation(t), a=a)(s) - self.constant_eta_function(t)(s,a)
                #ic(remaining_space)
                #ic(integrating_function)
                
                if integration_method == DBU.sampling_integrate_static:
                    error = integration_method(remaining_space,
                                               integrating_function,
                                               integral_percent)
                    
                else:
                    error = integration_method(remaining_space,
                                               integrating_function)
                
            if error<min_error:
                optimal_action = a
                min_error = error
    
            total_error += min_error
            total_bellman_error += min_error
            print(f'Total Bellman Error is {total_bellman_error}')
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
        
        #print(optimal_conditions, optimal_actions)
        
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
    
    @staticmethod
    def tuplify_2D_array(two_d_array):
        two_d_list = []
        n,m = two_d_array.shape
        
        for i in range(n):
            two_d_list.append([])
            for j in range(m):
                two_d_list[-1].append(two_d_array[i,j])
        
            two_d_list[-1] = tuple(two_d_array[-1])                                 
        two_d_tuple = tuple(two_d_list)                                             
        return two_d_tuple                                                     
        
    
    def plot_errors(self):
        '''
        Plot the errors obtained after we perform the VIDTR algorithm.          

        '''
        for t in range(len(self.optimal_errors)):                                  
            plt.plot(np.arange(len(self.optimal_errors[t])), self.optimal_errors[t])
            plt.title(f'Errors at time {t}')
            plt.xlabel('Lengths')
            plt.ylabel('Errors')
            plt.show()
            
            plt.plot(np.arange(len(self.optimal_bellman_errors[t])), self.optimal_bellman_errors[t])
            plt.title(f'Bellman Errors at time {t}')
            plt.xlabel('Time')                                                 
            plt.ylabel('Bellman Error')                                         
            plt.show()


#%%

space = DBU(1, 2, np.array([2,3]), np.array([0,0]))
S = disjoint_box_union.DBUIterator(space)
S = iter(S)

f = lambda s : 0.0

f_dict = VIDTR.convert_function_to_dict_s_a(f, S)

for k in f_dict.keys():
    print(f'Key {k}: Value {f_dict[k]}')


g = VIDTR.convert_dict_to_function(f_dict, S)
for k in f_dict.keys():
    print(f'Key {k}: Value {g(k)}')

