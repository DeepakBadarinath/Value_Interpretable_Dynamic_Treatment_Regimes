#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:29:39 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import disjoint_box_union
import constraint_conditions as cc
import experiments_VIDTR as exp_VIDTR
import IDTR_implementation
from importlib import reload

disjoint_box_union = reload(disjoint_box_union)
from disjoint_box_union import DisjointBoxUnion as DBU

cc = reload(cc)
exp_VIDTR = reload(exp_VIDTR)
IDTR_implementation = reload(IDTR_implementation)
from IDTR_implementation import IDTR
                                                                                
#%%                                                                            
                                                                                    
s1 = exp_VIDTR.Scenario1()                                                     
N1 = 100                                                                       

obs_states1 = []
obs_actions1 = []
obs_rewards1 = []

dimensions1 = exp_VIDTR.Scenario1.dimensions
max_lengths = 4
zetas = 0.3
rhos = 0.2
time_horizon = exp_VIDTR.Scenario1.time_horizon
action_spaces = exp_VIDTR.Scenario1.action_spaces
state_spaces = exp_VIDTR.Scenario1.state_spaces
complexities = 4
gamma = 0.9
lambdas = 5.0

for n in range(N1):
    states1 = s1.states
    actions1 = s1.actions
    rewards1 = s1.rewards
    obs_states1.append(states1)
    obs_actions1.append(actions1)
    obs_rewards1.append(rewards1)

print('Observed states are')
print(obs_states1)

print('Observed actions are')
print(obs_actions1)

print('Observed rewards are')
print(obs_rewards1)
#%%

print('Observed Actions are')
print(obs_actions1[0][0])


#%%
algo = IDTR(time_horizon, dimensions1, obs_states1, obs_actions1, obs_rewards1,
            state_spaces, max_lengths, zetas, rhos, gamma, lambdas,
            complexities, action_spaces, stepsizes = 0.1)

optimal_cond_DBUs_per_time, optimal_actions_per_time = algo.compute_interpretable_policies()

#%%

s2 = exp_VIDTR.Scenario2()
N2 = 500

obs_states2 = []
obs_actions2 = []
obs_rewards2 = []

dimensions2 = exp_VIDTR.Scenario2.dimensions
max_lengths = 4
zetas = 0.3
rhos = 0.2
time_horizon = exp_VIDTR.Scenario2.time_horizon
action_spaces = exp_VIDTR.Scenario2.action_spaces
state_spaces = exp_VIDTR.Scenario2.state_spaces
complexities = 4
gamma = 0.9
lambdas = 5.0

for n in range(N2):
    states2 = s2.states
    actions2 = s2.actions
    rewards2 = s2.rewards
    obs_states2.append(states1)
    obs_actions2.append(actions1)                                               
    obs_rewards2.append(rewards1)

print('Observed states are')
print(obs_states2)

print('Observed actions are')
print(obs_actions2)

print('Observed rewards are')
print(obs_rewards2)
#%%

print('Observed Actions are')
print(obs_actions2[0][0])


#%%
algo = IDTR(time_horizon, dimensions2, obs_states2, obs_actions2, obs_rewards2,
            state_spaces, max_lengths, zetas, rhos, gamma, lambdas,
            complexities, action_spaces, stepsizes = 0.1)

optimal_cond_DBUs_per_time, optimal_actions_per_time = algo.compute_interpretable_policies()

#%%

