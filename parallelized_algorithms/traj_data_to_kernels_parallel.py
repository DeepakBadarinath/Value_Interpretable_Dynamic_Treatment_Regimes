#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Mar 19 13:22:40 2025

@author: badarinath

'''
#%%

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

import disjoint_box_union_parallel
from disjoint_box_union_parallel import DisjointBoxUnion as DBU

import markov_decision_processes
from markov_decision_processes import MarkovDecisionProcess as MDP


#%%

def process_single_traj(traj, instance):
    """
    Helper function to process one trajectory.
    Calls instance.update_counts for each (s_t, a_t, r_t, s_tp1) in the trajectory.
    """
    for timestep, (s_t, a_t, r_t, s_tp1) in enumerate(traj):
        instance.update_counts(s_t, a_t, r_t, s_tp1, timestep)


def compute_single_transition(args):
    (s_bin, a), next_state_counts = args
    total = sum(next_state_counts.values())
    prob_dict = {
        s_next_bin: count / total
        for s_next_bin, count in next_state_counts.items()
    }
    return (s_bin, a), prob_dict


#%%

class MDPModelEstimator:
    """
    Class for estimating the transition probabilities and rewards of a finite-horizon 
    Markov Decision Process (MDP) using count-based Maximum Likelihood Estimation (MLE) 
    from trajectory data.

    Supports continuous state spaces by binning states based on user-specified 
    discretization (diff_vals) at each timestep.
    """

    def __init__(self, diff_vals_per_timestep):
        """
        Initialize the estimator.

        Parameters:
        -----------------------------------------------------------------------
        diff_vals_per_timestep : list[np.ndarray]
            A list of length T (time horizon), where each element is a 1D numpy array 
            specifying the bin widths (diff_vals) for each dimension of the continuous 
            state space at that timestep.
        """
        self.diff_vals_per_timestep = diff_vals_per_timestep
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # ((s_bin, a) -> {s'_bin -> count})
        self.reward_sums = defaultdict(float)  # (s_bin, a) -> sum of rewards
        self.sa_counts = defaultdict(int)      # (s_bin, a) -> count of (s,a) pairs
        self.state_bounds = []                 # Per timestep, stores (min_state, max_state) for state space initialization

    def bin_state(self, state, timestep):
        """
        Bins a continuous state into discrete bins based on diff_vals for the timestep.

        Parameters:
        -----------------------------------------------------------------------
        state : np.ndarray
            The continuous state vector at timestep t.
        timestep : int
            The time index.

        Returns:
        -----------------------------------------------------------------------
        tuple
            The binned state represented as a tuple of integers (multi-dimensional bin index).
        """
        
        diff_vals = self.diff_vals_per_timestep[timestep]
        binned = tuple(np.floor(state / diff_vals).astype(int))
        return binned

    def update_state_bounds(self, s_t, timestep):
        """
        Updates the min and max bounds for the state space at a given timestep.

        Parameters:
        -----------------------------------------------------------------------
        s_t : np.ndarray
            The continuous state vector.
        timestep : int
            The time index.
        """
        if len(self.state_bounds) <= timestep:
            self.state_bounds.append((np.copy(s_t), np.copy(s_t)))
        else:
            min_s, max_s = self.state_bounds[timestep]
            self.state_bounds[timestep] = (np.minimum(min_s, s_t), np.maximum(max_s, s_t))

    def update_counts(self, s_t, a_t, r_t, s_tp1, timestep):
        """
        Updates counts and rewards from a single transition sample.

        Parameters:
        -----------------------------------------------------------------------
        s_t : np.ndarray
            Current continuous state at time t.
        a_t : hashable
            Action taken at time t.
        r_t : float
            Reward observed after taking action a_t.
        s_tp1 : np.ndarray
            Next continuous state at time t+1.
        timestep : int
            The time index t (0-based).
        """
        # Update bounds for state space construction
        self.update_state_bounds(s_t, timestep)
        self.update_state_bounds(s_tp1, timestep + 1)

        # Bin current and next states
        s_bin = self.bin_state(s_t, timestep)
        s_next_bin = self.bin_state(s_tp1, timestep + 1)

        # Update counts for transition and reward estimation
        self.transition_counts[(s_bin, a_t)][s_next_bin] += 1
        self.reward_sums[(s_bin, a_t)] += r_t
        self.sa_counts[(s_bin, a_t)] += 1

    
    def process_trajectories(self, trajectories, n_jobs=4):
        """
        Processes multiple trajectories in parallel using threading.
        """
        print('Trajectories are given by:')
        print(trajectories)
    
        with ThreadPool(n_jobs) as pool:
            list(tqdm(pool.imap(lambda traj: process_single_traj(traj, self), trajectories),
                      total=len(trajectories), desc="Processing Trajectories"))

    def compute_transition_probabilities(self, n_jobs=4):
        """
        Computes empirical transition probabilities P(s' | s, a) using MLE.
        Parallelized over (s_bin, a) pairs.
    
        Returns:
        -----------------------------------------------------------------------
        dict
            Nested dictionary {(s_bin, a): {s_next_bin: probability}}
        """
        transition_probs = {}
        items = list(self.transition_counts.items())
    
        with ThreadPool(n_jobs) as pool:
            results = list(tqdm(pool.imap(compute_single_transition, items),
                                total=len(items), desc="Computing Transition Probs"))
    
        # Collect results
        for (s_bin, a), prob_dict in results:
            transition_probs[(s_bin, a)] = prob_dict
    
        return transition_probs


    def compute_expected_rewards(self):
        """
        Computes the expected reward function R(s, a) = E[reward | s, a] using MLE.

        Returns:
        -----------------------------------------------------------------------
        dict
            Dictionary {(s_bin, a): expected_reward} with average rewards for each (s, a) pair.
        """
        expected_rewards = {}
        for sa_pair in self.sa_counts:
            expected_rewards[sa_pair] = self.reward_sums[sa_pair] / self.sa_counts[sa_pair]
        return expected_rewards
    
    
    def build_mdp_from_estimator(self, gamma, state_spaces=None):
        """
        Constructs the MDP object from the collected counts and reward estimates 
        in the MDPModelEstimator. Optionally allows user-provided state_spaces.
    
        Parameters:
        -----------------------------------------------------------------------
        gamma : float
            The discount factor for the MDP.
    
        state_spaces : list of DBU instances, optional
            If provided, this is used as the state space at each timestep.
            Otherwise, state spaces are constructed from observed min-max state bounds.
    
        Returns:
        -----------------------------------------------------------------------
        MDP : MarkovDecisionProcess
            An instance of the MDP class populated with:
            - Dimensions of the state space at each timestep
            - State spaces (either user-provided or estimated from data)
            - Action spaces observed
            - Transition kernel functions estimated from data
            - Reward functions estimated from data
        """
        T = len(self.diff_vals_per_timestep)
        dimensions = [len(diff) for diff in self.diff_vals_per_timestep]
    
        # If user provided state_spaces, use them. Otherwise build from state_bounds
        if state_spaces is None:
            state_spaces = []
            for min_s, max_s in self.state_bounds:
                box = np.stack([min_s, max_s], axis=1)  # shape (dim, 2)
                lengths = np.array([r[1] - r[0] for r in box])
                centres = np.array([(r[1] + r[0]) / 2 for r in box])
    
                state_spaces.append(DBU(
                    no_of_boxes=1,
                    dimension=box.shape[0],
                    lengths=lengths,
                    centres=centres
                ))
    
        # Extract action space (unique actions observed in the data)
        actions = set(a for (s_bin, a) in self.sa_counts.keys())
        action_spaces = [list(actions) for _ in range(T)]
    
        # Compute transitions and rewards
        transitions = self.compute_transition_probabilities()
        rewards = self.compute_expected_rewards()
    
        # Build the transition kernels and reward functions
        transition_kernels = []
        reward_functions = []
    
        for t in range(T):
            def make_transition_kernel(timestep):
                def kernel(s_next_bin, s_bin, a, state_space=None, action_space=None):
                    return transitions.get((s_bin, a), {}).get(s_next_bin, 0.0)
                return kernel
    
            def make_reward_function(timestep):
                def reward(s_bin, a, state_space=None, action_space=None):
                    return rewards.get((s_bin, a), 0.0)
                return reward
    
            transition_kernels.append(make_transition_kernel(t))
            reward_functions.append(make_reward_function(t))
    
        # Build and return the MDP object
        return MDP(dimensions, state_spaces, action_spaces, T, gamma, transition_kernels, reward_functions)

#%%

diff_vals = [np.array([0.1, 0.2]), np.array([0.05]), np.array([0.1])]
gamma = 0.95
estimator = MDPModelEstimator(diff_vals)

# Example batch of 2 trajectories
trajectories = [
    [ (np.array([0.15, 0.45]), 'a1', 1.0, np.array([0.23])),
      (np.array([0.23]), 'a2', 0.5, np.array([0.29])) ],
      
    [ (np.array([0.25, 0.55]), 'a1', 0.8, np.array([0.33])),
      (np.array([0.33]), 'a2', 0.3, np.array([0.39])) ]
]

estimator.process_trajectories(trajectories)
mdp = estimator.build_mdp_from_estimator(gamma)

print("MDP Dimensions:", mdp.dimensions)
print("MDP Action Spaces:", mdp.action_spaces)
print("MDP State Spaces (DBU):", mdp.state_spaces)

#%%
