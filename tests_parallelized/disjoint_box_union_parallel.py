#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# To dos are also listed in this .py file
"""                    
Created on Fri May 24 22:29:40 2024
                       
@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import constraint_conditions_parallel as cc                                             
import interval_and_rectangle_operations_parallel as op
from icecream import ic
import random
import time

from importlib import reload
import itertools                                  
from itertools import combinations, product                                    
                                
from scipy import integrate
import multiprocessing as mp

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


op = reload(op)
                               
#%%                                                                                                                                                            
# Task breakdown:-                                                              
###############################################################################

# Baum Welch and other algorithms to get the models for the data - design and debug              
# VIDTR based on the examples from the IDTR paper                                    
                                                                                
# VIDTR based on the real world data                                             
###############################################################################         

#%% 

def compute_points_per_box(args):
    """
    Helper function for parallelizing the point count computation per box.
    """
    box_no, bounds, stepsizes, dimension = args
    lengths = [DBUIterator.count_points(bounds[box_no][d][0],
                                        bounds[box_no][d][1],
                                        stepsizes[box_no, d]) for d in range(dimension)]
    point_per_box = 1
    for l in lengths:
        point_per_box *= l
    return point_per_box

# Move this function to the top level
def compute_box_integral(bounds, stepsizes, dimension, function, box_no):
    coord_list = []
    for dim in range(dimension):
        coord_list.append(np.arange(bounds[box_no][dim][0],
                                    bounds[box_no][dim][1],
                                    stepsizes[box_no, dim]))

    # Compute the sum of the function over the box using itertools.product
    f_sum = 0.0
    for point in itertools.product(*coord_list):
        f_sum += function(np.array(point))
    
    return f_sum

# Helper function to perform the integration for one box
def compute_sampling_box_integral(bounds, stepsizes, box_no, point_percentages,
                                  sample_from_box, function, dimension):
    
    lengths = [DBUIterator.count_points(
                           bounds[box_no][d][0],
                           bounds[box_no][d][1],
                           stepsizes[box_no, d]) for d in range(dimension)]
    
    points_per_box = 1
    for l in lengths:
        points_per_box = points_per_box * l
    
    # Determine how many points to sample from the box
    sampling_number = int(points_per_box * point_percentages[box_no])
    function_sum = 0
    
    # Sample the points and compute the sum of the function evaluations
    for _ in range(sampling_number):
        function_sum += function(sample_from_box(box_no))
    
    return function_sum, sampling_number


def compute_dim_bound(centres, lengths, box_no, d):
    l_bound = centres[box_no, d] - lengths[box_no, d] / 2
    r_bound = centres[box_no, d] + lengths[box_no, d] / 2
    return [l_bound, r_bound]


def check_point_dim_parallel(point, box_bounds):
    '''
    Check if point is within a single box using parallelization over dimensions.
    '''
    # Parallelize the dimension check for the current box
    dim_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(lambda d: box_bounds[d][0] <= point[d] <= box_bounds[d][1])(d)
        for d in range(len(point))
    )
    return all(dim_results)


def check_point_in_box(args):
      point, box_no, bounds, dimension = args
      for d in range(dimension):
          if point[d] > bounds[box_no][d][1] or point[d] < bounds[box_no][d][0]:
              return None  # Not in box
      return box_no  # Found the box


def generate_conditions_worker(args):
    box_bounds, box_stepsizes, box_no, max_complexity, total_bounds, dimension = args
    local_conditions = []
    cartesian_prod = []

    # Iterate over the box's bounds (not total_bounds!)
    for dim in range(dimension):
        cartesian_prod.append(np.arange(
            box_bounds[dim][0],
            box_bounds[dim][1],
            box_stepsizes[dim]
        ))

    # Iterate over complexities
    for k in range(1, max_complexity + 1):
        all_k_tuples = DisjointBoxUnion.generate_k_tuples(cartesian_prod, k)
        for indices, k_tuples in all_k_tuples:
            for k_tuple in k_tuples:
                k_array = DisjointBoxUnion.tuples_to_array(k_tuple)
                constraint = cc.ConstraintConditions(
                    dimension=dimension,
                    non_zero_indices=np.array(indices),
                    bounds=k_array,
                    state_bounds=box_bounds  # Optional: could use box_bounds instead of total_bounds
                )
                local_conditions.append(constraint)

    return local_conditions



def dp_integrate_worker(args):
   
    dbu, box_no, function = args
    dp_cache = {}
    box_integral = 0.0
    coord_list = []
   
    for dim in range(dbu.dimension):
        coord_list.append(np.arange(
            dbu.bounds[box_no][dim][0],
            dbu.bounds[box_no][dim][1],
            dbu.stepsizes[box_no, dim]
            ))
   
    for point in itertools.product(*coord_list):
        point_key = tuple(point)
        if point_key not in dp_cache:
            dp_cache[point_key] = function(np.array(point))
            box_integral += dp_cache[point_key]
   
    return box_integral


def compute_closest_coord(args):
    
    i, point, point_box_no, centres, lengths, stepsizes = args
    min_bound = centres[point_box_no, i] - lengths[point_box_no, i] / 2
    max_bound = centres[point_box_no, i] + lengths[point_box_no, i] / 2 - stepsizes[point_box_no, i]
    closest_coord = round((point[i] - min_bound) / stepsizes[point_box_no, i]) * stepsizes[point_box_no, i] + min_bound
    closest_coord = max(min_bound, min(closest_coord, max_bound))
    
    return closest_coord


def compute_closest_point_parallel(point, point_box_no, centres,
                                   lengths, stepsizes, dimension,
                                   n_workers=None):
    
   args_list = [(i, point, point_box_no, centres, lengths, stepsizes) for i in range(dimension)]
   
   with ProcessPoolExecutor(max_workers=n_workers) as executor:
       closest_point = list(executor.map(compute_closest_coord, args_list))
   
   return np.array(closest_point)



#%%
                                                                           
class DisjointBoxUnion:                                                         
                                                                                                                                                    
    '''                                                                        
    Represent a disjoint union of d-dimensional boxes in \mathbb{R}^d.         
    '''

    def __init__(self, no_of_boxes, dimension, lengths, centres, stepsizes = [],
                 store_endpoints = True, store_bounds = True, complexity = 0.0):
        '''
        Parameters:    
        -----------------------------------------------------------------------
            no_of_boxes : int                                                  
                          Number of disjoint boxes                                 

            dimension : int                                                    
                        Dimension of the space we reside in                    

            lengths : np.array(no_of_boxes, dimension)                          
                      A numpy array that denotes the lengths of the different  
                      dimensions                                               
                                                                               
            centres : np.array(no_of_boxes, dimension)                          
                      A numpy array denoting the coordinates of the different  
                      centres of the many boxes                                  
                                                                               
            stepsizes : np.array(no_of_boxes, dimension) or int or float       
                        stepsizes[i, d] = the stepsize for the nth box and the 
                        dth dimension
            
        complexity : int                                                       
                     The complexity of the DBU in case it is of type cond_DBU   
            
        Stores:
        -----------------------------------------------------------------------
            bounds : list(2 * d) of length no_of_boxes
                     The lower and upper bound for each dimension and each box
            
        end_points : np.array(N * 2D * D)
                     The end points of the n'th box each of which has d 
                     coordinates
                        
        '''

        self.no_of_boxes = no_of_boxes
        self.dimension = dimension
        
        if len(centres.shape) == 1:
            self.centres = np.expand_dims(centres, axis=0)
        else:
            self.centres = centres
        
        if len(lengths.shape) == 1:
            self.lengths = np.expand_dims(lengths, axis=0)
        else:
            self.lengths = lengths
        
        self.point_count = None
        
        if type(stepsizes) == int or type(stepsizes) == float:
            self.stepsizes = stepsizes * np.ones((no_of_boxes, dimension))
        
        elif np.sum(stepsizes) == 0:
            self.stepsizes = np.ones((no_of_boxes, dimension))
        
        elif len(stepsizes.shape) == 1:
            self.stepsizes = np.expand_dims(stepsizes, axis=0)
        
        else:
            self.stepsizes = stepsizes
        
        if store_bounds:
            self.bounds = self.get_bounds()
        if store_endpoints:
            self.endpoints = self.get_end_points()
        
        self.complexity = complexity
        self.all_points = []
    
    def no_of_points(self):
        '''
        Given a DBU return the total number of points in it.
        Parallelized over boxes for performance.
        '''
        args_list = [(box_no, self.bounds, self.stepsizes, self.dimension)
                     for box_no in range(self.no_of_boxes)]
    
        with mp.Pool(processes=mp.cpu_count()) as pool:
            points_per_box = pool.map(compute_points_per_box, args_list)
    
        point_count = sum(points_per_box)
    
        return max(point_count, 1)  # Ensure non-zero
    
    def add_disjoint_rectangle(self, dimension, centre, lengths, stepsize=[]):
        '''
        Given a DBU add a new rectangle of the same dimension assuming it does
        not intersect with the existing DBU.
        
        Parameters:
        -----------------------------------------------------------------------
        dimension : int
                    The dimension of the new rectangle
        
        centre : np.array[dimension]
                 The centre of the new rectangle
        
        lengths : np.array[dimension]
                  The lengths of the sides of the new rectangle
        
        stepsize : np.array[dimension]
                   The stepsizes over the various dimensions of the new rectangle.
        
        '''
        
        if dimension != self.dimension:
            print(f'Dimension of new box which is {dimension} != {self.dimension} which is the dimension of the DBU.')
            return
        
        if self.no_of_boxes == 0:
            self.lengths = np.expand_dims(lengths, axis=0)
            self.stepsize = np.expand_dims(np.ones(dimension), axis=0)
            self.centres = np.expand_dims(centre, axis=0)
        
        else:
            self.lengths = np.vstack([self.lengths, lengths])
            stepsize = np.ones(dimension)
            self.stepsizes = np.vstack([self.stepsizes, stepsize])
            self.centres = np.vstack([self.centres, centre])

        self.no_of_boxes += 1        

    # Main sampling_integrate method inside your DisjointBoxUnion class
    def sampling_integrate(self, function, point_percentages=0.5):
        '''
        Integrate a function over a disjoint box union data structure. We assume
        to do an MCMC-based sampling integral by sampling points from each box.
        
        Parameters:
        -----------------------------------------------------------------------
            function : func
                       The d-dimensional input function to integrate over the DBU.
        
            point_percentages : float or list
                                The percentage of points sampled from each 
                                box of the DBU for the integration.
        
        Returns:
        -----------------------------------------------------------------------
            integral : float
                      The estimate of the Riemannian summation integral over the DBU.
        '''
        
        if type(point_percentages) == float:
            point_percentages = [point_percentages for _ in range(self.no_of_boxes)]
        
        # Parallelize the sampling and integration for each box
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.starmap(compute_sampling_box_integral, 
                                            [(self.bounds, self.stepsizes, box_no, point_percentages, self.sample_from_box, function, self.dimension) 
                                             for box_no in range(self.no_of_boxes)]),
                               total=self.no_of_boxes))
        
        # Aggregate the results from all boxes
        total_function_sum = sum(result[0] for result in results)
        total_sampled_points = sum(result[1] for result in results)
        
        return total_function_sum / total_sampled_points
        
    
    # Inside your DisjointBoxUnion class
    def integrate(self, function):
        '''
        Given a function taking inputs in d dimensions, evaluate the Riemann-Integral 
        of the function over the DBU using parallelization.
    
        Parameters:
            -----------------------------------------------------------------------
            function : func
                       The d-dimensional function to integrate over the DBU
    
        Returns:         
            -----------------------------------------------------------------------
            integral : float
                      The value of the integral of the function over the DBU.
        '''
    
        if self.bounds is None:
            self.get_bounds()
    
        # Parallelize the sum of integrals for each box using Pool
        with Pool(processes=cpu_count()) as pool:
            f_sums = list(tqdm(pool.starmap(compute_box_integral, 
                                            [(self.bounds, self.stepsizes, self.dimension, function, box_no) 
                                             for box_no in range(self.no_of_boxes)]), 
                                             total=self.no_of_boxes))
    
            # Sum the results from all boxes and normalize by the number of points
            total_sum = sum(f_sums)
            return total_sum / self.no_of_points()
         
    
    def subtract_constraint(self, constraint_condition):
        '''
        Given a DBU subtract a constraint condition from it to give another DBU.

        Parameters:
        -----------------------------------------------------------------------
            constraint_condition : type(constraint_conditions)
                                   The constraint condition we wish to subtract
                                   from the DBU

        Returns:
        -----------------------------------------------------------------------
            final_DBU : The final disjoint box union after removing the region
            under the constraint condition
        '''
        final_DBU = DisjointBoxUnion(0, self.dimension,
                                     np.array([[]]), np.array([[]]))
        
        for box_no in range(self.no_of_boxes):
            
            constraint_DBU = DisjointBoxUnion.condition_to_DBU(constraint_condition,
                                                               self.stepsizes[box_no])
            
            single_box_DBU = DisjointBoxUnion(1, self.dimension,
                                              np.array([self.lengths[box_no]]),
                                              np.array([self.centres[box_no]]),
                                              np.array([self.stepsizes[box_no]]))
            
            subtracted_DBU = single_box_DBU.subtract_single_DBUs(constraint_DBU)
            if np.sum(subtracted_DBU.lengths) > 0:
                final_DBU = final_DBU.append_disjoint_DBUs(subtracted_DBU)

        return final_DBU
    
    
    def subtract_DBUs(self, DBU):
        '''
        Given a DBU self and another DBU, subtract one from the other.
        
        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to subtract from the original DBU
        
        Returns:
        -----------------------------------------------------------------------
        subtracted_DBU : DisjointBoxUnion
                         The DBU we get after subtraction
        '''
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for i in range(self.no_of_boxes):
            
            DBU_single = DisjointBoxUnion(1, self.dimension,
                                          np.array([self.lengths[i]]),
                                          np.array([self.centres[i]]))
            
            subtracted_DBU = DBU_single
            #print(f'DBU single is {DBU_single}')
            for j in range(DBU.no_of_boxes):
                DBU_2 = DisjointBoxUnion(1, self.dimension,
                                         np.array([DBU.lengths[j]]),
                                         np.array([DBU.centres[j]]))
                
                #print(f'We remove {DBU_2} from {subtracted_DBU}')
                subtracted_DBU = subtracted_DBU.subtract_single_DBUs(DBU_2)
                if subtracted_DBU.no_of_boxes == 0:
                    break

            if subtracted_DBU.no_of_boxes != 0:    
                final_DBU = final_DBU.append_disjoint_DBUs(subtracted_DBU)
            
        return final_DBU
    
    @staticmethod
    def has_zero(arr):
        return np.any(arr == 0)
    
    
    def subtract_single_DBUs(self, DBU):
        '''
        Given a DBU with one box and another DBU with just one box, subtract
        the self DBU from the other DBU. The stepsizes are borrowed from the DBU
        we subtract from.
        
        We do not return degenerate rectangles, which are those rectangles which have
        one dimension of the side length be 0

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to subtract from the original DBU.

        Returns:
        -----------------------------------------------------------------------
        subtracted_DBU : DisjointBoxUnion
                         The DBU we get after subtraction       
        '''
        centre1 = self.centres
        lengths1 = self.lengths
        centre2 = DBU.centres
        lengths2 = DBU.lengths
        stepsizes = self.stepsizes
        
        #print(f'We subtract C={centre1[0]} and L={lengths1[0]} and C={centre2[0]} and L = {lengths2[0]}')
        
        rectangle_list = op.subtract_rectangles(centre1[0], lengths1[0],
                                                centre2[0], lengths2[0])
        
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for rect_no in range(len(rectangle_list)):
            
            centres = np.expand_dims(rectangle_list[rect_no][0], axis=0)
            non_lengths = rectangle_list[rect_no][1]
            
            if not DisjointBoxUnion.has_zero(non_lengths):
                lengths = np.expand_dims(non_lengths, axis=0)
                
                final_DBU= final_DBU.append_disjoint_DBUs(DisjointBoxUnion(1,
                                                                       self.dimension,
                                                                       lengths,
                                                                       centres,
                                                                       stepsizes))
            
        return final_DBU
    
    def append_disjoint_DBUs(self, DBU):
        '''
        Append a DBU to the original DBU

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The new DBU to append to the original DBU

        Returns:
        -----------------------------------------------------------------------
        appended_DBU : DisjointBoxUnion
                       The merged DBU

        '''
        new_box_number = DBU.no_of_boxes + self.no_of_boxes
        
        if np.sum(self.lengths) == 0:
            new_lengths = DBU.lengths
            new_centres = DBU.centres
            new_stepsizes = DBU.stepsizes
        else:
            new_lengths = np.concatenate((self.lengths, DBU.lengths), axis = 0)
            new_centres = np.concatenate((self.centres, DBU.centres), axis = 0)
            new_stepsizes = np.concatenate((self.stepsizes, DBU.stepsizes), axis =0)
        
        return DisjointBoxUnion(new_box_number, self.dimension, new_lengths, new_centres,
                                new_stepsizes)
    
    def DBU_union(self, DBU):
        '''
        Given a self DBU, take the union of this with the external DBU.
        
        # A1 sqcup A2 sqcup ... Ak union B = B sqcup A-B
        
        Returns:
        -----------------------------------------------------------------------
            new_DBU : DisjointBoxUnion
                      The new data structure which contains the union with the 
                      old DBU
        '''
        
        
        sub_DBU = self.subtract_DBUs(DBU)
        
        return DBU.append_disjoint_DBUs(sub_DBU)
        
    
    def get_bounds(self):
        '''
        Compute bounds for each box in the DBU, parallelizing over the dimension space.
        '''
        bounds = []
        for box_no in range(self.no_of_boxes):
            dim_bounds = Parallel(n_jobs=-1, prefer="threads")(
                delayed(compute_dim_bound)(self.centres, self.lengths, box_no, d)
                for d in range(self.dimension)
            )
            bounds.append(dim_bounds)
    
        self.bounds = bounds
        return bounds

    
    def sample_from_dbu(self):
        '''
        Given a DisjointBoxUnion, sample a point randomly from it.
        
        '''
        box_no = random.randrange(self.no_of_boxes)
        lengths = [DBUIterator.count_points(
                   self.bounds[box_no][d][0],
                   self.bounds[box_no][d][1],
                   self.stepsizes[box_no, d]) for
                   d in range(self.dimension)]
        
        point_index = [random.randrange(l) for l in lengths]
        sampled_point = DBUIterator.index_to_point(point_index, self, box_no)
        
        return sampled_point
    
    
    def sample_from_box(self, box_no):
        '''
        Given a DisjointBoxUnion, sample from a certain box with a given box_no.

        Parameters:
        -----------------------------------------------------------------------
        box_no : int
                 The box number from the DBU we wish to sample from.

        Returns:
        -----------------------------------------------------------------------
        point : np.array[d]
                The new point obtained after sampling
        '''
        lengths = [DBUIterator.count_points(
                   self.bounds[box_no][d][0],
                   self.bounds[box_no][d][1],
                   self.stepsizes[box_no, d]) for
                   d in range(self.dimension)]
        
        
        point_index = [random.randrange(l) for l in lengths]
        sampled_point = DBUIterator.index_to_point(point_index, self, box_no)
        
        return sampled_point
    
    
    @staticmethod
    def condition_to_DBU(condition, stepsizes):
        '''
        Given a constraint condition, create the corresponding DBU with one box
                                                                                
        Parameters:
        -----------------------------------------------------------------------
        condition : type(constraint_conditions)
            	    The condition which we want to get a DBU from
        	
        stepsizes : np.array[dimension] or int or float
                    The stepsizes of the new DBU in the different dimensions

        Returns:
        -----------------------------------------------------------------------
        constraint_DBU : DisjointBoxUnion
                         The DBU corresponding to the constraint 
        '''
        centres = []
        lengths = []
        
        if len(condition.state_bounds) > 0:
            
            for d in range(condition.dimension):
                centres.append((condition.bounds[d,0] + condition.bounds[d,1])/2)
                lengths.append(condition.bounds[d,1] - condition.bounds[d,0])
        
        else:
            centres = np.zeros(condition.dimension)
            lengths = np.array([np.inf for i in range(condition.dimension)])
        
        centres = np.expand_dims(np.array(centres), axis = 0)
        lengths = np.expand_dims(np.array(lengths), axis = 0)
        
        #print('Non zero indices are')
        #print(condition.non_zero_indices)
        
        for i,ind in enumerate(condition.non_zero_indices):                           
            
            centres[0, ind] = (condition.bounds[i, 0] + condition.bounds[i, 1])/2         
            lengths[0, ind] = (condition.bounds[i,1] - condition.bounds[i,0])             
            
            if lengths[0, ind] == 0:                                                    
                return DisjointBoxUnion.empty_DBU(condition.dimension)                                     
                                                                                   
        if type(stepsizes) == int or type(stepsizes) == float:                      
            stepsizes = stepsizes * np.ones((1, condition.dimension))                 
        elif len(stepsizes.shape) == 1:
            stepsizes = np.expand_dims(stepsizes, axis=0)
        
        
        return DisjointBoxUnion(1, condition.dimension, lengths, centres,
                                stepsizes, complexity = condition.complexity)
    
    
    def pick_random_point(self):
        '''
        Given a DisjointBoxUnion, pick a point randomly from it
        
        '''
        all_points = []
        if len(self.all_points) == 0:
            dbu_iter_class = DBUIterator(self)
            dbu_iterator = iter(dbu_iter_class)
            for s in dbu_iterator:
                all_points.append(s)
            
            self.all_points = all_points
        
        return random.sample(self.all_points, 1) 
        
    
    def get_total_bounds(self):
        '''
        Given a DBU get the maximum and the minimum bounds in the various dimensions for the distinct 
        boxes
        
        Returns:
        -----------------------------------------------------------------------
            total_bounds : np.array((dimension * 2))
                           The total bounds for the DBU
        Stores:
        -----------------------------------------------------------------------
            total_bounds : Same description as above
        '''
        total_bounds = np.array([[+np.inf, -np.inf] for d in range(self.dimension)])
        bounds = np.array(self.get_bounds())
        
        for d in range(self.dimension):
            for box_no in range(len(bounds)):
                
                if bounds[box_no,d,0] < total_bounds[d,0]:
                    total_bounds[d, 0] = bounds[box_no, d, 0]
                
                if bounds[box_no,d,1] > total_bounds[d,1]:
                    total_bounds[d,1] = bounds[box_no, d,1]
        
        self.total_bounds = total_bounds
        return total_bounds
    
    
    def get_end_points(self):
        '''
        Given a DBU, obtain the end points of the different boxes.

        Returns:
        -----------------------------------------------------------------------
            end_points : np.array(N * 2^D * D)
                         The end points of the n'th box each of which has d coordinates
                                                                                                         
        '''
        end_points = []
        for box_no in range(self.no_of_boxes):
            
            center = self.centres[box_no]
            side_lengths = np.array(self.lengths[box_no])
        
            # Create an array of all possible combinations of -1 and 1 of length d
            offsets = np.array(np.meshgrid(*[[-0.5, 0.5]] * self.dimension)).T.reshape(-1, self.dimension)
        
            # Scale by the side lengths and add the center to get the endpoints
            vertices = center + offsets * side_lengths                         
            
            end_points.append(vertices)
        
        self.end_points = end_points
        
        return end_points
    
    
    def find_point_in_DBU(self, point, parallel=True):
        '''
        Given a point, find the closest corresponding point in the discretized 
        state space (the DBU).
        -----------------------------------------------------------------------
        point : np.array
                Point which we wish to locate in the DBU
        
      parallel : Binary
                 Do we wish to parallelize the search? 
        '''
        bounds = self.get_bounds()
        point_box_no = np.inf

        # Parallel search for containing box
        if parallel:
            with ProcessPoolExecutor() as executor:
                args = [(point, box_no, bounds, self.dimension) for box_no in range(self.no_of_boxes)]
                for result in executor.map(check_point_in_box, args):
                    if result is not None:
                        point_box_no = result
                        break
        else:
            # Serial fallback
            for box_no in range(self.no_of_boxes):
                point_in_box = True
                for d in range(self.dimension):
                    if point[d] > bounds[box_no][d][1] or point[d] < bounds[box_no][d][0]:
                        point_in_box = False
                        break
                if point_in_box:
                    point_box_no = box_no
                    break

        # If not in any box, find the closest box
        if point_box_no == np.inf:
            closest_index, min_distance, closest_point_on_box = DisjointBoxUnion.find_closest_box_to_point(
                point, self.centres, self.lengths
            )
            point_box_no = closest_index
            point = closest_point_on_box
        
        
        return compute_closest_point_parallel(point, point_box_no, self.centres,
                                              self.lengths, self.stepsizes, self.dimension)
        

    def is_point_in_DBU(self, point):
        '''
        Given a point, check if it is in any box of the DBU.
        Fully parallel: over boxes and dimensions.
        -----------------------------------------------------------------------
        point : np.array[d]
                Check if this point is in the DBU
        '''
        bounds = self.get_bounds()
        
        # Parallelize over boxes
        box_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(check_point_dim_parallel)(point, bounds[box_no])
            for box_no in range(self.no_of_boxes)
        )
        
        return any(box_results)

        
    @staticmethod
    def integrate_static(dbu, function, parallel=True, n_workers=None):
        '''
        Integrate a function over a dbu in parallel.
    
        Parameters:
        -----------------------------------------------------------------------
        dbu : DisjointBoxUnion
              The DBU we integrate over
    
        function : callable
                   The function that we integrate
    
        parallel : bool
                   Whether to parallelize the computation (default: True)
        
        n_workers : int or None
                    Number of parallel workers (default: None -> auto)
    
        Returns:
        -----------------------------------------------------------------------
        integral : float
                   The total integral of the function over the dbu
        '''
    
        if dbu.bounds is None:
            dbu.get_bounds()
    
        # Prepare all points to evaluate
        all_points = []
        for box_no in range(dbu.no_of_boxes):
            coord_list = []
            for dim in range(dbu.dimension):
                coord_list.append(np.arange(dbu.bounds[box_no][dim][0],
                                            dbu.bounds[box_no][dim][1],
                                            dbu.stepsizes[box_no, dim]))
            # Cartesian product of coordinates within this box
            box_points = list(itertools.product(*coord_list))
            all_points.extend(box_points)
    
        # Define a wrapper to apply the function
        def evaluate(point):
            return function(np.array(point))
    
        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(evaluate, all_points)
                f_sum = sum(results)
        else:
            # Fallback to serial computation
            f_sum = sum(evaluate(p) for p in all_points)
    
        return f_sum / dbu.no_of_points()
        
          
    @staticmethod
    def easy_integral_static(dbu, function, parallel=True, n_workers=None):
        '''
        Perform an easy integral of the function over the dbu. This involves
        taking the average of the function over the different boxes in parallel.
    
        Parameters:
    ---------------------------------------------------------------------------
            dbu : DisjointBoxUnion
                  The DBU we average over
    
            function : callable
                       The function that we integrate
    
            parallel : bool
                       Whether to parallelize the computation (default: True)

            n_workers : int or None
                        Number of parallel workers (default: None -> auto)

        Returns:
    -----------------------------------------------------------------------
            integral : float
                   The integral value we obtain
    '''

        endpoints = np.array(dbu.get_end_points())
        no_of_boxes = dbu.no_of_boxes
        dim = dbu.dimension

        def integrate_box(i):
            box_sum = 0
            for c in range(endpoints.shape[1]):
                box_sum += function(endpoints[i, c, :])
            return box_sum / (2 ** dim)
    
        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                box_integrals = list(executor.map(integrate_box, range(no_of_boxes)))
            integral_sum = sum(box_integrals)
        else:
            integral_sum = sum(integrate_box(i) for i in range(no_of_boxes))
    
        return integral_sum / no_of_boxes
    
    @staticmethod
    def n_quad_integrate(dbu, function):
        '''
        Perform an n_quad integral of the function over the dbu. 
        
        Parameters:
        -----------------------------------------------------------------------
        dbu : DisjointBoxUnion
              The DBU we average over
        
        function : func
                   The function that we integrate with
        
        Returns:
        -----------------------------------------------------------------------
        integral : float
                   The integral value we obtain
        '''
        integral_sum = 0
        bounds = np.array(dbu.get_bounds())
        for i in range(dbu.no_of_boxes):
            print(f'{i}th bounds is')
            print(bounds[i])
            integral_sum += integrate.nquad(function, bounds[i])
        
        return integral_sum / dbu.no_of_boxes
    
    @staticmethod
    def sampling_integrate_static(dbu, function,
                                  point_percentages = 0.3):
        '''
        Integrate a function over a disjoint box union data structure. We assume
        to do a MCMC based sampling integral by sampling points_per_box number of points
        or point_percentage number of points in each box.
        
        Parameters:
        -----------------------------------------------------------------------
            function : func
                       The d dimensional input function to integrate over the DBU.
        
            
            point_percentages : float or list
                               The percentage of points we sample from each 
                               box of the DBU to perform the integration of the function
                                                                               
        Returns:
        -----------------------------------------------------------------------
            integral : float
                       The estimate of the Riemannian summation integral over the DBU. 
        
        '''
        
        if type(point_percentages) == float:
            point_percentages = [point_percentages for i in range(dbu.no_of_boxes)]
        
        
        sampled_points = 0
        function_sum = 0
        for box_no in range(dbu.no_of_boxes):
            
            lengths = [DBUIterator.count_points(
                                   dbu.bounds[box_no][d][0],
                                   dbu.bounds[box_no][d][1],
                                   dbu.stepsizes[box_no, d]) for
                                   d in range(dbu.dimension)]
            
            points_per_box = 1
            for l in lengths:
                points_per_box = points_per_box * l
        
            sampling_number = int(points_per_box * point_percentages[box_no])
            if sampling_number == 0:
                sampling_number = 1
            sampled_points += sampling_number
            
            for s in range(sampling_number):
                
                function_sum += function(dbu.sample_from_box(box_no))
        
        if sampled_points == 0:
            sampled_points = 1
        #print(f'Sampled points is {sampled_points}')
        return function_sum / sampled_points
    
    
    @staticmethod
    def find_closest_box_to_point(P, box_centers, box_lengths):
        P = np.array(P)
        closest_index = None
        min_distance = float('inf')
        closest_point_on_box = None
        
        for i, (center, length) in enumerate(zip(box_centers, box_lengths)):
            center = np.array(center)
            length = np.array(length) / 2  # Half-lengths
            
            # Compute the minimum distance to the box edges along each dimension
            lower_bounds = center - length
            upper_bounds = center + length
            
            # Distance to the nearest point on the box surface
            clamped_point = np.maximum(lower_bounds, np.minimum(P, upper_bounds))
            distance = np.linalg.norm(P - clamped_point)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                closest_point_on_box = clamped_point
        
        return closest_index, min_distance, closest_point_on_box
    
    
    @staticmethod
    def generate_k_tuples(array_list, k):                                                                   
        '''
        Given a list of lists, return all possible k-tuples from the list of lists,
        and also return the indices. 

        Parameters:
        -----------------------------------------------------------------------
        lists : list[list]
                The list of lists we wish to sample from
        
        k : int
            The number of elements we want to look at in the tuples

        Returns:
        -----------------------------------------------------------------------
        results : tuple
                  The tuple which denotes the values and the indices we wish to
                  sample from 

        '''
        # Step 3: Generate all possible combinations of indices of size k
        index_combinations = list(itertools.combinations(range(len(array_list)), k))

        # Step 4: Generate all possible k-tuples for each combination
        all_k_tuples = []
        for indices in index_combinations:
            # Extract the sub-arrays corresponding to the current combination of indices
            sub_arrays = []
            for i in indices:
                cart_arr = itertools.product(array_list[i], repeat = 2)
                unequal_tuples = [tup for tup in cart_arr if tup[0]<tup[1]]
                sub_arrays.append(unequal_tuples)

            # Generate all possible k-tuples from the sub-arrays
            k_tuples = list(itertools.product(*sub_arrays))
            # Store the results
            all_k_tuples.append((indices, k_tuples))

    
        return all_k_tuples
    
    @staticmethod
    def tuples_to_array(tup):
        '''
        Given a tuple of tuples convert it into an array
        -----------------------------------------------------------------------
        tup : tuple
              The tuple that needs to be converted to an array
        '''
        final_arr = []
        for i,elt in enumerate(tup):
            final_arr.append(np.array(elt))
        
        return np.array(final_arr)
    
    def dp_integrate(self, function, parallel=True, n_workers=None):
        '''
        Integrate a function over the DBU using dynamic programming 
        to avoid redundant computations.

        Parameters:
            function : callable
                The d-dimensional function to integrate over the DBU

        Returns:
            integral : float
                The value of the integral of the function over the DBU
        '''
        if self.bounds is None:
            self.get_bounds()

        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                box_integrals = list(executor.map(dp_integrate_worker,
                    [(self, box_no, function) for box_no in range(self.no_of_boxes)]))
            total_integral = sum(box_integrals)
        else:
            total_integral = sum(dp_integrate_worker((self, box_no, function)) for box_no in range(self.no_of_boxes))

        return total_integral / self.no_of_points()
    
    #Create an iterator method for conditions to speed up generate all conditions?
    
    def generate_all_conditions(self, max_complexity,
                                parallel=True, n_workers=None):
        '''
        For the given DBU, generate all the bounded conditions possible for
        a given complexity.
    
        Parameters:
        -----------------------------------------------------------------------
            max_complexity : int
                             Possible number of coordinates that can be
                             modified in the DBU.
    
        Returns:
        -----------------------------------------------------------------------
            conditions : list[constraint_conditions]
                         The list of bounded conditions generated for the given DBU.
        '''
        total_bounds = self.get_total_bounds()
        no_of_boxes = self.no_of_boxes
    
        if parallel:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                all_conditions = list(executor.map(
                    generate_conditions_worker,
                    [
                        (
                            self.get_bounds()[box_no],         
                            self.stepsizes[box_no] if self.stepsizes.ndim > 1 else self.stepsizes,
                            box_no,
                            max_complexity,
                            total_bounds,
                            self.dimension
                        )
                        for box_no in range(no_of_boxes)
                    ]
                ))
            condition_list = [c for sublist in all_conditions for c in sublist]
        else:
            condition_list = []
            for box_no in range(no_of_boxes):
                stepsize_box = self.stepsizes[box_no] if self.stepsizes.ndim > 1 else self.stepsizes
                condition_list.extend(generate_conditions_worker(
                    (
                        self.get_bounds()[box_no],         
                        stepsize_box,
                        box_no,
                        max_complexity,
                        total_bounds,
                        self.dimension
                    )
                ))
    
        return condition_list

    def plot_2D(self, ax=None, show_plot=True, title=''):
        '''
        Plot a disjoint union of boxes in 2D
        '''
        if ax == None:
            fig,ax = plt.subplots()
        
        if self.bounds == None:
            self.get_bounds()
        #print(self.total_bounds)
        #print(type(self.total_bounds))
        
        for box_no in range(self.no_of_boxes):
            ax.scatter([self.centres[box_no, 0]], [self.centres[box_no, 1]],
                       color = 'red', marker = 'x')
            
            ax.hlines(y=self.centres[box_no, 1] + self.lengths[box_no, 1]/2, color='blue',
                       xmin=self.centres[box_no, 0] - self.lengths[box_no, 0]/2,
                       xmax=self.centres[box_no, 0] + self.lengths[box_no, 0]/2)
            
            ax.hlines(y=self.centres[box_no, 1] - self.lengths[box_no, 1]/2, color='blue',
                       xmin=self.centres[box_no, 0] - self.lengths[box_no, 0]/2,
                       xmax=self.centres[box_no, 0] + self.lengths[box_no, 0]/2)
            
            ax.vlines(x=self.centres[box_no, 0] + self.lengths[box_no, 0]/2, color='blue',
                       ymin = self.centres[box_no, 1] - self.lengths[box_no, 1]/2,
                       ymax = self.centres[box_no, 1] + self.lengths[box_no, 1]/2)
            
            ax.vlines(x=self.centres[box_no, 0] - self.lengths[box_no, 0]/2, color='blue',
                       ymin = self.centres[box_no, 1] - self.lengths[box_no, 1]/2,
                       ymax = self.centres[box_no, 1] + self.lengths[box_no, 1]/2)
        
        #ax.set_xlim(self.total_bounds[0,0]-1, self.total_bounds[1,0]+1)
        #ax.set_ylim(self.total_bounds[0,1]-1, self.total_bounds[1,1]+1)
        ax.set_title(title)
        if show_plot:
            plt.show()
            
        return ax
    
    
    def __str__(self):
        '''
        Print the various rectangles present in the DBU.
        
        '''
        dbu_string = ''
        for box_no in range(self.no_of_boxes):
            dbu_string += f'Box: {box_no}, Centre: {self.centres[box_no]}, Lengths: {self.lengths[box_no]}, Stepsizes: {self.stepsizes[box_no]}\n'
        
        return dbu_string    
    
    def dbu_to_tuple(self):
        '''
        Given a DBU get the tuple associated to it which is to be used in storing the dicts
        '''
        return (self.no_of_boxes, self.dimension,
                DisjointBoxUnion.tuplify_2D_array(self.lengths),
                DisjointBoxUnion.tuplify_2D_array(self.centres))
    
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
    
    
    def dbu_to_constraint(self):
        
        '''
        Given a dbu find the corresponding constraint condition associated to it
        '''
        
        return        
        
    @staticmethod                                                              
    def empty_DBU(dimension):                                                  
        '''                                                                    
        A static method that returns an empty DBU of a certain dimension       
        Parameters:                                                             
        -----------------------------------------------------------------------
        dimension : int
                    Dimension of the empty DBU

        Returns:
        -----------------------------------------------------------------------
        empty_DBU : type(DisjointBoxUnion)
                    An empty disjoint box union data structure

        '''
        return DisjointBoxUnion(0, dimension, np.array([[]]), np.array([[]]))

#%%
    
class DBUIterator:
    '''
    Create a DBU iterator for iterating through the points of the different boxes
    '''
    def __init__(self, DBU):
    
        self.DBU = DBU
        self.curr_box_no = 0
        self.point_index = np.zeros(self.DBU.dimension)
        self.first_point = True
        
    @staticmethod
    def count_points(a, b, d):
        '''
        Count the number of points in the list given by np.arange(a,b,d)

        Parameters:
        -----------------------------------------------------------------------
        a : float
            Lower Bound
        b : float
            Upper Bound
        d : float
            Difference

        Returns:
        -----------------------------------------------------------------------
        number_of_points : int
                           The number of points in the list

        '''
        return max(0, int(np.ceil(b-a) / d))
    
    @staticmethod
    def index_to_point(point_index, DBU, box_no):
        '''
        Given the set of indices for a certain box_no for a DBU return the corr
        point on the DBU

        Parameters:
        -----------------------------------------------------------------------
        point_index : np.array([int])
                      The indices of the point in the box

        Returns:
        -----------------------------------------------------------------------
        point : np.array[float]
                The point values on the box
        '''
        point = []
        for d in range(DBU.dimension):
            
            point.append(DBU.bounds[box_no][d][0] + (point_index[d]) * DBU.stepsizes[box_no, d])
            
        return point
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        for box_no in range(self.curr_box_no, self.DBU.no_of_boxes):
            
            self.lengths = [DBUIterator.count_points(
                                   self.DBU.bounds[box_no][d][0],
                                   self.DBU.bounds[box_no][d][1],
                                   self.DBU.stepsizes[box_no, d]) for
                                   d in range(self.DBU.dimension)]
            
            if self.first_point:
                self.first_point = False
                return DBUIterator.index_to_point(np.zeros(self.DBU.dimension), self.DBU, box_no)
                
            
            # Update the position for the next point
            for i in range(self.DBU.dimension - 1, -1, -1):

                if self.point_index[i] + 1 < self.lengths[i]:
                    self.point_index[i] += 1
                    point = DBUIterator.index_to_point(self.point_index ,self.DBU, box_no)
                    # We manually add this change so that we can return the true point
                    
                    return point
                else:
                    self.point_index[i] = 0
                    #print(f'Point index : {point_index}')
                    #print(f'Else condition reached and i is {i}')
                    if i == 0:
                        self.curr_box_no += 1
                        self.first_point = True
                
            self.point_index = np.zeros(self.DBU.dimension)
            
        if self.curr_box_no >= self.DBU.no_of_boxes:
            raise StopIteration

#%%
class ConditionsIterator:
    '''
    For a DBU and a given box we define an iterator to iterate through all the conditions
    that could possibly be generated with this method.
    '''
    
    def __init__(self, DBU, max_complexity, cond_stepsizes = 0.0):
        '''
        We generate and iterate through all the possible conditions

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU over which we wish to generate all possible conditions
              
        curr_box_no : int
                 The box number in the DBU for which we wish to generate all possible constraint conditions
                 
        max_complexity : int
                         The maximum allowed complexity for a constraint condition

        Returns:
        -----------------------------------------------------------------------
        condition_iter class : ConditionIterator
                               The class which generates the iterations over all the conditions

        '''
        
        self.DBU = DBU
        self.max_complexity = max_complexity
        
        self.curr_box_no = 0
        self.complexity = 1
        self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
        self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
        
        self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
        self.bounds_per_box = DBU.get_bounds()
        self.curr_bounds = np.array(DBU.get_bounds()[0])
        self.total_bounds = DBU.get_total_bounds()
        
        if cond_stepsizes == 0.0:
            self.cond_stepsizes = self.DBU.stepsizes
        elif type(cond_stepsizes) == int or type(cond_stepsizes) == float:
            self.cond_stepsizes = np.ones((self.DBU.no_of_boxes, self.DBU.dimension))
        else:
            self.cond_stepsizes = cond_stepsizes
        
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        
        for box_no in range(self.curr_box_no, self.DBU.no_of_boxes):
            
            for complexity in range(self.complexity, self.max_complexity+1):
                
                for non_zero_indices in self.possible_non_zero_indices[-self.curr_non_zero_index_position:]:
                    non_zero_indices = np.array(non_zero_indices)
                    
                    for d in non_zero_indices[-self.iter_dim:]:
                        lengthstep = self.cond_stepsizes[box_no, d]
                        #print(f'We check if {curr_bounds[d,0]} + {lengthstep} < {curr_bounds[d,1]}')
                        if self.curr_bounds[d, 0] + lengthstep < self.curr_bounds[d, 1]:
                            self.curr_bounds[d, 0] += lengthstep
                            #print(f'New left bounds {self.curr_bounds[d,0]}')
                            
                            #print(f'Complexity : {complexity}, non_zero_indices : {non_zero_indices}, and dimension {d}')
                            constraint = cc.ConstraintConditions(self.DBU.dimension,
                                                                 non_zero_indices,
                                                                 self.curr_bounds,
                                                                 self.total_bounds)
                            #print(f'The constraint is {constraint}')
                            #print(f'Box no: {box_no}, complexity: {complexity}, dimension: {d}, non_zero_indices {non_zero_indices}')
                            return constraint
                        
                        elif self.curr_bounds[d, 1] - lengthstep > np.array(self.bounds_per_box[box_no])[d,0]:
                            
                             self.curr_bounds[d, 1] -= lengthstep
                             #print(f'New right bounds {self.curr_bounds[d,1]}')
                             self.curr_bounds[d, 0] = np.array(self.bounds_per_box[box_no])[d, 0]
                            
                            #print(f'Complexity : {complexity}, non_zero_indices : {non_zero_indices}, and dimension {d}')
                             constraint = cc.ConstraintConditions(self.DBU.dimension,
                                                                 non_zero_indices,
                                                                 self.curr_bounds,
                                                                 self.total_bounds)
                            #print(f'The constraint is {constraint}')
                             #print(f'Box no: {box_no}, complexity: {complexity}, dimension: {d}, non_zero_indices {non_zero_indices}')
                             return constraint
                        
                        else:
                            self.curr_non_zero_index_position -= 1
                            
                            if self.curr_non_zero_index_position > 0:
                                self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                            
                            else:
                                self.iter_dim -= 1
                                if self.iter_dim > 0:
                                    self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                    self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                else:
                                    self.complexity += 1
                                    if self.complexity <= self.max_complexity: 
                                        self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                        self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
                                        self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                        self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
                                    
                                    else:
                                        self.curr_box_no += 1
                                        if self.curr_box_no < self.DBU.no_of_boxes:
                                            self.complexity = 1
                                            self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                            self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
                                            self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                            self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
                                        
                                        else:                                   
                                            #print('Hurray! We reached the endpoint')                                    
                                            raise StopIteration                
                            
#%%

if __name__ == '__main__':
    dbu = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    dbu.get_bounds()[1]
    
    dbu = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    
    constraint_list = dbu.generate_all_conditions(2)
    fig,ax = plt.subplots()
    
    
    print('Constraint list length is')
    print(len(constraint_list))
    
#%%
    
    for i,constraint in enumerate(constraint_list):
        print(f'The {i}th constraint is {constraint}')                         
