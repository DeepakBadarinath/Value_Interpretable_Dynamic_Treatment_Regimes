#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:35:13 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU
import constraint_conditions as cc
import itertools
from importlib import reload

cc = reload(cc)
disjoint_box_union = reload(disjoint_box_union)

#%%
'''
Tests for add_disjoint_rectangle
'''

dbu = DBU.empty_DBU(2)
centres = np.array([[2, 4.5], [1, 1.5], [3, 2.5], [6, 5.5], [6, 3.5]])
lengths = np.array([[4, 3], [2, 3], [2, 1], [2, 1], [2, 3]])

for i in range(len(centres)):
    dbu.add_disjoint_rectangle(2, centres[i], lengths[i])
    dbu.plot_2D(title=f'After {i+1}th rectangle is plotted')


# %%
'''
Tests for subtract_single_DBUs
'''

c1 = np.array([0, 0])
l1 = np.array([4, 4])
c2 = np.array([1, -1])
l2 = np.array([2, 2])

DBU_1 = DBU(1, 2, np.expand_dims(l1, axis=0), np.expand_dims(c1, axis=0))
DBU_1.plot_2D(title='Initial DBU')

DBU_2 = DBU(1, 2, np.expand_dims(l2, axis=0), np.expand_dims(c2, axis=0))
DBU_2.plot_2D(title='DBU we remove')

DBU_new = DBU_1.subtract_single_DBUs(DBU_2)
DBU_new.plot_2D(title='Subtracted DBU')                                                                                             

print(DBU_new)

#%%
'''
Tests for single DBUs continued
'''
c1 = np.array([1,1.5])
l1 = np.array([2,3])

c2 = np.array([0,0])
l2 = np.array([1,1])

DBU_1 = DBU(1, 2, np.expand_dims(l1, axis=0), np.expand_dims(c1, axis=0))
DBU_1.plot_2D(title='Initial DBU')

DBU_2 = DBU(1, 2, np.expand_dims(l2, axis=0), np.expand_dims(c2, axis=0))
DBU_2.plot_2D(title='DBU we remove')

DBU_new = DBU_1.subtract_single_DBUs(DBU_2)
DBU_new.plot_2D(title='Subtracted DBU')

#%%
'''
Debugging a case when centres[0] is turning out to be NULL
'''

c1 = np.array([-1.5, -1])
l1 = np.array([1, 4])

c2 = np.array([-1.5, 2.5])
l2 = np.array([1, 1])

DBU_1 = DBU(1, 2, np.expand_dims(l1, axis=0), np.expand_dims(c1, axis=0))
DBU_1.plot_2D(title='Initial DBU')

DBU_2 = DBU(1, 2, np.expand_dims(l2, axis=0), np.expand_dims(c2, axis=0))
DBU_2.plot_2D(title='DBU we remove')

DBU_new = DBU_1.subtract_single_DBUs(DBU_2)
DBU_new.plot_2D(title='Subtracted DBU')

print(f'Old DBU is {DBU_1}')
print(f'Removed DBU is {DBU_2}')
print(f'Subtracted DBU is {DBU_new}')

#%%
'''
Other tests for subtract_DBUs
'''
centres = np.array([[1,1.5]])
lengths = np.array([[2,3]])
DBU_1 = DBU(1, 2, lengths, centres, 0.5)
DBU_1.plot_2D(title = 'First DBU')

new_centres = np.array([[0,0]])
new_lengths = np.array([[1,1]])
DBU_2 = DBU(1, 2, new_lengths, new_centres, 0.5)
DBU_2.plot_2D(title='Second DBU')

subtracted_DBU = DBU_1.subtract_DBUs(DBU_2)
subtracted_DBU.plot_2D(title='Subtracted DBU')

#%%
# Tests when we have that A contains B and we perform B-A

centres = np.array([[0.5, 0]])
lengths = np.array([[1,6]])
DBU_1 = DBU(1, 2, lengths, centres, 0.5)
DBU_1.plot_2D(title = 'First DBU')

new_centres = np.array([[0,0]])
new_lengths = np.array([[6,6]])
DBU_2 = DBU(1, 2, new_lengths, new_centres, 0.5)
DBU_2.plot_2D(title='Second DBU')

subtracted_DBU = DBU_1.subtract_DBUs(DBU_2)
print(subtracted_DBU.no_of_boxes)

#%%
'''
Tests for subtract_DBUs
'''

# A1 - (B_1 \cup B_2) = (A_1 - B_1) - B_2
centres = np.array([[2, 4.5], [1, 1.5], [3, 2.5]])
lengths = np.array([[4, 3], [2, 3], [2, 1]])
DBU_1 = DBU(3, 2, lengths, centres, 0.5)
DBU_1.plot_2D(title='First DBU')

new_centres = np.array([[0,0], [4,4]])
new_lengths = np.array([[1,1], [2,2]])
DBU_2 = DBU(2, 2, new_lengths, new_centres, 0.5)
DBU_2.plot_2D(title='Second DBU')

subtracted_DBU = DBU_1.subtract_DBUs(DBU_2)
subtracted_DBU.plot_2D(title='Subtracted DBU')

# %%
'''
Tests for subtract constraint
'''
c1 = np.array([0, 0])
l1 = np.array([4, 4])
constraint_bounds = np.array([[0,1], [0,1]])

DBU_1 = DBU(1, 2, np.expand_dims(l1, axis=0), np.expand_dims(c1, axis=0))
ax = DBU_1.plot_2D(show_plot=False)


DBU_1.add_disjoint_rectangle(2, np.array([6,1]), np.array([2,2]))
final_ax = DBU_1.plot_2D(ax, show_plot = False)

constraint = cc.ConstraintConditions(2, np.array([0,1]), constraint_bounds)
print('The first constraint is given by')
print(constraint)
print(DBU_1)

final_ax = constraint.plot_2D_constraints(ax, title = 'Plot with initial set of constraints',
                                          show_plot = False)

final_DBU = DBU_1.subtract_constraint(constraint)
print(final_DBU)

final_DBU.plot_2D(title='Plot after removal of constraints set 1', show_plot = True)

new_bounds = np.array([[5.5,6], [1,1.5]])
constraint_2 = cc.ConstraintConditions(2, np.array([0,1]), bounds = new_bounds)
print('The second constraint is given by')
print(constraint_2)

new_ax = constraint_2.plot_2D_constraints(ax=final_ax,
                                          title='Plot with new set of constraints',
                                          show_plot=True)

new_DBU = final_DBU.subtract_constraint(constraint_2)

print(new_DBU)
new_DBU.plot_2D(title = 'Plot after removal of constraints set 2')
#%%
'''
Tests for integrate
'''
def f(x):
    return np.sum(np.array(x)**2) / len(x)

g = lambda x: np.sqrt(np.sum(np.abs(x))) / len(x)

new_DBU = DBU(3, 2, np.array([[4,4], [2,2], [1,3]]),
              np.array([[0,0], [4,4], [7,7]]))

print(f'The integral of f is given by {new_DBU.integrate(f)}')
print(f'The integral of g is given by {new_DBU.integrate(g)}')
new_DBU.plot_2D(title='Initial DBU')
# %%
'''
Tests for append_disjoint_DBUs
'''

DBU_1 = DBU(1, 2, lengths = np.array([[4,4]]), centres = np.array([[0,0]]),
            stepsizes = np.array([[0.5,0.5]]))

DBU_2 = DBU(1, 2, lengths = np.array([[2,2]]), centres = np.array([[5,5]]),
            stepsizes = np.array([[0.5,0.5]]))

new_DBU = DBU_1.append_disjoint_DBUs(DBU_2)

print(new_DBU)
new_DBU.plot_2D(title = 'DBU with two boxes')
# %%
'''
Tests for get_endpoints and get_bounds
'''
# Tests for get_bounds and get_total_bounds

print(new_DBU.get_end_points()[1])
print(f'First box in the new dbu has the following bounds {new_DBU.get_bounds()[0]}')
print(f'Second box in the new dbu has the following bounds {new_DBU.get_bounds()[1]}')

print(f'Total bounds is {new_DBU.get_total_bounds()}')
# %%
'''
Tests for union of DBUs
'''
first_dbu = DBU(1,2,
                lengths = np.array([[2,4]]),
                centres = np.array([[0,0]]),
                stepsizes = 0.5)

first_dbu.plot_2D(title='First DBU plotted')

second_dbu = DBU(1,2,
                 lengths = np.array([[4,5]]),
                 centres = np.array([[-1,-1]]),
                 stepsizes = 0.5)

second_dbu.plot_2D(title='Second DBU plotted')

third_dbu = DBU(1, 2,
                lengths = np.array([[2,2]]),
                centres = np.array([[1,-3.5]]),
                stepsizes = 0.5)

union_dbu = first_dbu.DBU_union(second_dbu)
union_dbu.plot_2D(title='Union of 2 rectangles DBU')

next_union_dbu = union_dbu.DBU_union(third_dbu)
next_union_dbu.plot_2D(title='Union of 3 rectangles in the DBU')

# %%
'''
Tests for generate all conditions

There is a bug, some conditions are repeated and not all conditions are reached
'''

max_complexity = 2
dbu = DBU(2, 2, np.array([[3,3],[4,4]]), np.array([[0,0],[-6,-6]]))
                                                                                    
condition_list = dbu.generate_all_conditions(2)

for i,condition in enumerate(condition_list):
    fig,ax = plt.subplots()
    ax = dbu.plot_2D(ax, show_plot=False)
    condition.plot_2D_constraints(ax=ax, title=f'Constraint {i}',
                                  show_plot=True)


for i,condition in enumerate(condition_list):
    print(f'The {i}th condition is')
    print(condition)

print(f'The length of the condition list is {len(condition_list)}')

#%%
'''
Tests for DBUIterator
'''

dbu = DBU(4, 2,
          lengths = np.array([[2,2], [2, 2], [3,3], [5,6]]),
          centres = np.array([[0,0], [0,-2], [-5,-5], [-5,0]]),
          stepsizes = 0.5)

fig, ax = plt.subplots()
dbu_iter_class = disjoint_box_union.DBUIterator(dbu)
dbu_iterator = iter(dbu_iter_class)

for s in dbu_iterator:
    print(f'The point {s} is given')
    ax.scatter(s[0], s[1], marker = 'x')

ax = dbu.plot_2D(ax, show_plot = True, title = 'DBU with points plotted')

#%%

'''
Tests for find_point_in_DBU
'''

dbu = DBU(4, 2,
          lengths = np.array([[2,2], [2, 2], [3,3], [5,6]]),
          centres = np.array([[0,0], [0,-2], [-5,-5], [-5,0]]),
          stepsizes = 0.5)

fig, ax = plt.subplots()
dbu_iter_class = disjoint_box_union.DBUIterator(dbu)
dbu_iterator = iter(dbu_iter_class)

for s in dbu_iterator:
    print(f'The point {s} is given')
    ax.scatter(s[0], s[1], marker = 'x')

ax = dbu.plot_2D(ax, show_plot = False, title = 'DBU with points plotted')

P = np.array([-2.2, 0])
ax.scatter(P[0], P[1], color='red', label='Point')

closest_point = dbu.find_point_in_DBU(P)
ax.scatter(closest_point[0], closest_point[1], color = 'green', label = 'Closest point in DBU')

plt.legend()
plt.show()
print(f'Closest point is {closest_point}')

#%%
'''
Tests for is_point_in_DBU
'''

dbu = DBU(4, 2,
          lengths = np.array([[2,2], [2, 2], [3,3], [5,6]]),
          centres = np.array([[0,0], [0,-2], [-5,-5], [-5,0]]),
          stepsizes = 0.5)

fig, ax = plt.subplots()
dbu_iter_class = disjoint_box_union.DBUIterator(dbu)
dbu_iterator = iter(dbu_iter_class)

for s in dbu_iterator:
    print(f'The point {s} is given')
    ax.scatter(s[0], s[1], marker = 'x')

ax = dbu.plot_2D(ax, show_plot = False, title = 'DBU with points plotted')

P = np.array([-2.2, 0])
ax.scatter(P[0], P[1], color='red', label='Point')

closest_point = dbu.find_point_in_DBU(P)
ax.scatter(closest_point[0], closest_point[1], color = 'green', label = 'Closest point in DBU')

plt.legend()
plt.show()
print(f'Closest point is {closest_point}')
print(f'Point is in the DBU : {dbu.is_point_in_DBU(P)}')

fig, ax = plt.subplots()
dbu_iter_class = disjoint_box_union.DBUIterator(dbu)
dbu_iterator = iter(dbu_iter_class)

for s in dbu_iterator:
    print(f'The point {s} is given')
    ax.scatter(s[0], s[1], marker = 'x')

ax = dbu.plot_2D(ax, show_plot = False, title = 'DBU with points plotted')

Q = np.array([-5.1, -5.4])
ax.scatter(Q[0], Q[1], color = 'red', label='Point')

closest_point = dbu.find_point_in_DBU(Q)                                        
ax.scatter(closest_point[0], closest_point[1], color = 'green', label = 'Closest point in DBU')

plt.legend()
plt.show()
print(f'Closest point is {closest_point}')
print(f'Point is in the DBU : {dbu.is_point_in_DBU(Q)}')

#%%
'''
Tests for ConditionsIterator 

Check for bugs

'''

max_complexity = 2                                                             
dbu = DBU(2, 2, np.array([[3,3],[4,4]]), np.array([[0,0],[-6,-6]]))

fig,ax = plt.subplots()
ax = dbu.plot_2D(ax, show_plot = False)

cond_iter_class = disjoint_box_union.ConditionsIterator(dbu, max_complexity, 3.0)
cond_iterator = iter(cond_iter_class)

all_conditions = []
for i,c in enumerate(cond_iterator):
    if c != None:
        print(f'For condition number {i}, we print the following condition \n')
        print(c)
        ax = c.plot_2D_constraints(ax, show_plot = True, title = f'Plotting constraint {i}')
        all_conditions.append(c)
        fig, ax = plt.subplots()
        ax = dbu.plot_2D(ax, show_plot = False)

for i,c in enumerate(all_conditions):
    print(f'The {i}th condition is given by')
    print(c)
    
#%%
'''
Tests for conditions Iterator - 2 box case
'''

max_complexity  = 2
dbu = DBU(2, 2, np.array([[4,4],[3,3]]), np.array([[0,0], [4,2]]))

fig,ax = plt.subplots()
ax = dbu.plot_2D(ax, show_plot = False)
cond_iter_class = disjoint_box_union.ConditionsIterator(dbu, max_complexity)
cond_iterator = iter(cond_iter_class)

all_conditions = []
for i,c in enumerate(cond_iterator):
    if c != None:
        print(f'For condition number {i}, we print the following condition \n')
        print(c)
        ax = c.plot_2D_constraints(ax, show_plot = True, title = f'Plotting constraint {i}')
        all_conditions.append(c)
        fig, ax = plt.subplots()
        ax = dbu.plot_2D(ax, show_plot = False)

for i,c in enumerate(all_conditions):
    print(f'The {i}th condition is given by')
    print(c)  

#%%
'''
Tests for condition iterator - 1 box case
'''
max_complexity = 1
dbu = DBU(1, 2, np.array([[4,4]]), np.array([[-1,0]]))

fig,ax = plt.subplots()
ax = dbu.plot_2D(ax, show_plot = False)
cond_iter_class = disjoint_box_union.ConditionsIterator(dbu, max_complexity)
cond_iterator = iter(cond_iter_class)

all_conditions = []
for i,c in enumerate(cond_iterator):
    if c != None:
        print(f'For condition number {i}, we print the following condition \n')
        print(c)
        ax = c.plot_2D_constraints(ax, show_plot = True, title = f'Plotting constraint {i}')
        all_conditions.append(c)
        fig, ax = plt.subplots()
        ax = dbu.plot_2D(ax, show_plot = False)

for i,c in enumerate(all_conditions):
    print(f'The {i}th condition is given by')
    print(c)  


#%%
'''
Tests for condition iterator 3 dimension, 2 boxes
'''
max_complexity = 3
dbu = DBU(2, 3, np.array([[4,4,2], [2,2,2]]), np.array([[-1,0,-1], [-3,-3,-3]]))

fig,ax = plt.subplots()
cond_iter_class = disjoint_box_union.ConditionsIterator(dbu, max_complexity)
cond_iterator = iter(cond_iter_class)

all_conditions = []
for i,c in enumerate(cond_iterator):
    if c != None:
        print(f'The {i}th condition is given by')
        print(c)  

#%%
'''
Tests for condition iterator 4 dimension, 2 boxes
'''
max_complexity = 3
dbu = DBU(1, 4, np.array([[4,4,5,2]]), np.array([[-1,0,0,-2]]))

fig,ax = plt.subplots()
ax = dbu.plot_2D(ax, show_plot = False)
cond_iter_class = disjoint_box_union.ConditionsIterator(dbu, max_complexity)
cond_iterator = iter(cond_iter_class)

all_conditions = []
for i,c in enumerate(cond_iterator):
    print(f'The {i}th condition is given by')
    print(c)

#%%
max_complexity = 2
dbu = DBU(2, 2, np.array([[3,3],[4,4]]), np.array([[0,0],[-6,-6]]))

condition_list = dbu.generate_all_conditions(2)

for i,condition in enumerate(condition_list):
    fig,ax = plt.subplots()
    ax = dbu.plot_2D(ax, show_plot=False)
    condition.plot_2D_constraints(ax=ax, title=f'Constraint {i}',                                           
                                  show_plot=True)                               

print('Final bounds are')
print(dbu.get_total_bounds())

print('Bounds are')
print(dbu.get_bounds())
#%%

for i,condition in enumerate(condition_list):
    print(f'The {i}th condition is:')
    print(condition)
#%%

p = dbu.sample_from_dbu()

fig, ax = plt.subplots()
ax = dbu.plot_2D(ax, show_plot = False, title = 'DBU with points plotted')

ax.scatter(p[0], p[1], marker = 'x', label = 'Sampled point 1')
plt.legend()
plt.show()


q = dbu.sample_from_box(1)
fig, ax = plt.subplots()
ax = dbu.plot_2D(ax, show_plot = False, title = 'DBU with points plotted')

ax.scatter(q[0], q[1], marker = 'x', label = 'Sampled point 2')
plt.legend()

plt.show()
#%%

'''
Condition to DBU
'''
dimension = 2
indices = np.array([0,1])
bounds = np.array([[-2, 4], [4, 5]])
constraint = cc.ConstraintConditions(dimension, indices, bounds)
constraint.plot_2D_constraints()


dbu = disjoint_box_union.DisjointBoxUnion.condition_to_DBU(constraint,
                                                           stepsizes = 0.5)

dbu.plot_2D(title='DBU version of constraint')
print(dbu)
#%%

'''
Test DBU subtraction
'''

dbu1 = DBU(1, 3, np.array([[1,5,5]]), np.array([[-2, 0, 0]]))
print(dbu1)
dbu2 = DBU(1, 3, np.array([[5,5,5]]), np.array([[0,0,0]]))
print(dbu2)

sub_DBU = dbu1.subtract_DBUs(dbu2)

print(sub_DBU)

#%%

dbu1 = DBU(1, 3, np.array([[1,4,4]]), np.array([[-2, 0.5, 0.5]]))
print(dbu1)

dbu2 = DBU(1, 3, np.array([[1,5,5]]), np.array([[-2, 0, 0]]))
print(dbu2)

sub_DBU = dbu2.subtract_DBUs(dbu1)

print(sub_DBU)

#%%
'''
Sampling integrate static tests
'''

f = lambda x : np.sqrt(x[0]**2 + x[1]**2 + 2.5)

print(DBU.sampling_integrate(sub_DBU, f, point_percentages=0.5))
print(DBU.sampling_integrate(sub_DBU, f, point_percentages=0.8))
print(DBU.sampling_integrate(dbu2, f, point_percentages = 0.75))

print(DBU.integrate_static(dbu1, f))
print(DBU.sampling_integrate_static(dbu1, f))

#%%
'''
Sampling integration functional
'''
f = lambda x : np.sqrt(x[0]**2 + x[1]**2 + 2.5)
percent = 0.8
integral_functional = lambda dbu, f, perc : DBU.sampling_integrate(dbu, f, percent)
print(integral_functional(dbu1, f, percent))
print(integral_functional(sub_DBU, f, percent))
print(integral_functional(dbu2, f, percent))

print(dbu1)
print(dbu1.get_end_points()[0].shape)

#%%
'''
Easy integral
'''
f = lambda x : np.sqrt(x[0]**2 + x[1]**2 + 2.5)
print(DBU.easy_integral(dbu1, f))                                               
print(DBU.easy_integral(dbu2, f))
print(DBU.easy_integral(sub_DBU, f))

print(dbu1)

#%%
'''
Dynamic programming based integral
'''
dbu1 = DBU(1, 3,
           np.array([[1,4,4]]),
           np.array([[-2, 0.5, 0.5]]))


f = lambda x : np.sqrt(x[0]**2 + x[1]**2 + 1) 
print(DBU.dp_integrate(dbu1, f))


#%%
'''
Computation times for different methods
'''
import time

# Define some test functions for integration
def test_function_1(x):
    return np.sum(x ** 2)  # Simple quadratic function

def test_function_2(x):
    return np.sin(np.sum(x))  # Sum of sines

def test_function_3(x):
    return np.prod(np.cos(x))  # Product of cosines

# List of test functions
test_functions = [test_function_1, test_function_2, test_function_3]
function_names = ['Quadratic', 'Sum of Sines', 'Product of Cosines']

# Function to evaluate integration methods and plot the results
def plot_integration_comparison(dbu_instance):
    methods = ['Sampling Integrate', 'Regular Integrate', 'DP Integrate']
    results = {name: [] for name in function_names}  # Store results for each function
    times = {name: [] for name in function_names}  # Store computation times

    # Iterate over test functions
    for func, func_name in zip(test_functions, function_names):
        # Sampling Integrate
        start = time.time()
        sampling_result = dbu_instance.sampling_integrate(func, point_percentages=0.5)
        end = time.time()
        results[func_name].append(sampling_result)
        times[func_name].append(end - start)

        # Regular Integrate
        start = time.time()
        regular_result = dbu_instance.integrate(func)
        end = time.time()
        results[func_name].append(regular_result)
        times[func_name].append(end - start)

        # DP Integrate
        start = time.time()
        dp_result = dbu_instance.dp_integrate(func)
        end = time.time()
        results[func_name].append(dp_result)
        times[func_name].append(end - start)

    # Plotting the integration results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot for integration results
    x = np.arange(len(function_names))
    width = 0.2
    for i, method in enumerate(methods):
        ax1.bar(x + i * width, [results[func_name][i] for func_name in function_names],            
                width=width, label=method)                                                       
                                                                                
    ax1.set_xlabel("Test Functions")
    ax1.set_ylabel("Integration Result")
    ax1.set_title("Integration Results Comparison")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(function_names)
    ax1.legend()

    # Bar plot for computation times
    for i, method in enumerate(methods):
        ax2.bar(x + i * width, [times[func_name][i] for func_name in function_names],
                width=width, label=method)

    ax2.set_xlabel("Test Functions")
    ax2.set_ylabel("Computation Time (seconds)")
    ax2.set_title("Computation Time Comparison")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(function_names)
    ax2.legend()

    plt.tight_layout()
    plt.show()


#%%