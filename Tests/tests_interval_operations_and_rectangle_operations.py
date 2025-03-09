#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 6 17:25:39 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import interval_and_rectangle_operations as op

from importlib import reload
from disjoint_box_union import DisjointBoxUnion as DBU

op = reload(op)
#%%
'''
Tests for intersect intervals
'''
# Disjoint 1
c1 = 2.0
l1 = 2.0
c2 = -1.0
l2 = 2.0
print((op.intersect_intervals(c1, l1, c2, l2)) == ((c1+c2)/2, 0.0))

# Disjoint 2
c1 = -2.0
l1 = 1.0
c2 = 2.0
l2 = 1.0
print((op.intersect_intervals(c1, l1, c2, l2)) == ((c1+c2)/2, 0.0))

# Partial Intersection 1
c1 = -1.0
l1 = 3.0
c2 = 0.5
l2 = 1.0
print((op.intersect_intervals(c1, l1, c2, l2)) == (0.25, 0.5))

# Partial Intersection 2
c1 = 2.0
l1 = 4.0
c2 = -1.0
l2 = 4.0
print((op.intersect_intervals(c1, l1, c2, l2)) == (0.5, 1))

# Second interval inside first interval
c1 = 0.0
l1 = 4.0
c2 = 1.0
l2 = 2.0
print((op.intersect_intervals(c1, l1, c2, l2)) == (1.0, 2.0))

# First interval inside the second interval
c1 = 0.0
l1 = 2.0
c2 = 10.0
l2 = 30.0
print((op.intersect_intervals(c1, l1, c2, l2)) == (0.0, 2.0))
#%%
'''
Tests for subtract intervals
'''
# Disjoint 1
c1 = 2.0
l1 = 2.0
c2 = -1.0
l2 = 2.0
print((op.subtract_intervals(c1, l1, c2, l2)) == [(c1, l1)])

# Disjoint 2
c1 = -2.0
l1 = 1.0
c2 = 2.0
l2 = 1.0
print((op.subtract_intervals(c1, l1, c2, l2)) == [(c1, l1)])

# Partial Intersection 2
c1 = 2.0
l1 = 4.0
c2 = -1.0
l2 = 4.0
print((op.subtract_intervals(c1, l1, c2, l2)) == [(2.5, 3.0)])

# Second interval inside the first interval
c1 = 0.0
l1 = 10.0
c2 = 0.0
l2 = 4.0
print(op.subtract_intervals(c1, l1, c2, l2) == [(-3.5, 3), (3.5, 3)])

# First interval inside the second interval
c1 = 0.0
l1 = 2.0
c2 = 10.0
l2 = 30.0
print((op.subtract_intervals(c1, l1, c2, l2)) == [(1.0, 0.0)])

# Partial Intersection 1
c1 = -1.0
l1 = 3.0
c2 = 0.5
l2 = 1.0
print((op.subtract_intervals(c1, l1, c2, l2)) == [(-1.25, 2.5)])

#%%
'''
Tests for unions of intervals
'''
# Tests 1 - complete union
centres = np.array([0, 5, 8.5])
lengths = np.array([2, 4, 3])
additional_centre = 4.0
additional_length = 8.0

op.plot_intervals(centres, lengths, title = 'Intervals before merging')

new_centres, new_lengths = op.union_of_intervals(centres, lengths,
                                                 additional_centre,
                                                 additional_length,
                                                 glue = True)
op.plot_intervals(new_centres, new_lengths, title='Complete Union')

# Tests 2 - Last interval gets merged
centres = np.array([0, 4])
lengths = np.array([2,2])
additional_centre = 5.0
additional_length = 2.0

op.plot_intervals(centres, lengths, title = 'Intervals before merging')

new_centres, new_lengths = op.union_of_intervals(centres, lengths,
                                                 additional_centre,
                                                 additional_length,
                                                 glue=True)

op.plot_intervals(new_centres, new_lengths, title = 'Last interval gets merged')

# Tests 3 - New interval inside the given partition
centres = np.array([0,4])
lengths = np.array([4,2])
additional_centre = 0.0
additional_length = 2.0

op.plot_intervals(centres, lengths, title = 'Intervals before merging')

new_centres, new_lengths = op.union_of_intervals(centres, lengths,
                                                 additional_centre,
                                                 additional_length,
                                                 glue=False)

op.plot_intervals(new_centres, new_lengths, title = 'New interval inside partition')

# Tests 4 - We add a disjoint member to the partition
centres = np.array([0,4])
lengths = np.array([4,2])
additional_centre = 7.0
additional_length = 2.0

op.plot_intervals(centres, lengths, title = 'Intervals before merging')

new_centres, new_lengths = op.union_of_intervals(centres, lengths,
                                                 additional_centre,
                                                 additional_length,
                                                 glue=False)

op.plot_intervals(new_centres, new_lengths, title = 'Disjoint interval added to the partition')

#%%
'''
Tests for glue intervals
'''
#No gluing
centres = np.array([-1, 4, 8])
lengths = np.array([2, 4, 2])
op.plot_intervals(centres, lengths, title='Intervals before gluing')
glued_centres, glued_lengths = op.glue_intervals(centres, lengths)
op.plot_intervals(glued_centres, glued_lengths, title='Intervals after gluing')

#Complete gluing
centres = np.array([0, 2, -2])
lengths = np.array([4, 2, 2])
op.plot_intervals(centres, lengths, title='Intervals before gluing')
glued_centres, glued_lengths = op.glue_intervals(centres, lengths)
op.plot_intervals(glued_centres, glued_lengths, title='Intervals after gluing')

#One gluing
centres = np.array([0, 4])
lengths = np.array([4, 4])
op.plot_intervals(centres, lengths, title = 'Intervals before gluing')
glued_centres, glued_lengths = op.glue_intervals(centres, lengths)
op.plot_intervals(glued_centres, glued_lengths, title='Intervals after gluing')
#%%
'''
Tests for subtract rectangles
'''

c1 = np.array([0, 0])
l1 = np.array([5, 5])
c2 = np.array([0.5, 0.5])
l2 = np.array([4, 4])

new_rectangle_list = op.subtract_rectangles(c1, l1, c2, l2)
for i in range(len(new_rectangle_list)):
    print(f'Rectangle {i} is given with centre = {new_rectangle_list[i][0]}, and length = {new_rectangle_list[i][1]}')

lengths = np.array([[1,5],
                    [4,1]])

centres = np.array([[-2,0],
                    [0.5,-2]])

old_DBU = DBU(1, 2, np.expand_dims(l1, axis=0), np.expand_dims(c1, axis=0))
old_DBU.plot_2D(title='DBU before subtraction')

new_DBU = DBU(2, 2, lengths, centres)
new_DBU.plot_2D(title='The new plot')
#%%
'''
New tests for subtract rectangles 
'''
c1 = np.array([0,0])
l1 = np.array([5,5])

c2 = np.array([0.5, 0.5])
l2 = np.array([4, 4])

new_rectangle_list = op.subtract_rectangles(c1, l1, c2, l2)
print(new_rectangle_list)

