import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import unittest
import importlib
import copy
import itertools                                                               
import numpy as np                                                             
import unittest
import multiprocessing as mp
import tqdm
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from functools import partial


#%%

class TreeNode:
    
    """
    Represents a node in the binary decision tree,
    which may contain split criteria or be a leaf with an action.
    """
    
    def __init__(self, dimensions=None, values=None,
                 greater_than=None, action=None):
        
        '''                                                                    
        Data structure to represent a node in a decision tree; each node comes  
        with a splitiing condition or is a leaf with a corresponding action    
                                                                                
        Parameters:                                                            
        -----------------------------------------------------------------------
        dimensions : list or np.array                                          
                     The dimensions that are involved in the splitting condition
                                                                                
        values : list or np.array                                               
                 The values of the node we are splitting on                    
                 
  greater_than : list or np.array                                              
                 List of True/False booleans that says whether the condition   
                 for that dimension is True or False                           
                 
        action : Action                                                        
                 The action that is it to be taken if we are on a leaf node    
        
        '''
        
        
        if action is None:                                                     
            assert dimensions is not None and values is not None and greater_than is not None, \
                "Non-leaf nodes must have dimensions, values, and greater_than defined."
            assert len(dimensions) == len(values) == len(greater_than), \
                "Dimensions, values, and greater_than must all have the same length."
        
        self.dimensions = dimensions or []     # List of dimensions for splitting
        self.values = values or []             # Threshold values for each dimension
        self.greater_than = greater_than or [] # List of bools for each dimension split condition
        self.action = action                   # Action if this is a leaf node
        self.left = None                       # Left child node
        self.right = None                      # Right child node

                                                                                
    def is_leaf(self):
        
        """
        Check if this node is a leaf node.
        """
        
        return self.action is not None


    def attach_left(self, node):
        
        """
        Attach a node as the left child, ensuring it meets the attachment
        conditions.
        
        If the new node satisfies conditions for left attachment attach it to
        the left.
        
        Parameters:                                                            
        -----------------------------------------------------------------------
        node : TreeNode or DecisionTree                                                   
               The node we wish to attach to the left of the current node;      
        """
        if self.left is not None:
            raise ValueError("Left node already exists.") 
        
        if (type(node) == type(DecisionTree(root=TreeNode))):
            self.left = node.root
            return
        
        if node.is_leaf():
            self.left = node
            return
        
        if self.is_leaf():
            raise ValueError("Cannot attach a node to a leaf node.")            
        
        if not self._satisfies_conditions(node):                  
                                                                                
            print(f'We try to attach {self} to {node} on the left and this gives an error')
                                                                                 
            raise ValueError("Node does not satisfy conditions and connot be used for left attachment.")
                                                                                            
        self.left = node
                                                    
        
    def attach_right(self, node):
        
        """
        Attach a node as the right child, ensuring it meets the attachment
        conditions.
        
        If the new node does not satisfy conditions for attachment attach it
        to the right. 
        
        Parameters:
        -----------------------------------------------------------------------
        node : TreeNode                                                        
               The node we wish to attach to the right of the current node     
                                                                               
        """
        if self.right is not None:
            raise ValueError("Right node already exists.") 
        
        if (type(node) == type(DecisionTree(root=TreeNode))):
            self.right = node.root
            return
        
        if node.is_leaf():
            self.right = node
            return
        
        if self.is_leaf():
            raise ValueError("Cannot attach a node to a leaf node.")
        if self.right is not None:
            raise ValueError("Right node already exists.")
        if self._satisfies_conditions(node):
            raise ValueError("Node satisfies conditions and cannot be attached to the right.")
        self.right = node

    
    def copy_node(self):
        
        '''
        Return a node with the same parameters as the original node.
        
        '''
        new_node = TreeNode(dimensions = self.dimensions,
                            values = self.values,
                            greater_than = self.greater_than,
                            action = self.action)
        
        return new_node


    def _satisfies_conditions(self, node):
        
        """
        Check if the node satisfies or does not satisfy the current node's
        conditions for attachment.
                                                                                                    
        Parameters:
        -----------------------------------------------------------------------
        node : TreeNode                                                        
               The node we wish to check whether it satisfies conditions
        
        satisfy : True/False
                  Check for satistification of the conditons versus the latter.
        
        """
        
        for dim, val, gt in zip(self.dimensions, self.values, self.greater_than):
            node_value = node.values[dim]
            if ((gt and node_value > val) or (not gt and node_value <= val)):
                continue
            else:
                return False
        
        return True


    def __repr__(self):                                                         
        if self.is_leaf():                                                      
            return f"Leaf(action={self.action})"
        conditions = [
            f"(dim={d}, val={v}, {'>' if gt else '<='})"
            for d, v, gt in zip(self.dimensions, self.values, self.greater_than)
        ]
        return f"TreeNode({', '.join(conditions)})"



#%%                                                                      

def dimension_check(args):
    """
    Static method for parallel evaluation of a dimension condition in the tree.

    Parameters:
    -----------------------------------------------------------------------
    args : tuple
        (s, d, node_val, greater_than_flag)

    Returns:
    -----------------------------------------------------------------------
    bool : True if the dimension fails the condition (go right), False otherwise
    """
    s, d, node_val, greater_than_flag = args
    left_side = s[d] if isinstance(s, (list, np.ndarray)) else s

    if greater_than_flag:
        return left_side <= node_val  # Condition fails, must go right
    else:
        return left_side > node_val   # Condition fails, must go right

def build_subtree(current_depth, max_depth, split_combinations, action_set):
    if current_depth == max_depth:
        return [TreeNode(action=action) for action in action_set]

    all_trees = []
    for dimensions, values, greater_than in split_combinations:
        root_node = TreeNode(dimensions=dimensions, values=values, greater_than=greater_than)

        left_subtrees = build_subtree(current_depth + 1, max_depth, split_combinations, action_set)
        right_subtrees = build_subtree(current_depth + 1, max_depth, split_combinations, action_set)

        for left in left_subtrees:
            for right in right_subtrees:
                new_root = TreeNode(dimensions=root_node.dimensions,
                                    values=root_node.values,
                                    greater_than=root_node.greater_than)
                try:
                    new_root.attach_left(left)
                    new_root.attach_right(right)
                    all_trees.append(new_root)
                except ValueError:
                    pass
    return all_trees


def build_tree_at_node(args):
    """
    Worker function for multiprocessing: builds a tree given a specific split combination.
    """
    depth, max_depth, split, action_set, split_combinations = args
    dimensions, values, greater_than = split

    if depth == max_depth:
        return [DecisionTree(root=TreeNode(action=action)) for action in action_set]

    all_trees = []
    root_node = TreeNode(dimensions=dimensions, values=values, greater_than=greater_than)

    # Recursive serial build
    left_subtrees = build_subtree(depth + 1, max_depth, split_combinations, action_set)
    right_subtrees = build_subtree(depth + 1, max_depth, split_combinations, action_set)

    for left in left_subtrees:
        for right in right_subtrees:
            new_root = TreeNode(dimensions=root_node.dimensions,
                                values=root_node.values,
                                greater_than=root_node.greater_than)
            try:
                new_root.attach_left(left)
                new_root.attach_right(right)
                all_trees.append(DecisionTree(root=new_root))
            except ValueError:
                pass
    return all_trees

#%%

class DecisionTree:                                                             
    """
    Represents the binary decision tree and manages insertion, traversal,      
    and leaf node handling, including generation of all possible trees up to    
    a maximum depth and complexity.

    ---------------------------------------------------------------------------
    root : TreeNode
           The root of the decision tree
    
    """
    
    def __init__(self, root=None):
                                                                                
        self.root = root   # Root node of the tree                             
                                                                                             
    def traverse_in_order(self):                                                
        
        """                                                                    
        Traverse the tree in-order and return a list of nodes.
        """                                                                    
        
        nodes = []
        self._traverse_in_order_recursive(self.root, nodes)
        return nodes                                                           

    def _traverse_in_order_recursive(self, node, nodes):
        
        """
        Helper function for in-order traversal.
        
        """
                                                                                
        if node is not None:                                                   
            nodes.append(node)
            self._traverse_in_order_recursive(node.left, nodes)                 
            self._traverse_in_order_recursive(node.right, nodes)                

    def save_tree_as_png(self, filename="decision_tree.png"):
        
        """
        Save the tree structure as a .png file using matplotlib
        and networkx.
        """                                                                     
        
        if self.root is None:
            raise ValueError("Tree is empty.")
                                                                                
        G = nx.DiGraph()
        self._add_edges(self.root, G)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue",
                font_size=10, font_weight="bold", arrows=True)
        plt.savefig(filename)
        plt.close()

    def _add_edges(self, node, graph, parent_id=None, direction=None):
        
        """
        Helper function to add nodes and edges recursively to the graph.
        """
        
        label = repr(node)
        node_id = label
        graph.add_node(node_id, label=label)
        if parent_id is not None:
            graph.add_edge(parent_id, node_id, label=direction)

        if node.left:
            self._add_edges(node.left, graph, node_id, "left")                 
        if node.right:                                                          
            self._add_edges(node.right, graph, node_id, "right")                
            
    
    def print_tree(self, root, level=0, prefix="Root: "):
        #Works for 1 dimensions
        if root:
            print(" " * (level * 4) + prefix + str(root.values[0]))
            if root.left or root.right:
                self.print_tree(root.left, level + 1, "L--- ")
                self.print_tree(root.right, level + 1, "R--- ")

    @staticmethod
    def generate_tree_function(tree):
        """
        Generates a callable function that evaluates the decision tree.

        Parameters:
        -----------------------------------------------------------------------
        tree : DecisionTree
                The decision tree to generate the function from.

        Returns:
            -----------------------------------------------------------------------
        tree_function : function
                        Callable function mapping state 's' to an action.
        """
        def tree_function(s, tree=tree):
            """
            Inner function that evaluates the tree on state s.
            """
            node = tree.root
            pool = mp.get_context('fork').Pool(processes=mp.cpu_count())
            try:
                while not node.is_leaf():
                    tasks = [(s, d, node.values[i], node.greater_than[i])
                             for i, d in enumerate(node.dimensions)]

                    results = pool.map(dimension_check, tasks)

                    if any(results):
                        node = node.right
                    else:
                        node = node.left
                return node.action
            finally:
                pool.close()
                pool.join()
        return tree_function
    
    @staticmethod
    def _generate_combinations_for_complexity(args):
       """
       Helper method for parallel computation of split combinations for a specific complexity level.
       """
       complexity, bounds, state_differences = args
       split_combinations = []
       dimensions = range(len(bounds))
       dim_sets = itertools.combinations(dimensions, complexity)

       for dim_set in dim_sets:
           value_sets = (
               np.arange(bounds[d, 0] + state_differences[d], bounds[d, 1], state_differences[d])
               for d in dim_set
           )
           for values in itertools.product(*value_sets):
               for conditions in itertools.product([True, False], repeat=complexity):
                   split_combinations.append((list(dim_set), list(values), list(conditions)))

       return split_combinations


    @staticmethod
    def generate_split_combinations(max_complexity, bounds, state_differences):
        """
        Generate all possible split combinations for bounds with state_differences
        and max_complexity.
        """
        split_combinations = []
        dimensions = range(len(bounds))
        bounds = np.array(bounds)
    
        if isinstance(state_differences, (float, int)):
            state_differences = [state_differences for _ in range(len(bounds))]
    
        for complexity in range(1, max_complexity + 1):
            for dim_set in itertools.combinations(dimensions, complexity):
                value_sets = (np.arange(bounds[d, 0] + state_differences[d],
                                        bounds[d, 1], state_differences[d]) for d in dim_set)
                for values in itertools.product(*value_sets):
                    for conditions in itertools.product([True, False], repeat=complexity):
                        split_combinations.append((list(dim_set), list(values), list(conditions)))
        return split_combinations

    @staticmethod
    def _build_tree_from_split(split, max_depth, action_set):
        """
        Safely build tree from a given split.
        """
        try:
            dimensions, values, greater_than = split
            root_node = TreeNode(dimensions=dimensions, values=values, greater_than=greater_than)
    
            def build_subtree(depth):
                if depth == max_depth:
                    return [TreeNode(action=action) for action in action_set]
                all_subtrees = []
                left_subtrees = build_subtree(depth + 1)
                right_subtrees = build_subtree(depth + 1)
    
                for left in left_subtrees:
                    for right in right_subtrees:
                        new_node = TreeNode(dimensions=root_node.dimensions,
                                            values=root_node.values,
                                            greater_than=root_node.greater_than)
                        new_node.attach_left(left)
                        new_node.attach_right(right)
                        all_subtrees.append(DecisionTree(root=new_node))
                return all_subtrees
    
            return build_subtree(1)
    
        except ValueError:
            return []  # Ignore invalid trees}

    @staticmethod
    def generate_all_trees(max_depth, max_complexity, bounds, state_differences, action_set):
        """
        Parallel generation of all possible decision trees up to a given depth and complexity.
        """
        if isinstance(state_differences, (float, int)):
            state_differences = [state_differences for _ in range(len(bounds))]
    
        split_combinations = DecisionTree.generate_split_combinations(max_complexity, bounds, state_differences)
    
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = []
            for res in tqdm(
                    pool.imap_unordered(
                        partial(DecisionTree._build_tree_from_split,
                        max_depth=max_depth,
                        action_set=action_set),
                split_combinations),
                total=len(split_combinations),
                desc="Generating Trees"):
                results.extend(res)
    
        return results


#%%
'''
# Run the tests

if __name__ == "__main__":                                                      
    
    rootie = TreeNode(dimensions=[0],                                            
                      values = [10.0],                                              
                      greater_than = [False])                                     
                                                                                
    print(f'Root is {rootie}')                                                    
                                                                                
    tree = DecisionTree(root = rootie)                                           
                                                                                     
    left_node = TreeNode(dimensions=[0], values=[5], greater_than=[True])      
    right_node = TreeNode(dimensions=[0], values=[20], greater_than=[False])     
    leaf_node_up = TreeNode(action='up')                                       
    leaf_node_down = TreeNode(action='down')                                   
                                                                                                            
    leaf_node_left = TreeNode(action='left')                                        
    leaf_node_right = TreeNode(action='right')                                 
                                                                                        
    tree.root.attach_left(left_node)                                           
    tree.root.attach_right(right_node)                                         
    
    tree.root.left.attach_left(leaf_node_up)                                   
    tree.root.left.attach_right(leaf_node_down)                                 
    
    tree.root.right.attach_left(leaf_node_left)                                
    tree.root.right.attach_right(leaf_node_right)                              
    
    nodes = tree.traverse_in_order()                                            
    for i,node in enumerate(nodes):
        print(f'{i}th node is')
        print(node)                                                            
    
    tree.save_tree_as_png('Saved_tree_1.png')
    #tree.print_tree(root=rootie)

#%%%

    nodes = tree.traverse_in_order()
    print('Nodes after traversing are')
    print(nodes)
    
#%%%                                                                            
    
    new_tree_function = tree.generate_tree_function(tree)                       
    
#%%
    
    print(f'At {5.2} we have value {new_tree_function(5.2)}')
    print(f'At {10.0} we have value {new_tree_function(10.0)}')
    print(f'At [-1.4] we have value {new_tree_function([-1.4])}')
    print(f'At {np.array([-3.4])} we have value {new_tree_function(np.array([-3.4]))}')
    print(f'At {-np.array([19])} we have value {new_tree_function(np.array([-19]))}')
    
    #%%                                                                         
    
    rootie = TreeNode(dimensions = [0], values = [3], greater_than=[True])
    tree = DecisionTree(root=rootie)
    #tree.insert_root(dimensions)
    
    left_leaf = TreeNode(action="left")                                         
    right_leaf = TreeNode(action="right")                                                                        
    
    #tree.root.left.attach_left(left_leaf)  # This node becomes a leaf         
    tree.root.attach_right(right_leaf)                                    
    
    #tree.root.right.attach_left(left_leaf)                                      
    tree.root.attach_left(left_leaf)  # This node becomes a leaf       
    
    nodes = tree.traverse_in_order()                                           
    for node in nodes:                                                          
        print(node)  # Expected order: left_leaf, left_node, root, right_node, right_leaf
                                                                                
    tree.save_tree_as_png("saved_tree_1.png")                                     
    print("Tree saved as example_tree.png")                                     
    
    #tree.print_tree(root=rootie)
                                                                     
#%%                                                                            

# Debug and test generate_split_conditions
tree.generate_split_combinations(max_complexity=1,
                                 bounds=np.array([[-15,15]]),
                                 state_differences=np.array([1.0]))

#%%%
# Debug and test generate_all_trees

bounds = np.array([[0, 4], [0, 4]])  # 2D space: x ∈ [0,10], y ∈ [0,5]
state_differences = np.array([1, 1])  # Step size for each dimension
action_set = ["up", "down"]  # Example set of actions

all_trees = DecisionTree.generate_all_trees(max_depth=3, max_complexity=2,
                                            bounds=bounds,
                                            state_differences=state_differences,
                                            action_set=action_set)
print(f"Generated {len(all_trees)} trees.")

#%%
for i,tree in enumerate(all_trees):
    print(tree.traverse_in_order())
'''