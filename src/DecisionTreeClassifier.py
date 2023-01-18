import numpy as np
import pandas as pd
import sklearn.datasets

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        # the feature used to make the decision at this node
        self.feature = feature
        # the threshold value used to make the decision at this node
        self.threshold = threshold
        # the left child node of the current node
        self.left = left
        # the right child node of the current node
        self.right = right
        # the label assigned to this node (leaf node)
        self.label = label
        
class DecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2):
        self.root = None # Initialize the root node as None
        self.max_depth = max_depth # Maximum depth of the tree
        self.min_samples_split = min_samples_split # Minimum number of samples required to split a node
        
    def grow_tree(self, X, y):
        self.root = self._grow_tree(X, y, depth=0) # Grow the tree starting from the root node
        
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape # Get the number of samples and features
        n_labels = len(np.unique(y)) # Get the number of unique labels
        
        # If all samples belong to the same class or depth exceeds the maximum
        if n_labels == 1 or (self.max_depth is not None and depth >= self.max_depth):
            label = np.argmax(np.bincount(y)) # Assign the most common label as the label for this leaf node
            return Node(label=label)
        
        # If there are not enough samples to split
        if n_samples < self.min_samples_split:
            label = np.argmax(np.bincount(y)) # Assign the most common label as the label for this leaf node
            return Node(label=label)
        
        # Find the best feature and threshold to split the data
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)
        
        # If one of the split is empty, create a leaf node
        if len(y_left) == 0 or len(y_right) == 0:
            label = np.argmax(np.bincount(y)) # Assign the most common label as the label for this leaf node
            return Node(label=label)
        
        # Recursively grow the left and right subtrees
        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        best_feature = None # Initialize the best feature as None
        best_threshold = None # Initialize the best threshold as None
        best_gini = float('inf') # Initialize the best gini index as infinity
        for feature in range(X.shape[1]): # Iterate through all features
            thresholds = np.unique(X[:, feature]) # Get all unique thresholds for the current feature
            for threshold in thresholds: 
                gini = self._gini_index(X[:, feature], y, threshold) # Calculate the gini index for the current feature and threshold
                if gini < best_gini:  # If the current gini index is smaller than the current best gini index
                    best_feature = feature # Update the best feature
                    best_threshold = threshold # Update the best threshold
                    best_gini = gini # Update the best gini index
        return best_feature, best_threshold
    
    def _split_data(self, X, y, feature, threshold):
        if X[feature].dtype == 'category':
            left_idx = X[:, feature] == threshold # Get all indices for samples in the left split
            right_idx = X[:, feature] != threshold # Get all indices for samples in the right split
        else:
            left_idx = X[:, feature] <= threshold # Get all indices for samples in the left split
            right_idx = X[:, feature] > threshold # Get all indices for samples in the right split
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx] # Return the samples for the left and right splits
    
    def _gini_index(self, X, y, threshold):
        if X.dtype == 'category':
            y_left = y[X == threshold] # Get the labels for samples in the left split
            y_right = y[X != threshold] # Get the labels for samples in the right split
        else:
            y_left = y[X <= threshold] # Get the labels for samples in the left split
            y_right = y[X > threshold] # Get the labels for samples in the right split
        n_samples = len(y) 
        n_samples_left = len(y_left)
        n_samples_right = len(y_right)
        gini_left = 1 - sum((np.bincount(y_left) / n_samples_left) ** 2) # Calculate the gini index for the left split
        gini_right = 1 - sum((np.bincount(y_right) / n_samples_right) ** 2) # Calculate the gini index for the right split
        weight = n_samples_left / n_samples # Calculate the weight for the left split
        gini = weight * gini_left + (1 - weight) * gini_right # Calculate the weighted average gini index
        return gini
    
    def fit(self, X, y):
        self.grow_tree(X, y)
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X]) # make predictions for all samples in X
    
    def _predict(self, x):
        node = self.root # start from the root node
        while node.left and node.right: # while the current node has left and right children
            if x[node.feature] <= node.threshold: # if the feature value of the input sample is less than or equal to the threshold
                node = node.left # move to the left child
            else:
                node = node.right # move to the right child
        return node.label # return the label of the leaf node
    
    def print_tree(self, node, depth=0):
        if node.label is not None: # if the current node is a leaf node
            print("  " * depth + "Leaf Node:", node.label) # print the label of the leaf node
            return
        print("  " * depth + f"Feature {node.feature} <= {node.threshold}") # print the feature and threshold of the current node
        self.print_tree(node.left, depth + 1) # recursively call the function on the left child
        self.print_tree(node.right, depth + 1) # recursively call the function on the right child
