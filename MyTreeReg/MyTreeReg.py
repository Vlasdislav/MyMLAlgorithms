import numpy as np
import pandas as pd


class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.fi = {}

    def fit(self, X, y):
        self.tree = None
        self.split_values = {}
        self.fi = {col: 0 for col in X.columns}
        
        def create_tree(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()

            y_root_unique_size = len(y_root.unique())
            if y_root_unique_size == 0 or y_root_unique_size == 1 or \
              depth >= self.max_depth or len(y_root) < self.min_samples_split \
              or (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = y_root.mean()
                return root

            col_name, split_value, gain = self.get_best_split(X_root, y_root)

            self.fi[col_name] += gain * len(y_root) / len(y)

            X_left = X_root[X_root[col_name] <= split_value]
            y_left = y_root[X_root[col_name] <= split_value]

            X_right = X_root[X_root[col_name] > split_value]
            y_right = y_root[X_root[col_name] > split_value]

            if len(X_left) == 0 or len(X_right) == 0:
                root.side = side
                root.value_leaf = y_root.mean()
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, X, y)

    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            node = self.tree
            while node.feature:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.value_leaf)
        return np.array(y_pred)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{' ' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{' ' * depth}{node.side} = {node.value_leaf}")

    def get_best_split(self, X, y):
        mse_0 = self.mse(y)

        col_name = None
        split_value = None
        gain = -float('inf')

        for col in X.columns:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(X[col])
                if self.bins is None or len(x_unique_values) - 1 < self.bins:
                    self.split_values[col] = np.array([(x_unique_values[i - 1] + \
                    x_unique_values[i]) / 2 for i in range(1, len(x_unique_values))])
                else:
                    _, self.split_values[col] = np.histogram(X[col], bins=self.bins)

            for split_value_i in self.split_values[col]:
                mask = X[col] <= split_value_i
                left_split, right_split = y[mask], y[~mask]

                mse_left = self.mse(left_split)
                mse_right = self.mse(right_split)

                weight_left = len(left_split) / len(y)
                weight_right = len(right_split) / len(y)

                mse_i = weight_left * mse_left + weight_right * mse_right

                gain_i = mse_0 - mse_i
                if gain < gain_i:
                    col_name = col
                    split_value = split_value_i
                    gain = gain_i

        return col_name, split_value, gain

    def mse(self, t):
        t_mean = np.mean(t)
        return ((t - t_mean) ** 2).mean()
    
    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyTreeReg class: " + ", ".join(params)
