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

class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.__sum_tree_values = 0
        self.split_values = {}
        self.criterion = criterion
        self.fi = {}

    def fit(self, X, y):
        self.tree = None
        self.fi = { col: 0 for col in X.columns }
        
        def create_tree(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, split_value, ig = self.get_best_split(X_root, y_root)

            proportion_ones = len(y_root[y_root == 1]) / len(y_root) if len(y_root) else 0

            if proportion_ones == 0 or proportion_ones == 1 or depth >= self.max_depth or \
              len(y_root) < self.min_samples_split or \
              (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            self.fi[col_name] += len(y_root) / len(y) * ig

            X_left = X_root.loc[X_root[col_name] <= split_value]
            y_left = y_root.loc[X_root[col_name] <= split_value]

            X_right = X_root.loc[X_root[col_name] > split_value]
            y_right = y_root.loc[X_root[col_name] > split_value]

            if len(X_left) == 0 or len(X_right) == 0:
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, X, y)

    def predict_proba(self, X):
        for _, row in X.iterrows():
            node = self.tree
            while node.feature is not None:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            yield node.value_leaf

    def predict(self, X):
        y_pred = np.array(list(self.predict_proba(X)))
        return (y_pred >= 0.5).astype(int)

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
        count_labels = y.value_counts()
        p_zero = count_labels / count_labels.sum()
        s_zero = self.__node_rule(p_zero)

        col_name = None
        split_value = None
        s_cur_min = float('inf')

        for col in X.columns:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(X[col])
                if self.bins is not None and len(x_unique_values) - 1 >= self.bins:
                    _, self.split_values[col] = np.histogram(X[col], bins=self.bins)
                    self.split_values[col] = self.split_values[col][1:-1]
                else:
                    self.split_values[col] = (x_unique_values[1:] + x_unique_values[:-1]) / 2

            for split_value_cur in self.split_values[col]:
                left_split = y[X[col] <= split_value_cur]
                right_split = y[X[col] > split_value_cur]

                left_count_labels = left_split.value_counts()
                p_left = left_count_labels / left_count_labels.sum()
                s_left = self.__node_rule(p_left, left_split)

                right_count_labels = right_split.value_counts()
                p_right = right_count_labels / right_count_labels.sum()
                s_right = self.__node_rule(p_right, right_split)

                weight_left = len(left_split) / len(y)
                weight_right = len(right_split) / len(y)

                s_cur = weight_left * s_left + weight_right * s_right
                if s_cur_min > s_cur:
                    s_cur_min = s_cur
                    col_name = col
                    split_value = split_value_cur

        ig = s_zero - s_cur_min
        return col_name, split_value, ig

    def __node_rule(self, p, split=pd.Series()):
        if self.criterion == 'entropy':
            return -np.sum(p * np.log2(p)) if not split.empty else 0
        elif self.criterion == 'gini':
            return 1 - np.sum(p ** 2)
    
    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyTreeClf class: " + ", ".join(params)
