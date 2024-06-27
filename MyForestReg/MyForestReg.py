import random
import numpy as np
import pandas as pd
from collections import defaultdict


class MyForestReg:
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5,
                 random_state=42, max_depth=5, min_samples_split=2,
                 max_leafs=20, bins=16, oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.fi = {}
        self.oob_score = oob_score
        self.metrics = {
            'mae':  lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse':  lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2':   lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.oob_score_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.forest = []
        self.fi = {col: 0 for col in X.columns}
        self.leafs_cnt = 0

        random.seed(self.random_state)
        init_cols = list(X.columns)
        init_rows_cnt = n_samples
        cols_smpl_cnt = int(np.round(self.max_features * n_features))
        rows_smpl_cnt = int(np.round(self.max_samples * n_samples))
        all_rows = set(range(n_samples))

        oob_predictions = defaultdict(list)
        
        for _ in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            X_sample = X.iloc[rows_idx][cols_idx]
            y_sample = y.iloc[rows_idx]

            neg_rows_idx = list(all_rows - set(rows_idx))
            X_oob_sample = X.iloc[neg_rows_idx][cols_idx]
            y_oob_sample = y.iloc[neg_rows_idx]
            
            treeReg = MyTreeReg(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_leafs=self.max_leafs, bins=self.bins)
            
            treeReg.fit(X_sample, y_sample, len(y))
            self.forest.append(treeReg)
            self.leafs_cnt += treeReg.leafs_cnt
            self.fi = self.__union_dicts(self.fi, treeReg.fi)

            y_oob_pred = treeReg.predict(X_oob_sample)
            for idx, pred in zip(neg_rows_idx, y_oob_pred):
                oob_predictions[idx].append(pred)
        
        if self.oob_score is not None:
            oob_y_true = []
            oob_y_pred = []

            for idx, preds in oob_predictions.items():
                if preds:
                    oob_y_true.append(y.iloc[idx])
                    oob_y_pred.append(np.mean(preds))

            if oob_y_true:
                oob_y_true = np.array(oob_y_true)
                oob_y_pred = np.array(oob_y_pred)
                self.oob_score_ = self.metrics[self.oob_score](oob_y_true, oob_y_pred)

    def predict(self, X):
        pred = np.array([tree.predict(X) for tree in self.forest]).mean(axis=0)
        return pred

    def __union_dicts(self, dict1, dict2):
        dict_union = dict1.copy()
        for key, value in dict2.items():
            if key in dict1:
                dict_union[key] += value
            else:
                dict_union[key] = value
        return dict_union
    
    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyForestReg class: " + ", ".join(params)
