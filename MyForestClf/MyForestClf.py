import random
import numpy as np
import pandas as pd
from collections import defaultdict


class TableMetrics:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = np.array(y_true)
        self.y_pred = np.array((y_pred_proba > 0.5).astype(int))
        self.y_pred_proba = y_pred_proba
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))

    def score(self, metric):
        if metric == 'accuracy':
            return self.accuracy()
        elif metric == 'precision':
            return self.precision()
        elif metric == 'recall':
            return self.recall()
        elif metric == 'f1':
            return self.f1_score()
        elif metric == 'roc_auc':
            return self.roc_auc()
        elif metric == 'false_positive_rate':
            return self.false_positive_rate()
        return None

    def accuracy(self):
        if self.tp + self.tn + self.fp + self.fn == 0:
            return 0
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def false_positive_rate(self):
        return self.fp / (self.fp + self.tn)

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def roc_auc(self):
        positives = np.sum(self.y_true == 1)
        negatives = np.sum(self.y_true == 0)

        y_prob = np.round(self.y_pred_proba, 10)

        sorted_idx = np.argsort(-y_prob)
        y_sorted = self.y_true[sorted_idx]
        y_prob_sorted = y_prob[sorted_idx]

        roc_auc_score = 0

        for prob, pred in zip(y_prob_sorted, y_sorted):
            if pred == 0:
                roc_auc_score += (
                    np.sum(y_sorted[y_prob_sorted > prob])
                    + np.sum(y_sorted[y_prob_sorted == prob]) / 2
                )

        roc_auc_score /= positives * negatives

        return roc_auc_score

class MyForestClf:
    def __init__(self, n_estimators=10, max_features=0.5,
                 max_samples=0.5, random_state=42, max_depth=5,
                 min_samples_split=2, max_leafs=20, bins=16,
                 criterion='entropy', oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.criterion = criterion
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.forest = []

        self.fi = {col: 0 for col in X.columns}

        random.seed(self.random_state)
        init_cols = list(X.columns)
        cols_smpl_cnt = int(np.round(self.max_features * n_features))
        rows_smpl_cnt = int(np.round(self.max_samples * n_samples))
        all_rows = set(range(n_samples))

        oob_predictions = defaultdict(list)

        for _ in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(all_rows, rows_smpl_cnt)
            X_sample = X[cols_idx].iloc[rows_idx]
            y_sample = y.iloc[rows_idx]

            neg_rows_idx = list(all_rows - set(rows_idx))
            X_oob_sample = X.iloc[neg_rows_idx][cols_idx]

            treeClf = MyTreeClf(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_leafs=self.max_leafs, bins=self.bins,
                                criterion=self.criterion)
            
            treeClf.fit(X_sample, y_sample, len(y))
            self.forest.append(treeClf)
            self.leafs_cnt += treeClf.leafs_cnt
            self.fi = self.__union_dicts(self.fi, treeClf.fi)

            y_oob_pred_proba = list(treeClf.predict_proba(X_oob_sample))
            for idx, pred_proba in zip(neg_rows_idx, y_oob_pred_proba):
                oob_predictions[idx].append(pred_proba)

        if self.oob_score is not None:
            oob_y_true = []
            oob_y_pred_proba = []

            for idx, preds in oob_predictions.items():
                if preds:
                    oob_y_true.append(y.iloc[idx])
                    oob_y_pred_proba.append(np.mean(preds))

            if oob_y_true:
                oob_y_true = np.array(oob_y_true)
                oob_y_pred_proba = np.array(oob_y_pred_proba)
                tableMetrics = TableMetrics(oob_y_true, oob_y_pred_proba)
                self.oob_score_ = tableMetrics.score(self.oob_score)

    def predict(self, X, type):
        pred_probs = np.array([list(tree.predict_proba(X)) for tree in self.forest])
        if type == 'mean':
            pred = pred_probs.mean(axis=0)
            pred = (pred > 0.5).astype(int)
        elif type == 'vote':
            pred_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=0, arr=(pred_probs > 0.5).astype(int))
            pred = pred_votes.argmax(axis=0)
        return pred

    def predict_proba(self, X):
        pred = np.array([list(tree.predict_proba(X)) for tree in self.forest]).mean(axis=0)
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
        return "MyForestClf class: " + ", ".join(params)
