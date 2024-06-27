import random
import numpy as np
import pandas as pd


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
        sqr = 0
        n_ones = np.sum(self.y_true == 1)
        n_zeroes = np.sum(self.y_true == 0)
        m = n_ones * n_zeroes
        trip = sorted(zip(self.y_pred_proba, self.y_true), reverse=True)
        for _, true in trip:
            if true == 1:
                sqr += n_zeroes
            else:
                n_zeroes -= 1
        return sqr / m

class MyLogReg:
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1,
                 metric=None,
                 reg=None,
                 l1_coef=0.0,
                 l2_coef=0.0,
                 sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        
    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        ones = np.ones((n_samples, 1))
        X = np.hstack((ones, X))

        self.weights = np.ones((n_features + 1, 1))

        random.seed(self.random_state)

        EPS = 1e-15
        y = np.array(y)
        for iter in range(1, self.n_iter + 1):
            sample_rows_idx = range(n_samples)
            batch_size = self.sgd_sample if self.sgd_sample else n_samples
            if isinstance(batch_size, float):
                batch_size = int(n_samples * batch_size)
            sample_rows_idx = random.sample(range(n_samples), batch_size)

            X_sample = X[sample_rows_idx, :]
            y_sample = y[sample_rows_idx]

            z = X_sample @ self.weights
            y_pred = self.sigmoid(z).flatten()
            logloss = -np.mean(y_sample * np.log(y_pred + EPS) - (1 - y_sample) * np.log(1 - y_pred + EPS))

            grad = ((y_pred - y_sample) @ X_sample) / batch_size
            grad = grad.reshape(-1, 1)

            if self.reg:
                if self.reg == 'l1' or self.reg == 'elasticnet':
                    grad += self.l1_coef * np.sign(self.weights)
                    logloss += self.l1_coef * np.sum(np.abs(self.weights))
                if self.reg == 'l2' or self.reg == 'elasticnet':
                    grad += self.l2_coef * 2 * self.weights
                    logloss += self.l2_coef * np.sum(self.weights ** 2)

            if callable(self.learning_rate):
                lr = self.learning_rate(iter)
            else:
                lr = self.learning_rate

            self.weights -= lr * grad

            self.scores = TableMetrics(y_sample, y_pred, self.metric)

            if verbose and iter % verbose == 0:
                print(f"{iter if iter != 0 else 'start'} | loss: {logloss}", f"| {self.metric}: {self.scores.score()}" if self.metric else '', f"| learning_rate: {lr}")

    def predict_proba(self, X):
        n_samples, _ = X.shape
        ones = np.ones((n_samples, 1))
        X = np.hstack((ones, X))
        return self.sigmoid(X @ self.weights)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        self.scores.roc_auc()
        return self.scores.score()

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyLogReg class: " + ", ".join(params)
