import random
import numpy as np
import pandas as pd


class MyLineReg:
    def __init__(self,
                 n_iter=100,
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
        self.best_score = None
        self.metrics = {
            'mae':  lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse':  lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2':   lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        ones = np.ones((n_samples, 1))
        X = np.hstack((ones, X))
        
        random.seed(self.random_state)

        self.weights = np.ones(n_features + 1)

        for iter in range(1, self.n_iter + 1):
            y_pred = X @ self.weights
            loss = self.__calc_loss(y, y_pred)

            sample_rows_idx = range(n_samples)
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(n_samples), self.sgd_sample)
            if isinstance(self.sgd_sample, float):
                k = int(n_samples * self.sgd_sample)
                sample_rows_idx = random.sample(range(n_samples), k)

            X_sample = X.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]

            if callable(self.learning_rate):
                lr = self.learning_rate(iter)
            else:
                lr = self.learning_rate

            self.weights -= lr * self.__calc_grad(X_sample, y_sample)

            if self.metric is not None:
                self.best_score = self.metrics[self.metric](y_sample, y_pred)

            if verbose and iter % verbose == 0:
                print(f"{iter if iter != 0 else 'start'} | loss: {loss}", f"| {self.metric}: {self.best_score}" if self.metric else '', f"| learning_rate: {lr}")
    
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        return X @ self.weights
    
    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.best_score

    def __calc_loss(self, y, y_pred):
        loss = np.sum((y - y_pred) ** 2) / y.shape[0]
            
        if self.reg == 'l1' or self.reg == 'elasticnet':
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        if self.reg == 'l2' or self.reg == 'elasticnet':
            loss += self.l2_coef * np.sum(self.weights ** 2)

        return loss
    
    def __calc_grad(self, X, y):
        n_samples, _ = X.shape
        grad = 2 / n_samples * (X.T @ (X @ self.weights - y))
        
        if self.reg:
            if self.reg == 'l1' or self.reg == 'elasticnet':
                grad += self.l1_coef * np.sign(self.weights)
            if self.reg == 'l2' or self.reg == 'elasticnet':
                grad += self.l2_coef * 2 * self.weights
        
        return grad
        
    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyLineReg class: " + ", ".join(params)
