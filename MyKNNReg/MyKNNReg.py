import numpy as np
import pandas as pd


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.metrics = {
            'euclidean': lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=1)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
            'monhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=1),
            'cosine':    lambda x1, x2: 1 - x1 @ x2 / (np.sqrt(np.sum(x1 ** 2, axis=1)) * np.sqrt(np.sum(x2 ** 2)))
        }
        self.metric = metric
        self.weight = weight

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def predict(self, X):
        predictions = []
        for _, test_row in X.iterrows():
            distances = self.metrics[self.metric](self.X, test_row)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y.iloc[k_nearest_indices]
            if self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                rank_weights = 1 / ranks
                weighted_sum = np.sum(rank_weights * k_nearest_labels)
                predictions.append(weighted_sum / np.sum(rank_weights))
            elif self.weight == 'distance':
                distances = distances[k_nearest_indices]
                distances[distances == 0] = 1e-10
                dist_weights = 1 / distances
                weighted_sum = np.sum(dist_weights * k_nearest_labels)
                predictions.append(weighted_sum / np.sum(dist_weights))
            else:
                predictions.append(np.mean(k_nearest_labels))
        return np.array(predictions)
    
    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyKNNReg class: " + ", ".join(params)
