import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight

        self.metrics = {
            'euclidean': lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=1)),
            'chebyshev': lambda x1, x2: np.max(np.abs(x1 - x2), axis=1),
            'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=1),
            'cosine':    lambda x1, x2: 1 - (x1 @ x2) / (np.sqrt(np.sum(x1 ** 2, axis=1)) * np.sqrt(np.sum(x2 ** 2)))
        }

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()

    def predict(self, X):
        predictions = []
        for _, test_row in X.iterrows():
            distances = self.metrics[self.metric](self.X.values, test_row.values)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y.iloc[k_nearest_indices]
            if self.weight == 'uniform':
                counts = k_nearest_labels.value_counts()
                count_1 = counts.get(1, 0)
                count_0 = counts.get(0, 0)
                predictions.append([0, 1][count_1 >= count_0])
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                rank_weights = 1 / ranks
                weights = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights[label] += rank_weights[i]
                predictions.append(np.argmax(weights))
            elif self.weight == 'distance':
                distances[distances == 0] = 1e-10
                weights = 1 / distances[k_nearest_indices]
                weights_sum = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights_sum[label] += weights[i]
                predictions.append(np.argmax(weights_sum))
        return np.array(predictions)

    def predict_proba(self, X):
        probabilities = []
        for _, test_row in X.iterrows():
            distances = self.metrics[self.metric](self.X.values, test_row.values)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y.iloc[k_nearest_indices]
            if self.weight == 'uniform':
                prob_class_1 = np.mean(k_nearest_labels)
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1)
                rank_weights = 1 / ranks
                weights = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights[label] += rank_weights[i]
                prob_class_1 = weights[1] / np.sum(weights)
            elif self.weight == 'distance':
                distances[distances == 0] = 1e-10
                weights = 1 / distances[k_nearest_indices]
                weights_sum = np.zeros(2)
                for i, label in enumerate(k_nearest_labels):
                    weights_sum[label] += weights[i]
                prob_class_1 = weights_sum[1] / np.sum(weights_sum)
            probabilities.append(prob_class_1)
        return np.array(probabilities)
    
    def __str__(self):
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyKNNClf class: " + ", ".join(params)
