from typing import Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from models.clustering.clustering_interface import Clustering


class KMeansClustering(Clustering):
    def __init__(self, positions):
        super().__init__(positions)
        self.model = KMeans(n_clusters=9)
        self.fit()

    def fit(self):
        self.model.fit(self.positions)
        self.clustering_result = self.model.cluster_centers_
        self.labels = self.model.labels_

    def get_best_parameters(self):
        K = range(2, 10)
        best_score = float('-inf')
        best_k = None
        for k in K:
            # train the model for current value of k on training data
            self.model = KMeans(n_clusters=k, random_state=0, n_init=10)
            self.fit()
            score = self.combined_clustering_score()
            if score > best_score:
                best_k = k
                best_score = score

        print("Best k: ", best_k, ". With score: ", best_score)
        return best_k

    def plot_results(self):
        label_data: Dict[Tuple[float, float], int] = self.get_label_data()
        cluster_centers = self.clustering_result
        u_labels = np.unique(self.labels)
        # plotting the results:
        for i in u_labels:
            filtered = []
            for pos in label_data:
                if label_data[pos] == i:
                    filtered.append(pos)
            x_positions = map(lambda point: point[0], filtered)
            y_positions = map(lambda point: point[1], filtered)
            plt.scatter(list(x_positions), list(y_positions), label=i)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=80, color='k')
        plt.legend()
        plt.show()
