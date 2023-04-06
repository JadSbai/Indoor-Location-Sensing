from abc import abstractmethod, ABC
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class Clustering(ABC):
    def __init__(self, positions: List[Tuple[float, float]]):
        seen = []
        epsilon = 1e-6
        for i in range(len(positions)):
            if positions[i] in seen:
                positions[i] = (positions[i][0] + epsilon, positions[i][1] + epsilon)
                epsilon += 1e-7
            else:
                seen.append(positions[i])
        self.positions = positions
        self.clustering_result = None
        self.labels = None
        self.model = None

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def plot_results(self):
        pass

    @abstractmethod
    def get_best_parameters(self):
        pass

    def get_labels(self):
        return self.labels

    def get_result(self):
        return self.clustering_result

    def get_label_data(self) -> Dict[Tuple[float, float], int]:
        label_data: Dict[Tuple[float, float], int] = {}
        labels = self.labels
        for i in range(len(labels)):
            label = labels[i]
            label_data[self.positions[i]] = label
        return label_data

    def combined_clustering_score(self, silhouette_weight=1, db_weight=-1, ch_weight=1):
        new_pos = []
        new_labels = []
        for i in range(len(self.positions)):
            if self.labels[i] != -1:
                new_pos.append(self.positions[i])
                new_labels.append(self.labels[i])

        silhouette = silhouette_score(new_pos, new_labels)
        db_index = davies_bouldin_score(new_pos, new_labels)
        ch_index = calinski_harabasz_score(new_pos, new_labels)

        scores = np.array([silhouette, db_index, ch_index])
        return scores[0] + scores[2]
