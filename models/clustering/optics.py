from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import OPTICS
from models.clustering.clustering_interface import Clustering
import matplotlib.pyplot as plt


class OPTICSClustering(Clustering):
    def __init__(self, positions: List[Tuple[float, float]]):
        super().__init__(positions)
        self.min_samples = 34
        self.xi = 0.01
        self.model = OPTICS(min_samples=self.min_samples, xi=self.xi)
        self.fit()

    def get_best_threshold(self):
        # Get reachability distances
        reachability = self.model.reachability_

        # Sort reachability distances
        sorted_reachability = np.sort(reachability[~np.isnan(reachability)])

        # Compute the difference between consecutive sorted reachability distances
        diff = np.diff(sorted_reachability)

        # Find the index where the difference is the largest
        elbow_index = np.argmax(diff)

        # Get the threshold value corresponding to the elbow point
        threshold = sorted_reachability[elbow_index]
        return threshold

    def fit(self):
        self.model.fit(self.positions)
        threshold = self.get_best_threshold()
        core_distance = self.model.core_distances_
        reachability = self.model.reachability_
        # Extract the core points from the dataset using the indices
        core_points = [self.positions[i] for i, (cd, rd) in enumerate(zip(core_distance, reachability)) if
                       cd <= threshold and rd <= threshold]
        self.clustering_result = core_points
        self.labels = self.model.labels_

    def plot_results(self):
        label_data = self.get_label_to_core_points()
        for label in label_data:
            points = []
            for i in range(len(self.positions)):
                if self.labels[i] == label:
                    points.append(self.positions[i])
            x_positions = map(lambda point: point[0], points)
            y_positions = map(lambda point: point[1], points)
            plt.scatter(list(x_positions), list(y_positions), label=label)
        plt.legend()
        plt.show()

    def get_best_parameters(self):
        # Define the parameter ranges to explore
        min_samples_range = np.arange(10, 60)
        xi_range = np.arange(0.01, 0.1, 0.01)

        # Initialize the best parameters and score
        best_params = None
        best_score = float('-inf')

        # Perform a grid search for the best parameters
        for min_samples in min_samples_range:
            for xi in xi_range:
                self.model = OPTICS(min_samples=min_samples, xi=xi)
                self.fit()

                # Ignore results with only one cluster or noise
                if len(set(self.labels)) <= 1:
                    continue

                # Compute the combined clustering score
                score = self.combined_clustering_score()

                # Update the best parameters and score if the current score is better
                if score > best_score:
                    best_params = {'min_samples': min_samples, 'xi': xi}
                    best_score = score
        print("Best parameters:", best_params)
        return best_params

    def get_label_to_core_points(self):
        # Identify core points
        is_core_point = self.model.reachability_[self.model.ordering_] <= self.model.core_distances_[self.model.ordering_]

        # Get the labels of core points
        core_labels = self.labels[self.model.ordering_][is_core_point]

        # Get the positions of core points
        core_positions = np.array(self.positions)[self.model.ordering_][is_core_point]

        # Create a dictionary to map each label to its corresponding core points
        label_to_core_points: Dict[int, List[Tuple[float, float]]] = {}
        for label, position in zip(core_labels, core_positions):
            if label not in label_to_core_points:
                label_to_core_points[label] = []
            label_to_core_points[label].append(position)

        return label_to_core_points


