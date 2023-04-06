from collections import defaultdict
from typing import Dict, Tuple, List
from models.classification.classifier_interface import BaseClassifier
from models.clustering.clustering_interface import Clustering
from models.clustering.optics import OPTICSClustering
from models.particle_filter.particle_filter import ParticleFilter
from models.clustering.k_means import KMeansClustering
from models.helpers import get_positions_data, get_classifier_data, Map
from models.classification.rf_classifier import RFClassifier
from models.classification.KNN_classifier import KNeighborsClassifierModel


class CustomPredictor:
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]],
                 mean_speed=0.01, n_particles=500, is_k_means=True, isRF=True):
        self.training_data = training_data
        self.mean_speed = mean_speed
        self.n_particles = n_particles
        self.is_k_means = is_k_means
        self.isRF = isRF
        clustering_algo, classifier = self.train()
        self.clustering_model: Clustering = clustering_algo
        self.classification_model: BaseClassifier = classifier

    def train(self):
        positions = get_positions_data(self.training_data)
        clustering_algo = KMeansClustering(positions) if self.is_k_means else OPTICSClustering(positions)
        print("Clustering score: ", clustering_algo.combined_clustering_score())
        # clustering_algo.plot_results()
        label_data = clustering_algo.get_label_data()
        classifier_data = get_classifier_data(self.training_data, label_data)
        classifier = RFClassifier(classifier_data) if self.isRF else KNeighborsClassifierModel(
            classifier_data)
        score = classifier.get_score()
        print("Classifier score: ", score)
        classifier.fit()
        return clustering_algo, classifier

    def predict(self, measurement_data: Dict[str, Dict[float, List[Tuple[Tuple[int, int], int]]]], layout: Map):
        predicted_positions: Dict[str, Dict[float, Tuple[float, float]]] = defaultdict(dict)

        for mac_address in measurement_data:
            particle_filter = ParticleFilter(n_particles=self.n_particles, mean_speed=self.mean_speed, layout=layout)
            particle_filter.initialize_particles()
            sorted_timestamps = sorted(measurement_data[mac_address].keys())
            previous_timestamp: int = 0
            for timestamp in sorted_timestamps:
                if previous_timestamp != 0 and previous_timestamp != timestamp:
                    particle_filter.set_delta_t(timestamp - previous_timestamp)
                measurement = measurement_data[mac_address][timestamp]
                proba_distribution = self.classification_model.get_cluster_probabilities([measurement])[0]
                particle_filter.predict()
                centroids = self.clustering_model.clustering_result if self.is_k_means else None
                core_points = self.clustering_model.clustering_result if not self.is_k_means else None
                label_to_points = self.clustering_model.get_label_to_core_points() if not self.is_k_means else None
                particle_filter.update_weights(proba_distribution, centroids, core_points, label_to_points)
                estimated_position = particle_filter.average_estimate_position()
                predicted_positions[mac_address][timestamp] = estimated_position
        return predicted_positions
