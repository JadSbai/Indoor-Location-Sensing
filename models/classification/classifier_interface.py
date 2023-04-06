from typing import List, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV
from models.helpers import hot_encode


class BaseClassifier:
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], int]]):
        self.model = None
        self.fitted_model = None
        self.training_data = training_data
        feature_list, target_clusters = self.prep_data()
        self.features: List[List[Tuple[Tuple[int, int], int]]] = feature_list
        self.targets: List[int] = target_clusters

    def prep_data(self):
        feature_list: List[List[Tuple[Tuple[int, int], int]]] = []
        target_clusters: List[int] = []
        for data_point in self.training_data:
            feature_list.append(data_point[0])
            target_clusters.append(data_point[1])
        return feature_list, target_clusters

    def fit(self):
        encoded_features = hot_encode(self.features)
        self.fitted_model = self.model.fit(encoded_features, self.targets)

    def get_score(self):
        encoded_features = hot_encode(self.features)
        scores = cross_val_score(self.model, encoded_features, self.targets, cv=5, scoring='f1_macro')
        print(scores)
        average_score = sum(scores) / len(scores)
        return average_score

    def get_cluster_probabilities(self, measurements: List[List[Tuple[Tuple[int, int], int]]]) -> List[List[float]]:
        encoded_features = hot_encode(measurements)
        return self.fitted_model.predict_proba(encoded_features)

    def get_best_parameters(self, param_grid, cv=5):
        encoded_features = hot_encode(self.features)
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='f1_macro')
        grid_search.fit(encoded_features, self.targets)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
