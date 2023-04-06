from typing import List, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from models.helpers import prep_data, get_average_distance_error, hot_encode


class BaseRegressor:
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]]):
        self.model = None
        self.fitted_model = None
        feature_list, target_positions = prep_data(training_data)
        self.features: List[List[Tuple[Tuple[int, int], int]]] = feature_list
        self.targets: List[Tuple[float, float]] = target_positions
        features_train, features_test, targets_train, targets_test, train_indices, test_indices = train_test_split(
            self.features,
            self.targets,
            range(len(self.features)),
            test_size=.20, random_state=123)
        self.features_train: List[List[Tuple[Tuple[int, int], int]]] = features_train
        self.features_test: List[List[Tuple[Tuple[int, int], int]]] = features_test
        self.target_train: List[Tuple[float, float]] = targets_train
        self.target_test: List[Tuple[float, float]] = targets_test
        self.train_indices: List[int] = train_indices
        self.test_indices: List[int] = test_indices

    def fit(self):
        encoded_features = hot_encode(self.features_train)
        self.fitted_model = self.model.fit(encoded_features, self.target_train)

    def predict(self):
        encoded_features = hot_encode(self.features_test)
        predictions = self.fitted_model.predict(encoded_features)
        mse = mean_squared_error(self.target_test, predictions)
        print("Mean Squared Error", mse)
        print("Average distance error: ", get_average_distance_error(self.target_test, predictions))
        return predictions

    def score(self):
        encoded_features = hot_encode(self.features)
        scores = cross_val_score(self.model, encoded_features, self.targets, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        average_score = sum(scores) / len(scores)
        return average_score

    def get_best_parameters(self, param_grid, cv=5):
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='neg_mean_squared_error')
        encoded_features = hot_encode(self.features)
        grid_search.fit(encoded_features, self.targets)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_training_features(self) -> List[List[Tuple[Tuple[int, int], int]]]:
        return self.features_train

    def get_training_targets(self) -> List[Tuple[float, float]]:
        return self.target_train

    def get_test_features(self) -> List[List[Tuple[Tuple[int, int], int]]]:
        return self.features_test

    def get_test_targets(self) -> List[Tuple[float, float]]:
        return self.target_test

    def get_test_indices(self) -> List[int]:
        return self.test_indices

    def get_train_indices(self) -> List[int]:
        return self.train_indices
