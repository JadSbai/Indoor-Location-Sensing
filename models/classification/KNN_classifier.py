from typing import Tuple, List

from sklearn.neighbors import KNeighborsClassifier

from models.classification.classifier_interface import BaseClassifier


class KNeighborsClassifierModel(BaseClassifier):
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], int]]):
        super().__init__(training_data)
        self.model = KNeighborsClassifier(n_neighbors=100)

    def get_best_knn_parameters(self):
        param_grid = {'n_neighbors': [1, 5, 100]}
        best_params = self.get_best_parameters(param_grid)
        print("Best parameters for k-NN classifier: ", best_params)
        return best_params
