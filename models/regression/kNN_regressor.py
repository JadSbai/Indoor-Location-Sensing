from typing import Tuple, List
from sklearn.neighbors import KNeighborsRegressor
from models.regression.regressor_interface import BaseRegressor


class KNeighborsRegressorModel(BaseRegressor):
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]]):
        super().__init__(training_data)
        self.model = KNeighborsRegressor(n_neighbors=200)

    def get_best_knn_parameters(self):
        param_grid = {'n_neighbors': [1, 5, 500]}
        best_params = self.get_best_parameters(param_grid)
        print("Best parameters for k-NN regressor: ", best_params)
        return best_params
