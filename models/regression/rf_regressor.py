from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List
from models.regression.regressor_interface import BaseRegressor


class RFRegressor(BaseRegressor):
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]],
                 random_state=123):
        super().__init__(training_data)
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state,
                                           max_features=None,
                                           n_jobs=-1)

    def get_best_rf_parameters(self):
        param_grid = {'n_estimators': [50, 50, 500]}
        best_params = self.get_best_parameters(param_grid)
        print("Best parameters for RF regressor: ", best_params)
        return best_params
