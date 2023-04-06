from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from models.classification.classifier_interface import BaseClassifier


class RFClassifier(BaseClassifier):
    def __init__(self, training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], int]], random_state=123):
        super().__init__(training_data)
        self.model = RandomForestClassifier(criterion='entropy',
                                            n_estimators=1000,
                                            n_jobs=-1,
                                            max_features=None,
                                            random_state=random_state)

    def get_best_rf_parameters(self):
        param_grid = {'n_estimators': [100, 100, 1000]}
        best_params = self.get_best_parameters(param_grid)
        print("Best parameters for RF classifier: ", best_params)
        return best_params

