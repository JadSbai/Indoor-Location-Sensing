import numpy as np
from models.custom_predictor import CustomPredictor
from models.helpers import get_measurement_data, get_performance, get_training_data, get_flat_predictions, \
    RoastingPlantMap
from models.regression.rf_regressor import RFRegressor
import matplotlib.pyplot as plt

if __name__ == "__main__":
    map_layout = RoastingPlantMap(height=15, width=24)
    fingerprint_data = np.load(file='data/fingerprints/roasting_plant_fingerprints.npy', allow_pickle=True).item()
    full_data, training_data = get_training_data(fingerprint_data)

    regressor = RFRegressor(training_data=training_data)
    regressor.fit()

    test_features = regressor.get_test_features()
    test_targets = regressor.get_test_targets()
    test_indices = regressor.get_test_indices()

    measurement_data, trace_back = get_measurement_data(test_features, full_data, test_indices)
    custom_predictor = CustomPredictor(training_data=training_data)
    custom_predicted_positions = custom_predictor.predict(measurement_data, map_layout)
    flat_predictions = get_flat_predictions(custom_predicted_positions, trace_back, len(test_indices))

    print("\nCustom predictor performance: \n")
    get_performance(flat_predictions, test_targets)

    print("\nRegressor performance: \n")
    regressor.score()
    regressor_predicted_positions = regressor.predict()

    x_positions = map(lambda point: point[0], test_targets)
    y_positions = map(lambda point: point[1], test_targets)
    plt.scatter(list(x_positions), list(y_positions), label="Targets")

    x_positions = map(lambda point: point[0], flat_predictions)
    y_positions = map(lambda point: point[1], flat_predictions)
    plt.scatter(list(x_positions), list(y_positions), label="Custom Model")

    x_positions = map(lambda point: point[0], regressor_predicted_positions)
    y_positions = map(lambda point: point[1], regressor_predicted_positions)
    plt.scatter(list(x_positions), list(y_positions), label="Random Forest Regressor")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
