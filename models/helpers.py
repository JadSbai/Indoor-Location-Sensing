from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List, Tuple, Dict
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Map(ABC):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    @abstractmethod
    def is_valid_position(self, x: float, y: float) -> bool:
        pass

    @abstractmethod
    def get_fingerprint_positions(self, start_time, end_time):
        pass


class RoastingPlantMap(Map):
    def __init__(self, height, width):
        super().__init__(height, width)

    def is_valid_position(self, x: float, y: float) -> bool:
        forbidden = (7 <= x <= 8 and 3 <= y <= 10) or \
                    (0 <= x <= 1 and 9 <= y <= 12) or \
                    (0 <= x <= 3 and 0 <= y <= 3) or \
                    (9 <= x <= 11 and 0 <= y <= 1) or \
                    (13 <= x <= 24 and 5 <= y <= 15) or \
                    (23 <= x <= 24 and 0 <= y <= 3)
        return not forbidden and 0 <= x <= self.width and 0 <= y <= self.height

    def get_fingerprint_positions(self, start_time, end_time):
        path = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 9), (1, 10), (1, 11), (2, 11), (2, 10), (2, 9),
                (2, 8),
                (2, 7),
                (2, 6),
                (2, 5), (2, 4), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 11), (5, 11),
                (6, 11), (7, 11), (8, 11), (8, 10), (8, 9), (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2),
                (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (10, 11), (10, 10),
                (10, 9),
                (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (11, 2), (11, 1), (11, 0), (12, 0),
                (12, 1), (12, 2), (12, 3), (13, 3), (14, 3), (15, 3), (16, 3), (17, 3), (18, 3), (19, 3), (20, 3),
                (21, 3), (22, 3), (22, 2), (22, 1), (22, 0), (21, 0), (20, 0), (19, 0), (18, 0), (17, 0), (16, 0),
                ]
        path = [(x + 0.5, y + 0.5) for (x, y) in path]
        positions: Dict[str, Dict[Tuple[float, float], Tuple[int, int]]] = defaultdict(dict)
        mac = '9c:2e:7a:ac:70:d8'
        positions[mac][(float('-inf'), start_time)] = path[0]
        waiting_time = 20
        counter = start_time
        for pos in path:
            if counter >= 1680069280:
                waiting_time = 10
            positions[mac][(counter, counter + waiting_time)] = pos
            counter += waiting_time
        positions[mac][(end_time, float('inf'))] = path[len(path) - 1]

        path.reverse()

        mac = '78:02:8b:e2:4d:cb'
        positions[mac][(float('-inf'), start_time)] = path[0]
        counter = start_time
        waiting_time = 20
        for pos in path:
            if counter >= 1680069280:
                waiting_time = 10
            positions[mac][(counter, counter + waiting_time)] = pos
            counter += waiting_time
        positions[mac][(end_time, float('inf'))] = path[len(path) - 1]

        return positions


def prep_data(data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]]):
    feature_list: List[List[Tuple[Tuple[int, int], int]]] = []
    target_positions: List[Tuple[float, float]] = []
    for data_point in data:
        feature_list.append(data_point[0])
        target_positions.append(data_point[1])
    return feature_list, target_positions


def get_performance(predicted_positions: List[Tuple[float, float]],
                    true_positions: List[Tuple[float, float]]):
    mse = mean_squared_error(true_positions, predicted_positions)
    dist_e = get_average_distance_error(true_positions, predicted_positions)
    print("Mean squared error: ", mse)
    print("Average distance error: ", dist_e)
    return mse, dist_e


def get_average_distance_error(targets, predictions):
    # Calculate the Euclidean distance between each pair of corresponding points
    distances = [math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for ((x1, y1), (x2, y2)) in zip(targets, predictions)]

    # Calculate the mean of the distances
    avg_distance = sum(distances) / len(distances)

    return avg_distance * 0.5


def get_positions_data(data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]]):
    positions_data: List[Tuple[float, float]] = []
    for data_point in data:
        positions_data.append(data_point[1])
    return positions_data


def get_classifier_data(data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]],
                        label_data: Dict[Tuple[float, float], int]):
    classifier_data: List[Tuple[List[Tuple[Tuple[int, int], int]], int]] = []
    for i in range(len(data)):
        data_point = data[i]
        features = data_point[0]
        position = data_point[1]
        cluster = label_data[position]
        if cluster != -1:
            classifier_data.append((features, cluster))
    return classifier_data


def get_measurement_data(test_features_data: List[List[Tuple[Tuple[int, int], int]]],
                         full_data: List[Tuple[str, float, List[Tuple[Tuple[int, int], int]], Tuple[float, float]]],
                         test_indices: List[int]):
    measurement_data: Dict[str, Dict[float, List[Tuple[Tuple[int, int], int]]]] = defaultdict(dict)
    trace_back: Dict[Tuple[str, float], int] = {}
    for i in range(len(test_features_data)):
        measurements = test_features_data[i]
        index = test_indices[i]
        mac_address, timestamp, _, _ = full_data[index]
        measurement_data[mac_address][timestamp] = measurements
        trace_back[(mac_address, timestamp)] = i
    return measurement_data, trace_back


def get_training_data(
        fingerprints: Dict[str, Dict[float, Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]]]):
    data_list: List[Tuple[str, float, List[Tuple[Tuple[int, int], int]], Tuple[float, float]]] = []
    for mac_address, timestamp_data in fingerprints.items():
        for timestamp, (measurements, position) in timestamp_data.items():
            data_list.append((mac_address, timestamp, measurements, position))

    training_data: List[Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]] = [(measurements, position) for
                                                                                          _, _, measurements, position
                                                                                          in data_list]
    return data_list, training_data


def get_flat_predictions(predictions, trace_back, length):
    flat_predictions: List[Tuple[float, float]] = [(0, 0)] * length
    for mac_address in predictions:
        for timestamp in predictions[mac_address]:
            flat_predictions[trace_back[(mac_address, timestamp)]] = predictions[mac_address][timestamp]
    return flat_predictions


def hot_encode(features: List[List[Tuple[Tuple[int, int], int]]]):
    encoded_features = []
    for measurements in features:
        feature = [-100, 0, -100, 0]
        for measurement in measurements:
            rss, snr = measurement[0]
            anchor_id = measurement[1]
            if anchor_id == 1:
                feature[0] = rss
                feature[1] = snr
            if anchor_id == 2:
                feature[2] = rss
                feature[3] = snr
        encoded_features.append(feature)
    return encoded_features


def create_layout():
    num_rows = 15
    num_cols = 24

    # create a new figure and set the size
    fig, ax = plt.subplots(figsize=(12, 8))

    # create a grid of rectangles to represent the cells
    for i in range(num_rows):
        for j in range(num_cols):
            rect = plt.Rectangle((j, num_rows - i - 1), 1, 1, linewidth=1, edgecolor='black', facecolor='white',
                                 alpha=0.5)
            ax.add_patch(rect)

    areas = [(5, 14, 13, 23), (0, 2, 23, 23), (0, 0, 9, 10), (3, 9, 7, 7), (0, 2, 0, 2), (9, 11, 0, 0)]

    # loop over the rows and columns in the rectangular area and color each corresponding patch
    for area in areas:
        for i in range(area[0], area[1] + 1):
            for j in range(area[2], area[3] + 1):
                ax.patches[(num_rows - 1 - i) * num_cols + j].set_facecolor('red')

    # set the tick labels to show the indices of the cells
    ax.set_xticks(range(num_cols + 1))
    ax.set_yticks(range(num_rows + 1))
    ax.set_xticklabels(range(num_cols + 1))
    ax.set_yticklabels(range(num_rows + 1))

    # add grid lines to show the boundaries between the cells
    ax.set_xticks([x for x in range(num_cols + 1)], minor=True)
    ax.set_yticks([y for y in range(num_rows + 1)], minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # write text on top of certain cells
    ax.text(17.5, 5.5, 'COUNTER', ha='center', va='center', color='black', fontsize=20, fontweight='bold')
    ax.text(13, 0.5, 'ENTRANCE', ha='center', va='center', color='black', fontsize=15, fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    return fig, ax, plt


def get_cleaned_data(data):
    # create a dictionary to keep track of the lowest second value for each fourth value
    lowest_second = {}

    # create a dictionary to keep track of the sum and count of third values for each fourth value
    third_sums = {}
    third_counts = {}

    # create a dictionary to keep track of the sum and count of third values for each fourth value
    fourth_sums = {}
    fourth_counts = {}

    # iterate over each element in the data list
    for element in data:
        # get the fourth value of the current element
        fifth_value = element[4]

        # if the fourth value is not in the lowest_second dictionary, add it with the current element's second value
        if fifth_value not in lowest_second:
            lowest_second[fifth_value] = element[1]

        # otherwise, compare the current element's second value to the existing value for the fourth value,
        # and update the lowest_second dictionary if the current element has a lower second value
        else:
            if element[1] < lowest_second[fifth_value]:
                lowest_second[fifth_value] = element[1]

        # if the fourth value is not in the third_sums and third_counts dictionaries,
        # add it with the current element's third value as the sum and count
        if fifth_value not in third_sums:
            third_sums[fifth_value] = element[2]
            third_counts[fifth_value] = 1
            fourth_sums[fifth_value] = element[3]
            fourth_counts[fifth_value] = 1

        # otherwise, add the current element's third value to the sum for the fourth value,
        # and increment the count of elements for the fourth value
        else:
            third_sums[fifth_value] = tuple([sum(x) for x in zip(third_sums[fifth_value], element[2])])
            third_counts[fifth_value] += 1
            fourth_sums[fifth_value] = tuple([sum(x) for x in zip(fourth_sums[fifth_value], element[3])])
            fourth_counts[fifth_value] += 1

    # create a new list to hold the cleaned data
    cleaned_data = []

    # iterate over each element in the data list
    for element in data:
        # get the fourth value of the current element
        fifth_value = element[4]

        # if the current element has the lowest second value for the fourth value,
        # add the element to the cleaned data list with the averaged third value
        if element[1] == lowest_second[fifth_value]:
            third_value = tuple([x / third_counts[fifth_value] for x in third_sums[fifth_value]])
            fourth_value = tuple([x / fourth_counts[fifth_value] for x in fourth_sums[fifth_value]])
            cleaned_data.append((element[0], element[1], third_value, fourth_value, fifth_value))

    # print the cleaned data
    return cleaned_data
