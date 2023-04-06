from collections import defaultdict
from typing import Dict, List, Tuple
from models.helpers import Map
from wifi_sniffer.wifi_sniffer import WifiSniffer
import math
import numpy as np


def analyse_roasting_plant_data():
    wifi_sniffer_1 = WifiSniffer('data/captures/roasting_plant_2.pcap')
    m1 = wifi_sniffer_1.analyse_capture()
    np.save('data/measurements/roasting_plant_measurements.npy', m1)
    wifi_sniffer_2 = WifiSniffer('data/captures/roasting_plant_2.pcap')
    wifi_sniffer_2.set_anchor_node(2)
    m2 = wifi_sniffer_2.analyse_capture()
    np.save('data/measurements/roasting_plant_measurements_2.npy', m2)


def merge_roasting_plant_data():
    node_1_readings = np.load(file='data/measurements/roasting_plant_measurements.npy', allow_pickle=True).item()
    node_2_readings = np.load(file='data/measurements/roasting_plant_measurements_2.npy', allow_pickle=True).item()

    merged: Dict[str, Dict[float, List[Tuple[Tuple[int, int], int]]]] = defaultdict(dict)
    n = 2
    for mac, d in node_1_readings.items():
        if mac not in merged:
            merged[mac] = {}
        for timestamp, measurements in d.items():
            grouped_timestamp = math.floor(timestamp / n) * n
            if grouped_timestamp not in merged[mac]:
                merged[mac][grouped_timestamp] = []
            merged[mac][grouped_timestamp].extend(measurements)
    for mac, d in node_2_readings.items():
        if mac not in merged:
            merged[mac] = {}
        for timestamp, measurements in d.items():
            grouped_timestamp = math.floor(timestamp / n) * n
            if grouped_timestamp not in merged[mac]:
                merged[mac][grouped_timestamp] = []
            merged[mac][grouped_timestamp].extend(measurements)
    for mac, d in merged.items():
        for timestamp, measurements in d.items():
            grouped_measurements = {}
            for measurement, anchor_id in measurements:
                if anchor_id not in grouped_measurements:
                    grouped_measurements[anchor_id] = []
                grouped_measurements[anchor_id].append(measurement)
            averaged_measurements = []
            for anchor_id, measurements in grouped_measurements.items():
                avg_measurement = tuple(sum(x) / len(x) for x in zip(*measurements))
                averaged_measurements.append((avg_measurement, anchor_id))
            merged[mac][timestamp] = averaged_measurements

    np.save('data/measurements/roasting_plant_merged.npy', merged)


def produce_fingerprints(measurements: Dict[str, Dict[float, List[Tuple[Tuple[int, int], int]]]],
                         positions: Dict[str, Dict[Tuple[float, float], Tuple[int, int]]]):
    fingerprints: Dict[str, Dict[float, Tuple[List[Tuple[Tuple[int, int], int]], Tuple[float, float]]]] = defaultdict(
        dict)
    for address in positions:
        readings = measurements[address]
        windows = list(positions[address].keys())
        position_mapping = positions[address]
        for timestamp in readings:
            i = 0
            found = False
            while not found and i < len(windows):
                window = windows[i]
                if window[0] <= timestamp <= window[1]:
                    reading = readings[timestamp]
                    fingerprints[address][timestamp] = (reading, position_mapping[window])
                    found = True
                i += 1
    return fingerprints


def extract_training_data(mapLayout: Map):
    merged_data = np.load(file='data/measurements/roasting_plant_merged.npy', allow_pickle=True).item()
    start_time = 1680068220
    end_time = 1680069600
    roasting_plant_positions = mapLayout.get_fingerprint_positions(start_time, end_time)
    fingerprints = produce_fingerprints(merged_data, roasting_plant_positions)
    np.save('data/fingerprints/roasting_plant_fingerprints.npy', fingerprints)
