import time
from collections import defaultdict
from typing import Dict, List, Tuple
import pyshark


class WifiSniffer:
    def __init__(self, capture: str, anchor_node=1):
        self.capture = capture
        self.measurements: Dict[str, Dict[float, List[Tuple[Tuple[int, int], int]]]] = defaultdict(dict)
        self.anchor_node = anchor_node

    def set_capture(self, new_capture: str):
        self.capture = new_capture

    def set_anchor_node(self, new_anchor_node):
        self.anchor_node = new_anchor_node

    def get_num_measurements(self):
        return sum(len(subdict[timestamp]) for address, subdict in self.measurements.items() for timestamp in subdict)

    def get_measurements(self):
        return self.measurements

    def add_measurement(self, address: str, timestamp: float, rss: int, snr: int):
        is_real_mac_address = address[1] not in ['e', 'a', '2', '6']
        if is_real_mac_address:
            if timestamp in self.measurements[address]:
                self.measurements[address][timestamp].append(((rss, snr), self.anchor_node))
            else:
                self.measurements[address][timestamp] = [((rss, snr), self.anchor_node)]
            return True

    def analyse_capture(self):
        capture = pyshark.FileCapture(self.capture)
        capture.set_debug()
        start = time.time()
        for packet in capture:
            if hasattr(packet, 'wlan') and hasattr(packet.wlan, 'fc_type_subtype'):
                wlan_info = packet.wlan
                if hasattr(wlan_info, 'ta'):
                    transmitter = wlan_info.ta
                    receiver = wlan_info.ra
                    arrival_time = str(packet.frame_info.time_epoch)
                    timestamp = float(arrival_time)
                    rss = int(packet.wlan_radio.signal_dbm)  # Received Signal Strength
                    snr = int(packet.wlan_radio.snr)  # Signal Noise Ratio
                    self.add_measurement(transmitter, timestamp, rss, snr)
                    self.add_measurement(receiver, timestamp, rss, snr)
        end = time.time()
        print('time to analyse: ', end - start, ' seconds')
        return self.measurements

