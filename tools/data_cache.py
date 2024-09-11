from typing import Dict, Tuple, List
import json

import numpy as np


class DataCache:
    def __init__(
        self, split_data_path: str, target_path: str, features_path: Dict[str, str]
    ):
        self._load_split_data(split_data_path)
        self._load_target(target_path)
        self._load_features(features_path)
        self._crossval_split_names = [
            k for k in self.split_data.keys() if "cross_val" in k
        ]

    def _load_split_data(self, split_data_path: str) -> None:
        with open(split_data_path, "r") as f:
            self.split_data = json.load(f)

    def _load_features(self, features_path: Dict[str, str]) -> None:
        self.features = {}
        for feature_name, path in features_path.items():
            self.features[feature_name] = np.loadtxt(path, skiprows=1, delimiter=",")

    def _load_target(self, target_path: str) -> None:
        self.target = np.loadtxt(target_path, skiprows=1)

    def load_split(
        self, feature_name: str, split_name: str, split_label: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        split_indices = np.asarray(self.split_data[split_name][split_label])
        X = self.features[feature_name][split_indices]
        y = self.target[split_indices]
        return X, y

    def get_crossval_split_names(self) -> List[str]:
        return self._crossval_split_names

    def get_feature_names(self) -> List[str]:
        return list(self.features.keys())
