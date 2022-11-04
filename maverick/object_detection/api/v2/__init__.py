import json
from dataclasses import dataclass

import numpy

from abc import ABC, abstractmethod
from maverick.object_detection.api.v1 import Model, ODResult


@dataclass
class DetectionSequenceConfig:
    name: str
    model_names: list[str]

    @staticmethod
    def from_json(json_obj):
        return DetectionSequenceConfig(**json_obj)

    @staticmethod
    def from_file(path):
        configs = []
        with open(path, 'r', encoding='utf-8') as f:
            for config in json.load(f):
                configs.append(DetectionSequenceConfig.from_json(config))
        return configs


class ODServiceInterface(ABC):
    models: list[Model]

    class NoSuchModelException(Exception):
        pass

    def __init__(self):
        self.models = []

    def get_models(self) -> list[Model]:
        if len(self.models) == 0:
            self.update_models()
        return self.models

    def get_model_names(self):
        return [model.name for model in self.get_models()]

    @abstractmethod
    def update_models(self) -> None:
        pass

    @abstractmethod
    def do_detections(self, image: numpy.ndarray, model_names: list[str]) -> list[ODResult]:
        pass

    def convert_name(self, label) -> str:
        for model in self.models:
            if label in model.classes:
                return model.class_name_converter(label)
        return label

    def get_current_classes(self) -> list[str]:
        classes = set()
        for model in self.models:
            classes = classes.union(model.classes)
        return list(classes)


class ODServiceOverNetworkConfig:
    PATH_DETECT_WITH_BINARY = "/api/v2/detect_with_binary/"
    PATH_LIST_MODELS = "/api/v2/model/list"
