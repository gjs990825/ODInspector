import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy

from maverick.object_detection.api.v1 import Model, ODResult


@dataclass
class DetectionConfig:
    name: str
    model_names: list[str]
    analyzer_files: list[str]

    @staticmethod
    def from_json(json_obj):
        return DetectionConfig(**json_obj)

    @staticmethod
    def from_file(path):
        configs = []
        with open(path, 'r', encoding='utf-8') as f:
            for config in json.load(f):
                configs.append(DetectionConfig.from_json(config))
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
    def preload(self, model_name) -> None:
        pass

    @abstractmethod
    def update_models(self) -> None:
        pass

    @abstractmethod
    def do_detections(self, image: numpy.ndarray, model_names: list[str]) -> list[ODResult]:
        pass

    def convert_name(self, label: str) -> str:
        for model in self.models:
            if label in model.classes:
                return model.class_name_converter(label)
        return label

    @staticmethod
    def convert_name_for(label: str, models: list[Model]) -> str:
        for model in models:
            if label in model.classes:
                return model.class_name_converter(label)
        return label

    def get_name_converter(self, model_names: list[str]):
        models = [model for model in self.models if model.name in model_names]
        return lambda label, models=models: ODServiceInterface.convert_name_for(label, models)
        # TODO nani kore? WHY I did this?

    def get_current_classes(self) -> list[str]:
        classes = set()
        for model in self.models:
            classes = classes.union(model.classes)
        return list(classes)


class ODServiceOverNetworkConfig:
    PATH_DETECT_WITH_BINARY = "/api/v2/detect_with_binary/"
    PATH_LIST_MODELS = "/api/v2/model/list"
    PATH_PRELOAD_MODEL = "/api/v2/model/preload/"
