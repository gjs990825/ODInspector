from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy
from shapely.geometry import Polygon


@dataclass()
class ODResult:
    confidence: str
    label: str
    points: list[int]
    type: str
    object_id: Optional[int]

    def __init__(self, confidence: str, label: str, points: list[int], type: str, object_id: int = None):
        self.confidence = confidence
        self.label = label
        self.points = points
        self.type = type
        self.object_id = object_id

    @staticmethod
    def from_json_string(json_string):
        return ODResult.from_json(json.loads(json_string))

    @staticmethod
    def from_json(j):
        results = []
        for i in j:
            results.append(ODResult(**i))
        return results

    def get_point_at(self, a, b):
        """
        |---(a%)----→
        |    ↓      |
        |---→*←-----|(c%)
        |    ↑      |
        |-----------↓
        """
        x_start = self.points[0]
        y_start = self.points[1]
        dx = self.points[2] - self.points[0]
        dy = self.points[3] - self.points[1]

        x = int(dx * (a / 100) + x_start)
        y = int(dy * (b / 100) + y_start)

        return x, y

    def get_center_point(self):
        return self.get_point_at(50, 50)

    def get_polygon(self, abcd=(0, 100, 0, 100)):
        """
        |---(a%)--------(b%)---→
        |    |           |     |
        |----|-----------|-----|(c%)
        |    |###########|     |
        |    |#Take This#|     |
        |    |###########|     |
        |----|-----------|-----|(d%)
        |    |           |     |
        |----------------------↓
        """
        if self.type == 'rectangle':
            x_start = self.points[0]
            y_start = self.points[1]
            dx = self.points[2] - self.points[0]
            dy = self.points[3] - self.points[1]
            a, b, c, d = abcd
            xa = int(dx * (a / 100) + x_start)
            xb = int(dx * (b / 100) + x_start)
            yc = int(dy * (c / 100) + y_start)
            yd = int(dy * (d / 100) + y_start)
            return Polygon(((xa, yc), (xa, yd), (xb, yd), (xb, yc)))
        raise NotImplementedError

    def get_two_points_list(self):
        return self.points

    def get_anchor2(self):
        if self.type != 'rectangle':
            raise TypeError('Only rectangle can have 2 point anchor')
        return (self.points[0], self.points[1]), (self.points[2], self.points[3])

    def get_xyxy(self):
        return self.points

    def get_anchor4(self):
        if self.type != 'rectangle':
            raise TypeError('Only rectangle can have 4 point anchor')
        return (self.points[0], self.points[1]), \
               (self.points[0], self.points[3]), \
               (self.points[2], self.points[3]), \
               (self.points[2], self.points[1])

    @staticmethod
    def confidence_filter(results: list[ODResult], confidence_filter: Optional[dict[str, float]]) -> list[ODResult]:
        if confidence_filter is None:
            return results
        after = []
        for result in results:
            if result.label not in confidence_filter:
                continue
            if float(result.confidence) >= confidence_filter[result.label]:
                after.append(result)
            else:
                logging.info(f'Result: {result} has been ditched')
        return after

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ODResult):
            return False
        if self.object_id is None or other.object_id is None:
            return False
        return self.points == other.points \
               and self.label == other.label \
               and self.type == other.type \
               and self.confidence == other.confidence \
               and self.object_id == other.object_id

    def is_same_object(self, other: ODResult):
        if self.object_id is None or other.object_id is None:
            return False
        return self.object_id == other.object_id

    def __hash__(self):
        return hash(str(self))


@dataclass
class Model:
    name: str
    weight_path: str
    class_path: str
    classes: list[str]
    class_alter_names: list[str]

    class ClassNameConverter:
        class_alter_names: Optional[dict[str, str]] = None

        def __init__(self, class_names: list[str] = None, class_alter_names: list[str] = None):
            if class_names is None or class_alter_names is None:
                self.class_alter_names = None
                return

            self.class_alter_names = dict()
            for name, alter_name in zip(class_names, class_alter_names):
                self.class_alter_names[name] = alter_name

        def __call__(self, class_name):
            if self.class_alter_names is None:
                return class_name

            if class_name in self.class_alter_names:
                return self.class_alter_names[class_name]

            logging.warning(f'no class name matched')
            return class_name

    def __init__(self, name, weight_path, class_path, classes=None, class_alter_names: list[str] = None):
        if classes is None:
            classes = []
        self.name = name
        self.weight_path = weight_path
        self.class_path = class_path
        self.classes = classes

        self.class_alter_names = class_alter_names
        self.class_name_converter = self.ClassNameConverter(classes, class_alter_names)

        if len(classes) == 0 and len(self.class_path) != 0:
            with open(self.class_path, 'r') as f:
                self.classes = f.read().splitlines()

    def get_classes(self):
        return self.classes if self.classes is not None else []

    def get_class_number(self):
        return len(self.get_classes())

    @staticmethod
    def from_json(j):
        models = []
        for i in j:
            models.append(Model(**i))
        return models

    @staticmethod
    def from_json_string(j_str):
        return Model.from_json(json.loads(j_str))


class ODServiceInterface(ABC):
    models: list[Model]
    current_model: Optional[Model]

    class NoSuchModelException(Exception):
        pass

    def __init__(self):
        self.models = []
        self.current_model = None

    def get_models(self) -> list[Model]:
        if len(self.models) == 0:
            self.update_models()
        return self.models

    def get_model_names(self):
        return [model.name for model in self.get_models()]

    @abstractmethod
    def update_models(self) -> None:
        pass

    def get_current_classes(self) -> list[str]:
        return self.current_model.classes

    def set_current_model(self, model_name: str):
        if model_name not in [model.name for model in self.models]:
            raise self.NoSuchModelException()
        self.current_model = next(model for model in self.models if model.name == model_name)

    def get_current_model_name(self):
        return self.current_model.name

    @abstractmethod
    def do_detections(self, image: numpy.ndarray) -> list[ODResult]:
        pass

    def detect_using(self, image, weight_name: str):
        if weight_name is not None:
            self.set_current_model(weight_name)
        return self.do_detections(image)


class ODServiceOverNetworkConfig:
    PATH_DETECT_WITH_BINARY = "/api/v1/detect_with_binary/"
    PATH_DETECT_WITH_BINARY_FOR_IMAGE_RESULT = "/api/v1/detect_with_binary_for_image_result/"
    PATH_DETECT_WITH_JSON = "/api/v1/detect_with_json/"
    PATH_LIST_MODELS = "/api/v1/model/list"
    PATH_SET_MODEL = "/api/v1/model/set"
