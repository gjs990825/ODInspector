import json
from dataclasses import dataclass
from shapely.geometry import Polygon


@dataclass()
class ODResult:
    confidence: str
    label: str
    points: list[int]
    type: str

    def __init__(self, confidence: str, label: str, points: list[int], type: str):
        self.confidence = confidence
        self.label = label
        self.points = points
        self.type = type

    @staticmethod
    def from_json_string(json_string):
        return ODResult.from_json(json.loads(json_string))

    @staticmethod
    def from_json(j):
        results = []
        for i in j:
            results.append(ODResult(**i))
        return results

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

    def get_anchor4(self):
        if self.type != 'rectangle':
            raise TypeError('Only rectangle can have 4 point anchor')
        return (self.points[0], self.points[1]), \
               (self.points[0], self.points[3]), \
               (self.points[2], self.points[3]), \
               (self.points[2], self.points[1])

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ODResult):
            return False
        return self.points == o.points and self.label == o.label and self.type == o.type and self.confidence == o.confidence


@dataclass
class Model:
    name: str
    weight_path: str
    class_path: str
    classes: list[str]

    def __init__(self, name, weight_path, class_path, classes=None):
        if classes is None:
            classes = []
        self.name = name
        self.weight_path = weight_path
        self.class_path = class_path
        self.classes = classes
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


class ODServiceInterface:
    PATH_DETECT_WITH_BINARY = "/api/v1/detect_with_binary/"
    PATH_DETECT_WITH_BINARY_FOR_IMAGE_RESULT = "/api/v1/detect_with_binary_for_image_result/"
    PATH_DETECT_WITH_JSON = "/api/v1/detect_with_json/"
    PATH_LIST_MODELS = "/api/v1/model/list"
    PATH_SET_MODEL = "/api/v1/model/set"

    class NoSuchModelException(Exception):
        pass

    def __init__(self):
        self.models = Model.from_json_string(open('./model_config.json').read())
        if len(self.models) == 0:
            raise FileNotFoundError("Bad config")
        self.current_model = self.models[0]

    def get_available_weights(self) -> list[Model]:
        return self.models

    def set_current_model(self, model_name: str):
        if model_name == self.current_model.name:
            print('No need to change model')
            return
        if model_name not in [model.name for model in self.models]:
            raise self.NoSuchModelException()
        self.current_model = next(model for model in self.models if model.name == model_name)

    def get_current_model_name(self):
        return self.current_model.name

    def detect(self, image) -> list[ODResult]:
        raise NotImplementedError

    def detect_for_image_result(self, image):
        raise NotImplementedError

    def detect_using(self, image, weight_name: str):
        if weight_name is not None:
            self.set_current_model(weight_name)
        return self.detect(image)
