import json
from dataclasses import dataclass


@dataclass()
class ODResult:
    def __init__(self, confidence, label, points, type):
        self.confidence = confidence
        self.label = label
        self.points = points
        self.type = type

    @staticmethod
    def from_json_string(json_string):
        ODResult.from_json(json.loads(json_string))

    @staticmethod
    def from_json(j):
        results = []
        for i in j:
            results.append(ODResult(**i))
        return results

    confidence: str
    label: str
    points: list[int]
    type: str


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
        if len(classes) == 0:
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
