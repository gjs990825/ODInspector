import json
import os
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


class ObjectDetectionServiceInterface:
    class NoSuchWeightException(Exception):
        pass

    def __init__(self):
        self.weight_files = os.listdir('weights')
        if len(self.weight_files) == 0:
            raise FileNotFoundError("No weight file found!")
        self.current_weight = self.weight_files[0]

    def reload_weights(self):
        self.__init__()  # is this legal?

    def get_available_weights(self) -> list[str]:
        return self.weight_files

    def set_current_weight(self, weight_name: str):
        if weight_name not in self.weight_files:
            raise self.NoSuchWeightException()
        self.current_weight = weight_name

    def get_current_weight(self):
        return self.current_weight

    def detect(self, image) -> list[ODResult]:
        raise NotImplementedError

    def detect_using(self, image, weight_name: str):
        if weight_name is not None:
            self.set_current_weight(weight_name)
        return self.detect(image)
