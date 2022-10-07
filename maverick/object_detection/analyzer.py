import cv2
from shapely.geometry import Polygon

from maverick.object_detection.utils import in_target_proportion, draw_polygon_outline
from maverick.object_detection.api.v1 import ODResult


class ODResultAnalyzer:
    def __init__(self):
        self.last_results = []

    def analyze(self, results: list[ODResult]):
        raise NotImplementedError

    def draw_conclusion(self, image):
        raise NotImplementedError

    def get_last_results(self):
        return self.last_results


class TrespassingAnalyzer(ODResultAnalyzer):
    def __init__(self, forbidden_areas: list[Polygon], detection_targets: list[str], threshold, abcd, color):
        super().__init__()
        self.color = color
        self.threshold = threshold
        self.detection_targets = detection_targets
        self.forbidden_areas = forbidden_areas
        self.abcd = abcd

    def analyze(self, results: list[ODResult]):
        self.last_results.clear()
        for result in results:
            if result.label not in self.detection_targets:
                continue
            for area in self.forbidden_areas:
                proportion = in_target_proportion(result.get_polygon(self.abcd), area)
                if proportion >= self.threshold:  # and result not in self.last_results??
                    self.last_results.append(result)

    def draw_conclusion(self, image):
        thickness = max(int(min((image.shape[1], image.shape[0])) / 150), 1)
        self.draw_forbidden_area(image, thickness, self.color)
        for result in self.last_results:
            label = f'{result.label} trespassing'
            p1, p2 = result.get_anchor2()
            cv2.rectangle(image, p1, p2, self.color, thickness)
            cv2.putText(image, label, (result.points[0], result.points[1] - thickness), cv2.FONT_HERSHEY_COMPLEX, 1,
                        self.color, 2)
            draw_polygon_outline(image, result.get_polygon(self.abcd), thickness, self.color)

    def draw_forbidden_area(self, image, thickness, color):
        for forbidden_area in self.forbidden_areas:
            draw_polygon_outline(image, forbidden_area, thickness, color)
