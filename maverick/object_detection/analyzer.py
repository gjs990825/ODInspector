import json

import cv2
from shapely.geometry import Polygon

from maverick.object_detection import get_line_thickness
from maverick.object_detection.api.v1 import ODResult
from maverick.object_detection.utils.polygon import draw_polygon_outline, in_target_proportion, get_polygon_points


class ODResultAnalyzer:
    def __init__(self):
        self.last_results = []
        self.drew_results = []

    def analyze(self, results: list[ODResult]):
        raise NotImplementedError

    def draw_conclusion(self, image):
        raise NotImplementedError

    def get_last_results(self):
        return self.last_results

    def get_drew_results(self):
        return self.drew_results


class TrespassingAnalyzer(ODResultAnalyzer):
    forbidden_areas: list[Polygon]
    detection_targets: list[str]
    threshold: float
    # use frame sync for correct timeing
    ignore_below_frames: int
    abcd: tuple[int, int, int, int]
    color: tuple[int, int, int]

    def __init__(self,
                 forbidden_areas: list[Polygon],
                 detection_targets: list[str],
                 threshold: float,
                 abcd: tuple[int, int, int, int],
                 color: tuple[int, int, int],
                 ignore_below_frames: int = 0):
        super().__init__()
        self.ignore_below_frames = ignore_below_frames
        self.color = color
        self.threshold = threshold
        self.detection_targets = detection_targets
        self.forbidden_areas = forbidden_areas
        self.abcd = abcd
        self.frame_counter = 0

    def analyze(self, results: list[ODResult]):
        self.last_results.clear()
        for result in results:
            if result.label not in self.detection_targets:
                continue
            for area in self.forbidden_areas:
                proportion = in_target_proportion(result.get_polygon(self.abcd), area)
                if proportion >= self.threshold:  # and result not in self.last_results??
                    self.last_results.append(result)
        if len(self.last_results) == 0:
            self.frame_counter = 0
        else:
            self.frame_counter += 1

    def draw_conclusion(self, image):
        thickness = get_line_thickness(image)
        self.draw_forbidden_area(image, thickness, self.color)
        if self.frame_counter < self.ignore_below_frames:
            self.drew_results = []
            return

        self.drew_results = self.last_results
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

    def __str__(self):
        areas = []
        for p in self.forbidden_areas:
            areas.append(get_polygon_points(p))
        return json.dumps({
            'forbidden_areas': areas,
            'detection_targets': self.detection_targets,
            'threshold': self.threshold,
            'ignore_below_frames': self.ignore_below_frames,
            'abcd': self.abcd,
            'color': self.color
        })

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    @staticmethod
    def from_json(json_obj):
        try:
            ignore_below_frames = json_obj['ignore_below_frames']
        except KeyError:
            ignore_below_frames = 0

        forbidden_areas = [Polygon(points) for points in json_obj['forbidden_areas']]
        return TrespassingAnalyzer(forbidden_areas,
                                   json_obj['detection_targets'],
                                   json_obj['threshold'],
                                   json_obj['abcd'],
                                   json_obj['color'],
                                   ignore_below_frames)

    @staticmethod
    def from_file(path):
        analyzers = []
        with open(path, 'r') as f:
            for analyzer in json.load(f):
                analyzers.append(TrespassingAnalyzer.from_json(analyzer))
        return analyzers


class IllegalEnteringAnalyzer(ODResultAnalyzer):
    inspection_targets: list[str]
    detection_targets: list[str]
    threshold: float
    color_inspection: tuple[int, int, int]
    color_detection: tuple[int, int, int]
    abcd: tuple[int, int, int, int]

    def __init__(self,
                 inspection_targets: list[str],
                 detection_targets: list[str],
                 threshold: float,
                 abcd: tuple[int, int, int, int],
                 color_inspection: tuple[int, int, int],
                 color_detection: tuple[int, int, int]):
        super().__init__()
        self.inspection_targets = inspection_targets
        self.detection_targets = detection_targets
        self.threshold = threshold
        self.abcd = abcd
        self.color_detection = color_detection
        self.color_inspection = color_inspection

    def inspection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.inspection_targets, results))

    def detection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.detection_targets, results))

    def analyze(self, results: list[ODResult]):
        self.last_results.clear()

        for inspection_target in self.inspection_target_filter(results):
            p2 = inspection_target.get_polygon()
            for detection_target in self.detection_target_filter(results):
                proportion = in_target_proportion(detection_target.get_polygon(self.abcd), p2)
                if proportion > self.threshold:
                    self.last_results.append(inspection_target)
                    self.last_results.append(detection_target)
        self.last_results = list(set(self.last_results))

    def draw_conclusion(self, image):
        thickness = get_line_thickness(image)
        for inspection_target in self.inspection_target_filter(self.last_results):
            draw_polygon_outline(image, inspection_target.get_polygon(), thickness, self.color_inspection)
        for detection_target in self.detection_target_filter(self.last_results):
            draw_polygon_outline(image, detection_target.get_polygon(), thickness, self.color_detection)
            draw_polygon_outline(image, detection_target.get_polygon(self.abcd), thickness, self.color_detection)
        self.drew_results = self.last_results

    # TODO Add configuration loading
