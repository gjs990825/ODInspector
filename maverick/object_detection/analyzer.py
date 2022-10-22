from __future__ import annotations

import json
import logging
import os.path
import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy
from PIL import Image
from shapely.geometry import Polygon

from deep_sort import build_tracker
from deep_sort.utils.parser import get_config
from maverick.object_detection.api.v1 import ODResult
from maverick.object_detection.utils import get_line_thickness, xyxy_to_xywh
from maverick.object_detection.utils.polygon import draw_polygon_outline, in_target_proportion, get_polygon_points


class ODResultAnalyzer(ABC):
    def __init__(self, ignore_below_frames: int, minimum_saving_interval: int,
                 confidence_filter: Optional[dict[str, float]],
                 saving_path='./image_output'):
        self.frame_counter = 0
        self.saving_frame_passed = 0
        self.ignore_below_frames = ignore_below_frames
        self.last_results: list[ODResult] = []
        self.drew_results: list[ODResult] = []
        self.current_image: Optional[numpy.ndarray] = None
        self.minimum_saving_interval = minimum_saving_interval
        self.saving_path = saving_path
        self.confidence_filter = confidence_filter

    def filter(self, results: list[ODResult]) -> list[ODResult]:
        if self.confidence_filter is None:
            return results
        after = []
        for result in results:
            if result.label not in self.confidence_filter:
                continue
            if float(result.confidence) >= self.confidence_filter[result.label]:
                after.append(result)
            else:
                logging.info(f'Result: {result} has been ditched')
        return after

    def analyze(self, results: list[ODResult], image: numpy.ndarray):
        results = self.filter(results)

        self.current_image = image
        self.last_results.clear()
        self.drew_results.clear()
        self.do_analyzing(results)
        self.last_results = list(set(self.last_results))

        if len(self.last_results) == 0:
            self.frame_counter = 0
        else:
            self.frame_counter += 1
            self.saving_frame_passed += 1

        # check saving options
        if self.minimum_saving_interval > 0:
            if not self.in_ignoring_range() and self.saving_frame_passed > self.minimum_saving_interval:
                self.save()
                self.saving_frame_passed = 0

        return results

    @abstractmethod
    def do_analyzing(self, results: list[ODResult]):
        pass

    def redirect_saving_path(self, path: str):
        self.saving_path = path

    def in_ignoring_range(self):
        return self.frame_counter < self.ignore_below_frames

    def overlay_conclusion(self, image):
        raise NotImplementedError

    def get_last_results(self):
        return self.last_results

    def get_drew_results(self):
        return self.drew_results

    def save(self, name=None):
        if self.current_image is None:
            return

        image = self.current_image.copy()
        self.overlay_conclusion(image)

        if name is None:
            name = f'{time.time()}.jpg'
        image_path = os.path.join(self.saving_path, name)
        logging.info(f'saving to {image_path}')
        Image.fromarray(image, 'RGB').save(image_path)

    @classmethod
    def from_json(cls, json_obj) -> ODResultAnalyzer:
        return cls(**json_obj)

    @classmethod
    def from_file(cls, path):
        analyzers = []
        with open(path, 'r') as f:
            for analyzer in json.load(f):
                analyzers.append(cls.from_json(analyzer))
        return analyzers


class TrespassingAnalyzer(ODResultAnalyzer):
    forbidden_areas: list[Polygon]
    detection_targets: list[str]
    threshold: float
    abcd: tuple[int, int, int, int]
    color: tuple[int, int, int]

    def __init__(self,
                 forbidden_areas: list[list[int]],
                 detection_targets: list[str],
                 threshold: float,
                 abcd: tuple[int, int, int, int],
                 color: tuple[int, int, int],
                 ignore_below_frames: int = 0,
                 minimum_saving_interval: int = -1,
                 confidence_filter: dict[str, float] = None,
                 prompt: str = 'Trespassing: {detection_target}',
                 **kwargs):
        super().__init__(ignore_below_frames, minimum_saving_interval, confidence_filter)
        self.color = color
        self.threshold = threshold
        self.detection_targets = detection_targets
        self.forbidden_areas = [Polygon(points) for points in forbidden_areas]
        self.abcd = abcd
        self.frame_counter = 0
        self.prompt = prompt
        logging.warning(f'kwargs: {kwargs}')

    def do_analyzing(self, results: list[ODResult]):
        for result in results:
            if result.label not in self.detection_targets:
                continue
            for area in self.forbidden_areas:
                proportion = in_target_proportion(result.get_polygon(self.abcd), area)
                if proportion >= self.threshold:  # and result not in self.last_results??
                    self.last_results.append(result)

    def overlay_conclusion(self, image):
        thickness = get_line_thickness(image)
        for forbidden_area in self.forbidden_areas:
            draw_polygon_outline(image, forbidden_area, thickness, self.color)
        if self.in_ignoring_range():
            self.drew_results = []
            return

        self.drew_results = self.last_results
        for result in self.last_results:
            draw_polygon_outline(image, result.get_polygon(), thickness, self.color)
            draw_polygon_outline(image, result.get_polygon(self.abcd), thickness, self.color)
            prompt = self.prompt.replace('{detection_target}', result.label)
            cv2.putText(image, prompt, (result.points[0], result.points[1] - thickness), cv2.FONT_HERSHEY_COMPLEX, 1,
                        self.color, 2)


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
                 color_detection: tuple[int, int, int],
                 ignore_below_frames: int = 0,
                 minimum_saving_interval: int = -1,
                 confidence_filter: dict[str, float] = None,
                 prompt: str = 'Warning: {detection_target} in {inspection_target}',
                 **kwargs):
        super().__init__(ignore_below_frames, minimum_saving_interval, confidence_filter)
        self.inspection_targets = inspection_targets
        self.detection_targets = detection_targets
        self.threshold = threshold
        self.abcd = abcd
        self.color_detection = color_detection
        self.color_inspection = color_inspection
        self.prompt = prompt
        self.mapping: dict[ODResult, ODResult] = dict()
        logging.warning(f'kwargs: {kwargs}')

    def inspection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.inspection_targets, results))

    def detection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.detection_targets, results))

    def do_analyzing(self, results: list[ODResult]):
        self.mapping.clear()
        for inspection_target in self.inspection_target_filter(results):
            p2 = inspection_target.get_polygon()
            for detection_target in self.detection_target_filter(results):
                proportion = in_target_proportion(detection_target.get_polygon(self.abcd), p2)
                if proportion > self.threshold:
                    self.last_results.append(inspection_target)
                    self.last_results.append(detection_target)
                    self.mapping[detection_target] = inspection_target

    def overlay_conclusion(self, image):
        if self.in_ignoring_range():
            self.last_results = []
            return
        thickness = get_line_thickness(image)
        for inspection_target in self.inspection_target_filter(self.last_results):
            draw_polygon_outline(image, inspection_target.get_polygon(), thickness, self.color_inspection)

        for detection_target in self.detection_target_filter(self.last_results):
            draw_polygon_outline(image, detection_target.get_polygon(), thickness, self.color_detection)
            draw_polygon_outline(image, detection_target.get_polygon(self.abcd), thickness, self.color_detection)

            prompt = self.prompt.replace('{detection_target}', detection_target.label)
            prompt = prompt.replace('{inspection_target}', self.mapping[detection_target].label)
            cv2.putText(image, prompt, (detection_target.points[0], detection_target.points[1] - thickness),
                        cv2.FONT_HERSHEY_COMPLEX, 1, self.color_detection, 2)

        self.drew_results = self.last_results


class DeepSortPedestrianAnalyzer(ODResultAnalyzer):
    def __init__(self,
                 color: tuple[int, int, int],
                 **kwargs):
        super().__init__(0, -1, None)
        cfg = get_config(config_file="deep_sort/configs/deep_sort.yaml")
        cfg.USE_FASTREID = False
        self.tracker = build_tracker(cfg, use_cuda=True)
        self.size = 1280, 720
        self.outputs = None
        self.person_results: list[ODResult] = []
        self.max_person_id = 0
        self.color = color
        logging.warning(f'kwargs: {kwargs}')

    def do_analyzing(self, results: list[ODResult]):
        person_results = list(filter(lambda x: x.label == 'person', results))
        self.person_results = person_results

        bbox_xywh = numpy.ndarray((0, 4), dtype=float)
        confidences = []

        for item in person_results:
            xywh = xyxy_to_xywh(item.get_xyxy())
            bbox_xywh = numpy.append(bbox_xywh, numpy.array(xywh, dtype=float).reshape(1, 4), axis=0)
            confidences.append(float(item.confidence))

        outputs = self.tracker.update(bbox_xywh, confidences, self.current_image)

        # draw boxes for visualization
        if len(outputs) == 0:
            return

        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        logging.debug(f'xyxy: {bbox_xyxy}, id: {identities}')

        for xyxy, identity, confidence in zip(bbox_xyxy, identities, confidences):
            identity = int(identity)
            self.max_person_id = max(self.max_person_id, identity)
            od_result = ODResult(str(confidence), 'person', xyxy, 'rectangle', identity)
            self.last_results.append(od_result)

    def overlay_conclusion(self, image):
        cv2.putText(image, f'Pedestrian Count: {self.max_person_id}',
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, self.color, 2)

        for result in self.last_results:
            text = 'PERSON-{} {:.2f}'.format(result.object_id, float(result.confidence))
            thickness = get_line_thickness(image)
            p1, p2 = result.get_anchor2()
            cv2.rectangle(image, p1, p2, self.color, thickness)
            cv2.putText(image, text, (result.points[0], result.points[1] - thickness), cv2.FONT_HERSHEY_COMPLEX, 1,
                        self.color, 2)
        self.drew_results = self.person_results


class ObjectMissingAnalyzer(ODResultAnalyzer):
    inspection_targets: Optional[list[str]] = None
    inspection_areas: Optional[list[Polygon]] = None

    def __init__(self,
                 detection_targets: list[str],
                 threshold: float,
                 abcd: tuple[int, int, int, int],
                 color_inspection: tuple[int, int, int],
                 ignore_below_frames: int = 0,
                 minimum_saving_interval: int = -1,
                 confidence_filter: dict[str, float] = None,
                 prompt: str = 'Warning: {detection_target} not in {inspection_target}',
                 **kwargs):
        super().__init__(ignore_below_frames, minimum_saving_interval, confidence_filter)

        if 'inspection_targets' in kwargs:
            self.inspection_targets = kwargs['inspection_targets']
        if 'inspection_areas' in kwargs:
            self.inspection_areas = [Polygon(points) for points in kwargs['inspection_areas']]

        self.detection_targets = detection_targets
        self.threshold = threshold
        self.abcd = abcd
        self.color_inspection = color_inspection
        self.prompt = prompt
        self.mapping: dict[ODResult, list[str]] = dict()
        logging.warning(f'kwargs: {kwargs}')

    def inspection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.inspection_targets, results))

    def detection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.detection_targets, results))

    def do_analyzing(self, results: list[ODResult]):
        # inspection_targets = []
        # if self.inspection_targets is None:
        #     for inspection_areas in self.inspection_areas:
        #         inspection_targets.append(
        #             ODResult('1', 'AREA', get_polygon_points(inspection_areas), 'rectangle')
        #         )
        # else:
        #     inspection_targets = self.inspection_target_filter(results)

        for inspection_target in self.inspection_target_filter(results):
            objs = []
            p2 = inspection_target.get_polygon()
            for detection_target in self.detection_target_filter(results):
                proportion = in_target_proportion(detection_target.get_polygon(self.abcd), p2)
                if proportion > self.threshold:
                    objs.append(detection_target.label)

            for detection_target in self.detection_targets:
                if detection_target not in objs:
                    self.last_results.append(inspection_target)
                    if inspection_target not in self.mapping:
                        self.mapping[inspection_target] = [detection_target]
                    else:
                        self.mapping[inspection_target].append(detection_target)

    def overlay_conclusion(self, image):
        if self.in_ignoring_range():
            self.last_results = []
            return
        thickness = get_line_thickness(image)

        # if self.inspection_targets is None:
        #     for area in self.inspection_areas:
        #         draw_polygon_outline(area)
        #     # cv2.putText(image, f'{self.last_results} missing',(50, 50),
        #     #             cv2.FONT_HERSHEY_COMPLEX, 1, self.color_inspection, 2)
        #     return

        for result in self.last_results:
            draw_polygon_outline(image, result.get_polygon(), thickness, self.color_inspection)

            cv2.putText(image, f'{self.mapping[result]} missing',
                        (result.points[0], result.points[1] - thickness),
                        cv2.FONT_HERSHEY_COMPLEX, 1, self.color_inspection, 2)

        self.drew_results = self.last_results
