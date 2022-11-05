from __future__ import annotations

import json
import logging
import os.path
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy
from PIL import Image
from shapely.geometry import Polygon

from maverick.object_detection.api.v1 import ODResult
from maverick.object_detection.utils import get_line_thickness, xyxy_to_xywh, draw_text, get_colors, \
    get_euclidean_distance
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
        self.class_name_converter = None

    @staticmethod
    def draw_text(image: numpy.ndarray, text: str, position: tuple[int, int], color=(0, 0, 0), size=1.5):
        draw_text(image, text, position, color, size)

    def set_class_name_converter(self, converter=None):
        self.class_name_converter = converter

    def get_alter_name(self, name):
        if self.class_name_converter is None:
            return name
        return self.class_name_converter(name)

    def analyze(self, results: list[ODResult], image: numpy.ndarray):
        results = ODResult.confidence_filter(results, self.confidence_filter)

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

    @staticmethod
    def from_file(path) -> list[ODResultAnalyzer]:
        analyzers = []
        analyzer_dict = dict()
        for cls in ODResultAnalyzer.__subclasses__():
            analyzer_dict[cls.__name__] = cls
        with open(path, 'r', encoding='utf-8') as f:
            for analyzer in json.load(f):
                analyzer_name = analyzer['analyzer_name']
                if analyzer_name not in analyzer_dict:
                    logging.error(f'no analyzer named {analyzer_name}')
                    continue
                cls = analyzer_dict[analyzer_name]
                analyzers.append(cls.from_json(analyzer))
        return analyzers


class TrespassingAnalyzer(ODResultAnalyzer):
    forbidden_areas: list[Polygon]
    detection_targets: list[str]
    threshold: float
    abcd: tuple[int, int, int, int]
    color: tuple[int, int, int]

    def __init__(self,
                 forbidden_areas: list[list[tuple[int, int]]],
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
            prompt = self.prompt.replace('{detection_target}', self.get_alter_name(result.label))
            self.draw_text(image, prompt, (result.points[0], result.points[1] - thickness), self.color)


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

            detection_target_text = self.get_alter_name(detection_target.label)
            inspection_target_text = self.get_alter_name(self.mapping[detection_target].label)
            prompt = self.prompt.replace('{detection_target}', detection_target_text)
            prompt = prompt.replace('{inspection_target}', inspection_target_text)
            self.draw_text(image, prompt, (detection_target.points[0], detection_target.points[1] - thickness),
                           self.color_detection)

        self.drew_results = self.last_results


class DeepSortPedestrianAnalyzer(ODResultAnalyzer):
    def __init__(self,
                 color: tuple[int, int, int],
                 **kwargs):
        super().__init__(0, -1, None)

        from deep_sort import build_tracker
        from deep_sort.utils.parser import get_config

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
        self.draw_text(image, f'Pedestrian Count: {self.max_person_id}', (50, 50), self.color)

        for result in self.last_results:
            text = 'PERSON-{} {:.2f}'.format(result.object_id, float(result.confidence))
            thickness = get_line_thickness(image)
            p1, p2 = result.get_anchor2()
            cv2.rectangle(image, p1, p2, self.color, thickness)
            self.draw_text(image, text, (result.points[0], result.points[1] - thickness), self.color)
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
                 prompt: str = 'Warning: {missing_targets} not in {inspection_target}',
                 **kwargs):
        super().__init__(ignore_below_frames, minimum_saving_interval, confidence_filter)
        if 'inspection_targets' not in kwargs and 'inspection_areas' not in kwargs:
            raise KeyError(r"one of 'inspection_targets' and 'inspection_areas' needed")
        if 'inspection_targets' in kwargs and 'inspection_areas' in kwargs:
            raise KeyError(r"only one of 'inspection_targets' and 'inspection_areas' is needed")

        if 'inspection_targets' in kwargs:
            self.inspection_targets = kwargs['inspection_targets']
        if 'inspection_areas' in kwargs:
            self.inspection_areas = [Polygon(points) for points in kwargs['inspection_areas']]

        self.detection_targets = detection_targets
        self.threshold = threshold
        self.abcd = abcd
        self.color_inspection = color_inspection
        self.prompt = prompt
        self.mapping: dict[Union[ODResult, int], list[str]] = dict()
        logging.warning(f'kwargs: {kwargs}')

    def inspection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.inspection_targets, results))

    def detection_target_filter(self, results: list[ODResult]):
        return list(filter(lambda x: x.label in self.detection_targets, results))

    def get_area_id(self, area: Polygon):
        return self.inspection_areas.index(area)

    def do_analyzing(self, results: list[ODResult]):
        self.mapping.clear()
        if self.inspection_targets is None:
            inspection_objects = self.inspection_areas
        else:
            inspection_objects = self.inspection_target_filter(results)

        for inspection_target in inspection_objects:
            objs = []
            p2 = inspection_target.get_polygon() if isinstance(inspection_target, ODResult) else inspection_target
            for detection_target in self.detection_target_filter(results):
                proportion = in_target_proportion(detection_target.get_polygon(self.abcd), p2)
                if proportion > self.threshold:
                    objs.append(detection_target.label)

            for detection_target in self.detection_targets:
                if detection_target in objs:
                    continue

                if isinstance(inspection_target, ODResult):
                    self.last_results.append(inspection_target)
                    key = inspection_target
                else:
                    # is there a better solution? :\
                    self.last_results.append(ODResult('1.0', 'dummy', [0, 0, 100, 100], 'rectangle'))
                    key = self.get_area_id(inspection_target)

                if key not in self.mapping:
                    self.mapping[key] = [detection_target]
                else:
                    self.mapping[key].append(detection_target)

    def parse_list_text(self, lst: list[str]):
        text = '['
        length = len(lst)
        for i in range(length):
            name = self.get_alter_name(lst[i])
            if i != length - 1:
                text += f'{name}, '
            else:
                text += f'{name}'
        return text + ']'

    def overlay_conclusion(self, image):
        thickness = get_line_thickness(image)
        if self.inspection_targets is None:
            for area in self.inspection_areas:
                draw_polygon_outline(image, area, thickness, self.color_inspection)
        if self.in_ignoring_range():
            self.last_results = []
            return

        if self.inspection_targets is None:
            for area in self.inspection_areas:
                area_id = self.get_area_id(area)
                if area_id in self.mapping:
                    x, y = get_polygon_points(area)[0]
                    missing_targets = self.parse_list_text(self.mapping[area_id])
                    prompt = self.prompt.replace('{missing_targets}', missing_targets)
                    self.draw_text(image, prompt, (x + 50, y + 50), self.color_inspection)
            return

        for result in self.last_results:
            draw_polygon_outline(image, result.get_polygon(), thickness, self.color_inspection)

            missing_targets = self.parse_list_text(self.mapping[result])
            prompt = self.prompt.replace('{missing_targets}', missing_targets)
            prompt = prompt.replace('{inspection_target}', self.get_alter_name(result.label))
            self.draw_text(image, prompt, (result.points[0], result.points[1] - thickness),
                           self.color_inspection)

        self.drew_results = self.last_results


class UnsafePassingAnalyzer(ODResultAnalyzer):
    class Tracker:
        records: list[Record]

        @dataclass
        class Record:
            points: list[tuple[int, int]]
            counter_val: int

        def __init__(self, label, max_toleration: int, keep_obj_for: int) -> None:
            self.keep_obj_for = keep_obj_for
            self.label = label
            self.max_toleration = max_toleration
            self.records = []
            self.filter = {label: 0}
            self.counter = 0
            self.is_moving = False

        @staticmethod
        def get_average_distance(points: list[tuple[int, int]]):
            size = len(points)
            if size <= 1:
                return -1
            distances = []
            for i in range(size - 1):
                distances.append(get_euclidean_distance(points[i], points[i + 1]))
            return sum(distances) / (size - 1)

        def update(self, objs: list[ODResult]):
            objs = ODResult.confidence_filter(objs, self.filter)

            self.records = [self.Record(record.points[-self.keep_obj_for:], record.counter_val)
                            for record in self.records if self.counter - record.counter_val < self.keep_obj_for]

            for obj in objs:
                self.update_single(obj)

            flags = []
            for record in self.records:
                distance = self.get_average_distance(record.points)
                if distance > 1:
                    flags.append(1)
            self.is_moving = sum(flags) != 0

            self.counter += 1

        def update_single(self, obj):
            center_point = obj.get_center_point()
            candidate: Optional[tuple[float, UnsafePassingAnalyzer.Tracker.Record]] = None
            for record in self.records:
                last_point = record.points[-1]
                distance = get_euclidean_distance(center_point, last_point)
                if distance > self.max_toleration:
                    continue
                if candidate is None or distance < candidate[0]:
                    candidate = distance, record

            if candidate is None:
                self.records.append(self.Record([center_point], self.counter))
            else:
                candidate[1].points.append(center_point)
                candidate[1].counter_val = self.counter

    def __init__(self,
                 ignore_below_frames: int,
                 minimum_saving_interval: int,
                 color: tuple[int, int, int],
                 confidence_filter: Optional[dict[str, float]],
                 inspection_target: str,
                 max_toleration: int,
                 keep_obj_for: int,
                 saving_path='./image_output',
                 prompt: str = 'Warning: Unsafe passing',
                 **kwargs):
        logging.warning(f'kwargs: {kwargs}')
        self.prompt = prompt
        self.color = color
        super().__init__(ignore_below_frames, minimum_saving_interval, confidence_filter, saving_path)
        self.tracker = self.Tracker(inspection_target, max_toleration, keep_obj_for)
        self.trespassing_analyzer = TrespassingAnalyzer(ignore_below_frames=ignore_below_frames,
                                                        minimum_saving_interval=minimum_saving_interval,
                                                        confidence_filter=confidence_filter,
                                                        saving_path=saving_path,
                                                        color=color,
                                                        prompt=prompt,
                                                        **kwargs)

    def redirect_saving_path(self, path: str):
        super().redirect_saving_path(path)
        self.trespassing_analyzer.redirect_saving_path(path)

    def do_analyzing(self, results: list[ODResult]):
        self.tracker.update(results)

    def analyze(self, results: list[ODResult], image: numpy.ndarray):
        results = super().analyze(results, image)
        if self.tracker.is_moving:
            self.trespassing_analyzer.analyze(results, image)
        return results

    def overlay_conclusion(self, image):
        thickness = get_line_thickness(image)
        for record in self.tracker.records:
            points = record.points
            colors = get_colors(points)
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i + 1], colors[i], thickness)
            cv2.line(image, points[-1], points[-1], colors[-1], thickness * 3)

        if self.tracker.is_moving:
            self.draw_text(image, self.prompt, (100, 100), self.color)
            self.trespassing_analyzer.overlay_conclusion(image)
            self.drew_results = self.trespassing_analyzer.drew_results
