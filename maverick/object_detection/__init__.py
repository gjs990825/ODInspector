import cv2
import numpy

from maverick.object_detection.analyzer import ODResultAnalyzer
from maverick.object_detection.api.v1 import ODResult, ODServiceInterface
from maverick.object_detection.utils import get_colors


class ImageProcessingHelper:
    colors: list[tuple[int, int, int]]
    analyzers: list[ODResultAnalyzer]
    service: ODServiceInterface

    def __init__(self, service: ODServiceInterface, analyzers: list[ODResultAnalyzer] = None):
        super().__init__()
        self.service = service
        self.colors = []
        self.analyzers = analyzers if analyzers is not None else []

    def detect(self, image: numpy.ndarray) -> numpy.ndarray:
        cp = image.copy()
        results = self.service.do_detections(image)
        drew_results = []
        for analyzer in self.analyzers:
            results = analyzer.analyze(results, image)
            analyzer.overlay_conclusion(cp)
            drew_results.extend(analyzer.get_drew_results())
        for drew_result in set(drew_results):
            results.remove(drew_result)
        self.draw_result_image(cp, results)
        return cp

    def draw_result_image(self, image: numpy.ndarray, results: list[ODResult]):
        thickness = max(int(min((image.shape[1], image.shape[0])) / 150), 1)
        for result in results:
            label = result.label
            confidence = float(result.confidence)
            if result.object_id is None:
                text = '{} {:.2f}'.format(label, confidence)
            else:
                text = '{}({}) {:.2f}'.format(label, str(result.object_id)[:5], confidence)
            color = self.get_class_color(result.label)
            p1, p2 = result.get_anchor2()
            cv2.rectangle(image, p1, p2, color, thickness)
            cv2.putText(image, text, (result.points[0], result.points[1] - thickness), cv2.FONT_HERSHEY_COMPLEX, 1,
                        color, 2)

    def set_current_model(self, model_name: str):
        self.service.set_current_model(model_name)
        self.colors = get_colors(self.service.get_current_classes())

    def set_analyzers(self, analyzers: list[ODResultAnalyzer]):
        self.analyzers = analyzers

    def get_class_color(self, class_name: str) -> tuple[int, int, int]:
        try:
            index = self.service.get_current_classes().index(class_name)
            return self.colors[index]
        except ValueError:
            return 0xFF, 0xFF, 0xFF

    def get_model_names(self):
        return self.service.get_model_names()
