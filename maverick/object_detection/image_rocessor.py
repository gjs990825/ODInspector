import cv2

from maverick.object_detection.analyzer import ODResultAnalyzer
from maverick.object_detection.api.v1 import ODResult, Model
from maverick.object_detection.utils import create_in_memory_image, get_colors


class ImageProcessorInterface:
    def __init__(self, binary_result=False, analyzers: list[ODResultAnalyzer] = None):
        self.binary_result = binary_result
        self.models = []
        self.current_model = None
        self.current_classes = []
        self.colors = []
        self.analyzers = analyzers if analyzers is not None else []

    def detect(self, image):
        in_memory_image = create_in_memory_image(image)
        if not self.binary_result:
            cp = image.copy()
            results = self.request_detection(in_memory_image)
            drew_results = []
            for analyzer in self.analyzers:
                analyzer.analyze(results)
                analyzer.draw_conclusion(cp)
                drew_results.extend(analyzer.get_last_results())
            for drew_result in set(drew_results):
                results.remove(drew_result)
            self.draw_result_image(cp, results)
            return cp
        else:
            return self.request_detection_for_image_result(in_memory_image)

    def request_detection(self, in_memory_image) -> list[ODResult]:
        raise NotImplementedError

    def request_detection_for_image_result(self, in_memory_image):
        raise NotImplementedError

    def draw_result_image(self, image, results: list[ODResult]):
        thickness = max(int(min((image.shape[1], image.shape[0])) / 150), 1)
        for result in results:
            label = '{} {:.2f}'.format(result.label, float(result.confidence))
            color = self.get_class_color(result.label)
            p1, p2 = result.get_anchor2()
            cv2.rectangle(image, p1, p2, color, thickness)
            cv2.putText(image, label, (result.points[0], result.points[1] - thickness), cv2.FONT_HERSHEY_COMPLEX, 1,
                        color, 2)

    def get_models(self) -> list[Model]:
        if len(self.models) == 0:
            self.update_models()
        return self.models

    def update_models(self):
        raise NotImplementedError

    def get_current_classes(self) -> list[str]:
        return self.current_model.classes

    def set_current_model(self, model_name):
        self.current_model = next(model for model in self.models if model.name == model_name)
        self.current_classes = self.current_model.classes
        self.colors = get_colors(self.current_classes)

    def set_analyzers(self, analyzers: list[ODResultAnalyzer]):
        self.analyzers = analyzers

    def get_class_color(self, class_name):
        try:
            index = self.current_classes.index(class_name)
            return self.colors[index]
        except ValueError:
            return 0xFF, 0xFF, 0xFF

    def get_model_names(self):
        return [model.name for model in self.get_models()]
