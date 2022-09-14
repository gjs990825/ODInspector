import base64
import io
from logging.config import dictConfig

from PIL import Image
from flask import Flask
from flask import request

from maverick.object_detection.api.v1 import ObjectDetectionServiceInterface

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask("ObjectDetectionService")
service: ObjectDetectionServiceInterface


@app.post("/api/v1/detect/")
@app.route("/api/v1/detect/<model_name>", methods=['POST'])
def detection_using_binary(model_name=None):
    image = Image.open(io.BytesIO(request.get_data()))
    if model_name is None:
        results = service.detect(image)
    else:
        results = service.detect_using(image, model_name)
    return results


@app.post("/api/v1/detect_using_json/")
@app.route("/api/v1/detect_using_json/<model_name>", methods=['POST'])
def detection_using_json(model_name=None):
    image_encoded = request.json['image']
    image_data = base64.b64decode(image_encoded)
    image = Image.open(io.BytesIO(image_data))
    if model_name is None:
        results = service.detect(image)
    else:
        results = service.detect_using(image, model_name)
    return results


@app.route("/api/v1/model/list")
def list_weights():
    return service.get_available_weights()


@app.route("/api/v1/model/set")
def set_weight():
    model_name = request.values['model_name']
    service.set_current_model(model_name)
    return 'OK'

# USAGE:
# from maverick.object_detection.api.v1 import ObjectDetectionServiceInterface
# from maverick.object_detection import server
#
#
# class ODService(ObjectDetectionServiceInterface):#
#     def set_current_model(self, model_name: str):
#         super(ODService, self).set_current_model(model_name)
#         # load model here
#
#     def detect(self, image):
#         if self.yolo is not None:
#             return # do detection here
#         logging.warning('No model selected')
#         return []
#
#
# if __name__ == '__main__':
#     server.service = ODService()
#     server.app.run()
