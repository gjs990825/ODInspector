import io
from logging.config import dictConfig

import numpy
from PIL import Image
from flask import Flask
from flask import request

from maverick.object_detection.api.v1 import ODServiceOverNetworkConfig, ODServiceInterface

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
service: ODServiceInterface


@app.post(ODServiceOverNetworkConfig.PATH_DETECT_WITH_BINARY)
@app.route(ODServiceOverNetworkConfig.PATH_DETECT_WITH_BINARY + '<model_name>', methods=['POST'])
def detection_using_binary(model_name=None):
    pil_image = Image.open(io.BytesIO(request.get_data()))
    np_image = numpy.array(pil_image)
    if model_name is None:
        results = service.do_detections(np_image)
    else:
        results = service.detect_using(np_image, model_name)
    return results


@app.route(ODServiceOverNetworkConfig.PATH_LIST_MODELS)
def list_models():
    return service.models


@app.route(ODServiceOverNetworkConfig.PATH_SET_MODEL)
def set_model():
    model_name = request.values['model_name']
    service.set_current_model(model_name)
    return 'OK'
