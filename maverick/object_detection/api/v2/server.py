import io
from logging.config import dictConfig

import numpy
from PIL import Image
from flask import Flask
from flask import request

from maverick.object_detection.api.v2 import ODServiceOverNetworkConfig, ODServiceInterface

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

app = Flask("ObjectDetectionServiceV2")
service: ODServiceInterface


@app.post(ODServiceOverNetworkConfig.PATH_DETECT_WITH_BINARY)
@app.route(ODServiceOverNetworkConfig.PATH_DETECT_WITH_BINARY, methods=['POST'])
def detection_using_binary():
    model_names = request.args.getlist('model_names')
    pil_image = Image.open(io.BytesIO(request.get_data()))
    np_image = numpy.array(pil_image)
    results = service.do_detections(np_image, model_names)
    return results


@app.route(ODServiceOverNetworkConfig.PATH_LIST_MODELS)
def list_models():
    return service.models
