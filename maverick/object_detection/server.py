import base64
import io
import time
from logging.config import dictConfig

from PIL import Image
from flask import Flask
from flask import request

from maverick.object_detection.api.utils import create_in_memory_image_from_pil_image
from maverick.object_detection.api.v1 import ODServiceInterface

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


@app.post(ODServiceInterface.PATH_DETECT_WITH_BINARY)
@app.route(ODServiceInterface.PATH_DETECT_WITH_BINARY + '<model_name>', methods=['POST'])
def detection_using_binary(model_name=None):
    image = Image.open(io.BytesIO(request.get_data()))
    if model_name is None:
        results = service.detect(image)
    else:
        results = service.detect_using(image, model_name)
    return results


@app.post(ODServiceInterface.PATH_DETECT_WITH_BINARY_FOR_IMAGE_RESULT)
def detection_using_binary_for_image_result():
    t_s = time.time()
    image = Image.open(io.BytesIO(request.get_data()))
    print(f'open:{time.time()-t_s}')
    result_image = service.detect_for_image_result(image)
    print(f'od:{time.time() - t_s}')
    in_memory_image = create_in_memory_image_from_pil_image(result_image)
    print(f'create:{time.time() - t_s}')
    return in_memory_image


@app.post(ODServiceInterface.PATH_DETECT_WITH_JSON)
@app.route(ODServiceInterface.PATH_DETECT_WITH_JSON + '<model_name>', methods=['POST'])
def detection_using_json(model_name=None):
    image_encoded = request.json['image']
    image_data = base64.b64decode(image_encoded)
    image = Image.open(io.BytesIO(image_data))
    if model_name is None:
        results = service.detect(image)
    else:
        results = service.detect_using(image, model_name)
    return results


@app.route(ODServiceInterface.PATH_LIST_MODELS)
def list_models():
    return service.get_available_weights()


@app.route(ODServiceInterface.PATH_SET_MODEL)
def set_model():
    model_name = request.values['model_name']
    service.set_current_model(model_name)
    return 'OK'
