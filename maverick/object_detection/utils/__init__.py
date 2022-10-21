import colorsys
import os
import re
from io import BytesIO

import numpy
from PIL import Image
from numpy import ndarray


def camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def get_line_thickness(image):
    return max(int(min((image.shape[1], image.shape[0])) / 150), 1)


def save_numpy_image(image: numpy.ndarray, path, name):
    Image.fromarray(image, 'RGB').save(os.path.join(path, name + '.jpg'))


def xyxy_to_xywh(boxes_xyxy):
    boxes_xywh = list([0., 0., 0., 0.])
    boxes_xywh[0] = (boxes_xyxy[0] + boxes_xyxy[2]) / 2.
    boxes_xywh[1] = (boxes_xyxy[1] + boxes_xyxy[3]) / 2.
    boxes_xywh[2] = boxes_xyxy[2] - boxes_xyxy[0]
    boxes_xywh[3] = boxes_xyxy[3] - boxes_xyxy[1]
    return boxes_xywh


def clamp(n, min_n, max_n):
    if n < min_n:
        return min_n
    elif n > max_n:
        return max_n
    else:
        return n


def create_in_memory_image(image: ndarray) -> BytesIO:
    file = BytesIO()
    image_f = Image.frombuffer('RGB', (image.shape[1], image.shape[0]), image, 'raw')
    image_f.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
    file.name = 'test.bmp'
    file.seek(0)
    return file


def create_in_memory_image_from_pil_image(image: Image) -> BytesIO:
    file = BytesIO()
    image.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
    file.name = 'test.bmp'
    file.seek(0)
    return file


def get_colors(classes):
    num_classes = len(classes)
    # see: https://github.com/bubbliiiing/yolov7-pytorch/blob/master/yolo.py#L92
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors
