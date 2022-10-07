import colorsys
from io import BytesIO

import cv2
from PIL import Image
from numpy import ndarray
from shapely.geometry import Polygon


def clamp(n, min_n, max_n):
    if n < min_n:
        return min_n
    elif n > max_n:
        return max_n
    else:
        return n


def create_in_memory_image(image: ndarray):
    file = BytesIO()
    image_f = Image.frombuffer('RGB', (image.shape[1], image.shape[0]), image, 'raw')
    image_f.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
    file.name = 'test.bmp'
    file.seek(0)
    return file


def create_in_memory_image_from_pil_image(image: Image):
    file = BytesIO()
    image.save(file, 'bmp')  # png format seems too time-consuming, use bmp instead
    file.name = 'test.bmp'
    file.seek(0)
    return file


def get_iou(polygon1: Polygon, polygon2: Polygon):
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou


def in_target_proportion(polygon1: Polygon, polygon2: Polygon):
    p1_area = polygon1.area
    if p1_area == 0:
        return 0
    intersect = polygon1.intersection(polygon2).area
    return intersect / p1_area


def draw_polygon_outline(image, polygon: Polygon, thickness, color):
    x, y = polygon.exterior.coords.xy
    for i in range(len(x) - 1):
        p1, p2 = (int(x[i]), int(y[i])), (int(x[i + 1]), int(y[i + 1]))
        cv2.line(image, p1, p2, color, thickness)


def get_colors(classes):
    num_classes = len(classes)
    # see: https://github.com/bubbliiiing/yolov7-pytorch/blob/master/yolo.py#L92
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def rectangle_polygon_to_two_points(polygon: Polygon):
    x, y = polygon.exterior.coords.xy
    return (x[0], y[0]), (x[2], y[2])
