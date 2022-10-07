import cv2
from shapely.geometry import Polygon


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


def rectangle_polygon_to_two_points(polygon: Polygon):
    x, y = polygon.exterior.coords.xy
    return (x[0], y[0]), (x[2], y[2])
