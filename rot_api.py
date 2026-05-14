import math
import numpy as np
import cv2


def rotate_image(image, angle):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def largest_rotated_rect(w, h, angle):
    """Largest axis-aligned rectangle that fits inside a rotated rectangle."""
    angle = abs(angle) % (math.pi / 2)
    if w < h:
        w, h = h, w
    if angle < 1e-10:
        return w, h
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    if 2 * sin_a * cos_a * w > (cos_a ** 2 - sin_a ** 2) * h:
        x = 0.5 * h / cos_a
        return 2 * x * cos_a, 2 * x * sin_a
    cos_2a = cos_a ** 2 - sin_a ** 2
    wr = (w * cos_a - h * sin_a) / cos_2a
    hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def crop_around_center(image, width, height):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - int(width / 2))
    y1 = max(0, cy - int(height / 2))
    x2 = min(w, x1 + int(width))
    y2 = min(h, y1 + int(height))
    return image[y1:y2, x1:x2]
