import cv2
import numpy as np


def detect_specular_highlights_bgr(image: np.ndarray) -> np.ndarray:
    """Simple HSV heuristic baseline for bright low-saturation highlights."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    del h
    mask = ((v >= 230) & (s <= 40)).astype(np.uint8) * 255
    return mask

