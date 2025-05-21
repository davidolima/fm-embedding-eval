from dataclasses import dataclass
from typing import *

@dataclass
class Config:
    SUPPORTED_IMAGE_TYPES: Tuple[str] = ('png', 'jpg', 'jpeg')
    CLASSES: Tuple[str] = ("Crescent", "Hypercelularidade", "Membranous", "Normal", "Podocitopatia", "Sclerosis")

    # Transformations
    P_HORIZONTAL_FLIP = 0.5
    P_VERTICAL_FLIP = 0.5
    MAX_ROTATION_ANGLE = 30
    P_ROTATION = 0.4

    BRIGHTNESS_FACTOR = 0.4
    CONTRAST_FACTOR = 0.4
    SATURATION_FACTOR = 0.4
    HUE_FACTOR = 0.1
    P_COLOR_JITTER = 0.4

    GAUSS_NOISE_VAR_RANGE = [5.0, 30.0]
    GAUSS_NOISE_MEAN = 0.0
    P_GAUSS_NOISE = 0.4
    GAUSS_BLUR_LIMIT = [3, 7] 
    P_GAUSS_BLUR = 0.4

    MAX_HOLES = 2
    MAX_H = 56
    MAX_W = 56
    MIN_HOLES = 1
    MIN_H = 14
    MIN_W = 14
    P_COARSE_DROPOUT = 0.2

    P_OPTICAL_DISTORTION = 0.4
    P_GRID_DISTORTION = 0.1
    P_PIECEWISE_AFFINE = 0.4
    P_DISTORTION = 0.3

    SHIFT_LIMIT = 0.0625
    SCALE_LIMIT = [-0.2, 0.2]
    ROTATE_LIMIT = [-30, 30]
    P_SHIFT = 0.5
