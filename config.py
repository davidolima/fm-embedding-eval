from dataclasses import dataclass
from typing import *

@dataclass
class Config:
    SUPPORTED_IMAGE_TYPES: Tuple[str] = ('png', 'jpg', 'jpeg')
