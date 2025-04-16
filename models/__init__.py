from .uni2 import UNI2
from .uni import UNI
from .phikon import Phikon
from .phikonv2 import PhikonV2
from .mae_wrapper import MAE

HF_MODELS = [
    UNI,
    UNI2,
    Phikon,
    PhikonV2,
]

MODELS = [
    MAE,
    *HF_MODELS
]
