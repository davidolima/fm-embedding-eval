from .uni2 import UNI2
from .uni import UNI
from .phikon import Phikon
from .phikonv2 import PhikonV2
from .mae_wrapper import MAE, MAE_SIZES, MAE_REPR_METHODS

HF_MODELS = [
    UNI,
    UNI2,
    Phikon,
    PhikonV2,
]

MAE_MODELS = [
    f"mae_{size}_{method}"
    for size in MAE_SIZES
    for method in MAE_REPR_METHODS
]

MODELS = [
    *HF_MODELS,
    *MAE_MODELS,
]