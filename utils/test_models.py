import torch

from huggingface_hub import login

from utils.download_models import MODELS
from models import UNI, UNI2, Phikon, PhikonV2, MAE

test_input = torch.rand((1,3,224,224))

def test_wrapper(model):
    print(f"----- {model.name} -----")
    print("Input shape:", test_input.shape)
    model = model(device='cpu')
    feats = model(test_input)
    print("Output shape:", feats.shape)
       
if __name__ == '__main__':
    for model in MODELS:
        test_wrapper(model)
    print("----- END OF TESTS -----")
