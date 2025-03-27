import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel

from typing import *

class PhikonV2(nn.Module):
    """
    Embedding extractor from the Phikon-V2 model.
    https://huggingface.co/owkin/phikon-v2
    """
    name = "PhikonV2"
    feat_dim = 1024
    def __init__(self, device:Literal['cuda','cpu'] = 'cuda'):
        super().__init__()

        self.device = device

        self.model = PhikonV2.download_model()
        self.model.to(self.device)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")

    @staticmethod
    def download_model():
        return AutoModel.from_pretrained("owkin/phikon-v2")

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            x = self.processor(x.to(torch.uint8), return_tensors="pt").to(self.device)
            outputs = self.model(**x)
            features = outputs.last_hidden_state[:, 0, :]
        return features

if __name__ == '__main__':
    m = PhikonV2()
    print(m(torch.rand((1,3,224,224))).shape)
