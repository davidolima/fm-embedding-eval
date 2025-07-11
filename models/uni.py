import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from typing import *

class UNI(nn.Module):
    """
    Embedding extractor from the UNI model.
    https://huggingface.co/MahmoodLab/UNI
    """
    def __init__(self, device: Literal['cpu', 'cuda'] = 'cuda', **kwargs):
        super().__init__()
        self.name = "UNI"
        self.feat_dim = 1024

        # Load model with specified configs
        self.model = UNI.download_model()
        self.model.to(device)
        self.model.eval()

        # Get model transforms
        self.transforms = create_transform(
            **resolve_data_config(
                self.model.pretrained_cfg,
                model=self.model
            )
        )
    
    def get_name(self):
        return self.name

    def get_feat_dim(self):
        return self.feat_dim

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    @staticmethod
    def download_model():
        return timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        )

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            x = self.transforms(x)
            return self.model(x)

if __name__ == '__main__':
    m = UNI()
    print(m(torch.rand((1,3,224,224))).shape)
