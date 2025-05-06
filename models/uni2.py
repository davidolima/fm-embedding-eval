import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from typing import *

timm_kwargs = {
    'img_size': 224, 
    'patch_size': 14, 
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5, 
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0, 
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 
    'act_layer': torch.nn.SiLU, 
    'reg_tokens': 8, 
    'dynamic_img_size': True
}

class UNI2(nn.Module):
    """
    Embedding extractor from the UNI2-h model.
    https://huggingface.co/MahmoodLab/UNI2-h
    """
    def __init__(self, device: Literal['cpu','cuda'] = 'cuda', **kwargs):
        super().__init__()
        self.name = "UNI2"
        self.feat_dim = 1536

        for key,value in kwargs:
            timm_kwargs[key] = value

        # Load model with specified configs
        self.model = UNI2.download_model()
        self.model.to(device)
        self.model.eval()

        # Get model transforms
        self.transforms = create_transform(
            **resolve_data_config(
                self.model.pretrained_cfg,
                model=self.model
            )
        )

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    def get_name(self):
        return self.name

    def get_feat_dim(self):
        return self.get_feat_dim

    @staticmethod
    def download_model(): 
        return timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **timm_kwargs
        )

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            x = self.transforms(x)
            return self.model(x)

if __name__ == '__main__':
    m = UNI2()
    print(m(torch.rand((1,3,224,224))).shape)
