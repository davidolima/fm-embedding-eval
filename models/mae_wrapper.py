import torch
import torch.nn as nn
from torchvision import transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import PIL
from typing import *

from models.mae.models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

class MAE(nn.Module):
    """
    Embedding extractor for the Masked Autoencoder model.
    """
    def __init__(self, model_size: Literal['base','large','huge'] = 'base', device: Literal['cpu', 'cuda'] = 'cuda', **kwargs):
        super().__init__()
        self.name = "MAE"

        # Load model with specified configs
        model_size = model_size.lower()
        if model_size == 'base':
            self.model = mae_vit_base_patch16()
            self.name += "_base"
            self.feat_dim = 768
        elif model_size == 'large':
            self.model = mae_vit_large_patch16()
            self.name += "_large"
            self.feat_dim = 1024
        elif model_size == 'huge':
            self.model = mae_vit_huge_patch14()
            self.name += "_huge"
            self.feat_dim = 1280
        else:
            raise ValueError(f"Specified MAE model size does not exist: `{model_size}`. Available options are `base`, `large` and `huge`.")

        self.model.to(device)
        self.model.eval()

        # MAE Transforms based on eval transforms found in
        # https://github.com/facebookresearch/mae/blob/main/util/datasets.py#L51

        self.transforms = T.Compose([
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    @staticmethod
    def download_model():
        return

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            x = self.transforms(x)
            x, _, ids_restore =  self.model.forward_encoder(x,mask_ratio=0)
            #x = MAE.get_ordered_mae_embeddings(x, ids_restore)
            return x

    @staticmethod
    def get_ordered_mae_embeddings(latent, ids_restore):
        """
        Uses id_restore to get original embedding positions
        """
        B, N, D = latent.shape
        unshuffled = torch.zeros_like(latent)

        for b in range(B):
            for i in range(N):
                idx = ids_restore[b, i].item()
                if idx < N:
                    unshuffled[b, idx] = latent[b, i]

        return unshuffled

if __name__ == '__main__':
    x = torch.rand((1,3,224,224)).cuda()
    m = MAE(model_size='base')
    print(m(x))
    m = MAE(model_size='large')
    print(m(x))
    m = MAE(model_size='huge')
    print(m(x))
