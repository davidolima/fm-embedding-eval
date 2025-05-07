import torch
import torch.nn as nn
from torchvision import transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import PIL
from typing import *

from models.mae.models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

MAE_SIZES = ('base', 'large', 'huge')
MAE_REPR_METHODS = ('full', 'cls_token_only', 'mean', 'mean+cls')

class MAE(nn.Module):
    """
    Embedding extractor for the Masked Autoencoder model.
    """
    def __init__(
        self,
        model_size: Literal[*MAE_SIZES] = 'base', 
        repr_method: Literal[*MAE_REPR_METHODS] = 'mean',
        device: Literal['cpu', 'cuda'] = 'cuda',
        **kwargs
    ):
        super().__init__()
        self.name = f"MAE-{model_size}-{repr_method}"
        self.model_size = model_size
        self.repr_method = repr_method
        
        # Load model with specified configs
        model_size = model_size.lower()
        if model_size == 'base':
            self.model = mae_vit_base_patch16()
            self.feat_dim = 768
        elif model_size == 'large':
            self.model = mae_vit_large_patch16()
            self.feat_dim = 1024
        elif model_size == 'huge':
            self.model = mae_vit_huge_patch14()
            self.feat_dim = 1280
        else:
            raise ValueError(f"Specified MAE model size does not exist: `{model_size}`. Available options are `base`, `large` and `huge`.")

        self.model.to(device)
        self.model.eval()

        if repr_method not in ('full', 'cls_token_only', 'mean', 'mean+cls'):
            raise ValueError(f"MAE Embedding representation method not recognized: `{repr_method}`")
        self.repr_method = repr_method

        if self.repr_method == 'mean+cls':
            self.feat_dim *= 2
        elif self.repr_method == 'full':
            self.feat_dim *= self.model.patch_embed.num_patches-1

        # MAE Transforms based on eval transforms found in
        # https://github.com/facebookresearch/mae/blob/main/util/datasets.py#L51

        self.transforms = T.Compose([
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    @staticmethod
    def download_model():
        return

    def get_name(self):
        return self.name 

    def get_feat_dim(self):
        return self.feat_dim

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    def load_checkpoint(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = f"./models/checkpoints/mae-{self.model_size}/checkpoint-90.pth"
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model'], strict=False)

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            x = self.transforms(x)
            x, _, ids_restore =  self.model.forward_encoder(x,mask_ratio=0)
            x = MAE.get_ordered_mae_embeddings(x, ids_restore)
            x = self.get_embed_repr(x)
            return x
    
    def get_embed_repr(self, latent):
        if self.repr_method == 'full':
            return MAE.embed_repr_full(latent)
        elif self.repr_method == 'cls_token_only':
            return MAE.embed_repr_cls_tkn_only(latent)
        elif self.repr_method == 'mean':
            return MAE.embed_repr_avg_per_patch(latent)
        elif self.repr_method == 'mean+cls':
            return MAE.embed_repr_mean_and_cls_tkn(latent)
        else:
            raise RuntimeError("Unreachable.")

    @staticmethod
    def embed_repr_cls_tkn_only(latent):
        return latent[:, 0, :]

    @staticmethod
    def embed_repr_full(latent):
        return latent[:,1:,:]

    @staticmethod
    def embed_repr_avg_per_patch(latent):
        return torch.mean(MAE.embed_repr_full(latent), dim=1)

    @staticmethod
    def embed_repr_mean_and_cls_tkn(latent):
        cls_embedding = MAE.embed_repr_cls_tkn_only(latent)
        patch_mean = MAE.embed_repr_avg_per_patch(latent)
        full_embedding = torch.cat([cls_embedding,patch_mean], dim=1)
        return full_embedding
    
    @staticmethod
    def get_ordered_mae_embeddings(latent, ids_restore):
        """
        Uses id_restore to get original embedding positions
        """
        B, N, D = latent.shape

        unshuffled = torch.gather(
            latent,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1,1,D)
        )
        return unshuffled

if __name__ == '__main__':
    x = torch.rand((1,3,224,224)).cuda()
    m = MAE(model_size='base')
    print(m(x))
    m = MAE(model_size='large')
    print(m(x))
    m = MAE(model_size='huge')
    print(m(x))
