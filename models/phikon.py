import torch
import torch.nn as nn

from typing import *

from transformers import AutoImageProcessor, AutoModel

class Phikon(nn.Module):
    """
    Embedding extractor from the Phikon model.
    https://huggingface.co/owkin/phikon
    """
    def __init__(self, device: Literal['cpu', 'cuda'] = 'cuda'):
        super().__init__()
        self.name = "Phikon"
        self.feat_dim = 768
        
        self.device = device

        self.model = Phikon.download_model()
        self.model.to(self.device)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon")

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
        return AutoModel.from_pretrained("owkin/phikon")

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            x = self.processor(x.to(torch.uint8), return_tensors='pt').to(self.device)
            outputs = self.model(**x)
            features = outputs.last_hidden_state[:, 0, :]
        return features

if __name__ == '__main__':
    m = Phikon(device='cuda')
    rand_sample = torch.rand((1,3,224,224)).to('cuda')
    output = m(rand_sample)
    print(output.shape)
