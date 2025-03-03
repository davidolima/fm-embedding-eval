import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel

class PhikonV2(nn.Module):
    """
    Embedding extractor from the Phikon-V2 model.
    https://huggingface.co/owkin/phikon-v2
    """
    name = "PhikonV2"
    feat_shape = 1024
    def __init__(self):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = PhikonV2.download_model()
        self.model.eval()

    @staticmethod
    def download_model():
        return AutoModel.from_pretrained("owkin/phikon-v2")

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            outputs = self.model(**self.processor(x, return_tensors="pt"))
            features = outputs.last_hidden_state[:, 0, :]
        return features

if __name__ == '__main__':
    m = PhikonV2()
    print(m(torch.rand((1,3,224,224))).shape)
