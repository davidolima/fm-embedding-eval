import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel

class Phikon(nn.Module):
    """
    Embedding extractor from the Phikon model.
    https://huggingface.co/owkin/phikon
    """
    name = "Phikon"
    feat_shape = 768
    def __init__(self):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        self.model = Phikon.download_model()
        self.model.eval()

    @staticmethod
    def download_model():
        return AutoModel.from_pretrained("owkin/phikon")

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            outputs = self.model(**self.processor(x, return_tensors="pt"))
            features = outputs.last_hidden_state[:, 0, :]
        return features

if __name__ == '__main__':
    m = Phikon()
    print(m(torch.rand((1,3,224,224))).shape)
