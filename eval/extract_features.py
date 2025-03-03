import os
from time import perf_counter
import numpy as np
from tqdm import tqdm

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from data.GlomerulusDataset import GlomerulusDataset 
from utils.download_models import MODELS, download_models

def extract_features_to_dir(dataset: Dataset, batch_size: int, output_path: str):
    #download_models()
    os.makedirs(os.path.join(output_path), exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    start_time = perf_counter()
    print(f"[!] Starting evaluation. Outputting to `{output_path}`.")
    for model in MODELS:
        model = model()
        print(f"    > Evaluating {model.name}.")
        os.makedirs(os.path.join(output_path, model.name), exist_ok=True)
        extracted_feats = np.zeros((len(dataset), model.feat_shape))
        for idx, (x,y) in tqdm(enumerate(dataloader), desc=model.name):
            extracted_feats[idx*batch_size:idx*batch_size+batch_size,:] = model(x)
            print(extracted_feats.shape)
            if idx == 5:
                break

    end_time = perf_counter()
    dt = end_time - start_time
    print(f"[!] Extraction finished in {dt:.2f} seconds.")

    return

if __name__ == '__main__':
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize(224),
    ])
    dataset = GlomerulusDataset(root_dir="/datasets/terumo-data-jpeg/", transforms=transforms) 
    print(dataset)

    extract_features_to_dir(dataset=dataset, batch_size=64, output_path='./features/')
