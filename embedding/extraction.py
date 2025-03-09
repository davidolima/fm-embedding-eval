import os
from time import perf_counter
import numpy as np
from tqdm import tqdm
from typing import *

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from embedding.utils import save_embedding_to_file
from data.GlomerulusDataset import GlomerulusDataset 
from utils.download_models import MODELS, download_models, authenticate_hf


def single_model_extraction(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    output_path: str,
    verbose: bool = True,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    os.makedirs(output_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = model(device=device)
    extracted_feats = np.zeros((0, model.feat_dim))
    labels = []
    fnames = []

    start_time = perf_counter()
    for idx, (x, y, path) in enumerate(tqdm(dataloader, desc=model.name)):
        x = x.to(device)
        batch_features = model(x).squeeze().detach().cpu().numpy()
        extracted_feats = np.vstack((extracted_feats, batch_features))
        labels.extend(y)
        fnames.extend(path)
    end_time = perf_counter()
    dt = end_time - start_time
    if verbose:
        print(f"[!] Extraction using {model.name} finished in {dt:.2f} seconds.")

    save_embedding_to_file(
        fpath = output_path,
        model = model.name,
        fnames = fnames,
        labels = labels,
        embeddings = extracted_feats,
        classes = dataset.classes,
    )

def multiple_model_extraction(
    models: list[torch.nn.Module],
    dataset: Dataset, 
    batch_size: int,
    output_path: str,
    pre_download_models: bool = True,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    if pre_download_models:
        download_models()

    os.makedirs(output_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    start_time = perf_counter()
    print(f"[!] Starting evaluation. Outputting to `{output_path}`.")
    for model in models:
        print(f"    > Evaluating {model.name}.")
        model_output_path = os.path.join(output_path, model.name)
        os.makedirs(model_output_path, exist_ok=True)
        
        single_model_extraction(
            model = model,
            dataset = dataset,
            output_path = model_output_path,
            verbose = True,
            device = device,
        )

    end_time = perf_counter()
    dt = end_time - start_time
    print(f"[!] Extraction finished in {dt:.2f} seconds.")

    return

if __name__ == '__main__':
    authenticate_hf()

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize(224),
    ])
    
    dataset = GlomerulusDataset(root_dir="/datasets/terumo-data-jpeg/", transforms=transforms) 
    
    multiple_model_extraction(
        models=MODELS,
        dataset=dataset,
        batch_size=64,
        output_path="./features/",
        pre_download_models=True,
        device='cuda',
    )
