import os
from time import perf_counter
import numpy as np
from tqdm import tqdm
from typing import *

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from models import MODELS
from embedding.utils import save_embedding_to_file
from data.glomerulus import GlomerulusDataset 
from utils.download_models import download_models, authenticate_hf

def single_model_extraction(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    mae_model_size: str, mae_repr_method: str,
    output_path: str,
    verbose: bool = True,
    model_checkpoint: Optional[str] = None,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    os.makedirs(output_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    if model.__name__.lower().startswith('mae'):
        model = model(
            checkpoint_path=model_checkpoint,
            model_size=mae_model_size,
            repr_method=mae_repr_method,
            device=device
        )
    else:
        model = model(device=device)
    extracted_feats = np.zeros((0, model.feat_dim))
    labels = []
    fnames = []

    start_time = perf_counter()
    for idx, (x, y, path) in enumerate(tqdm(dataloader, desc=model.name)):
        x = x.to(device)
        batch_features = model(x).squeeze().detach().cpu().numpy()
        print(batch_features.shape)
        extracted_feats = np.vstack((extracted_feats, batch_features))
        labels.extend(y)
        fnames.extend(path)
    end_time = perf_counter()

    dt = end_time - start_time
    if verbose:
        print(f"[!] Extraction using {model.name} finished in {dt:.2f} seconds.")

    save_embedding_to_file(
        fpath = os.path.join(output_path, model.name),
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
    mae_model_size: str, mae_repr_method: str,
    checkpoints: Optional[str] = None,
    pre_download_models: bool = True,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    print("[*] Evaluating the following models:", [model.__name__ for model in models])

    if pre_download_models:
        download_models(models=models)

    os.makedirs(output_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    start_time = perf_counter()
    print(f"[!] Starting evaluation. Outputting to `{output_path}`.")
    for model in models:
        print(f"    > Evaluating {model.__name__}.")
        model_output_path = os.path.join(output_path, model.__name__)
        os.makedirs(model_output_path, exist_ok=True)
        
        single_model_extraction(
            model = model,
            dataset = dataset,
            batch_size = batch_size,
            output_path = model_output_path,
            verbose = True,
            device = device,
            model_checkpoint=None if model.__name__ != 'MAE' else checkpoints, #FIXME
            mae_model_size=mae_model_size,
            mae_repr_method=mae_repr_method,
        )

    end_time = perf_counter()
    dt = end_time - start_time
    print(f"[!] Extraction finished in {dt:.2f} seconds.")

    return

if __name__ == '__main__':
    import argparse
    def read_cli_args():
        parser = argparse.ArgumentParser(prog="Embedding extraction using foundation models")
        parser.add_argument("-i", "--input-dir",  type=str, default="/datasets/terumo-data-jpeg/")
        parser.add_argument("-o", "--output-dir", type=str, default="./extracted-embeddings/")
        parser.add_argument("-b", "--batch-size", type=int, default=64)
        parser.add_argument("-d", "--device",     type=str, default="cuda")
        parser.add_argument("-s", "--skip-model", action="extend", nargs='+', default=[])
        parser.add_argument("--image-size", type=int, default=224)
        parser.add_argument("--mae-checkpoint", type=str, default="./models/checkpoints/MAE/base.pth", help="checkpoint path for MAE.")
        parser.add_argument("--mae-size", type=str, default="base", help="MAE model size: base, large or huge")
        parser.add_argument("--mae-repr-method", type=str, default="mean", help="Strategy for representing MAE patch-wise embeddings as a single embedding vector. Available options: full, mean, mean+cls or cls_tokens_only")
        parser.add_argument("--download-on-demand", action="store_true", help="Downloads models on demand. By default, downloads all models before the extraction begins.")
        return parser.parse_args()

    args = read_cli_args()
    models_to_skip = [x.lower() for x in args.skip_model]
    
    authenticate_hf()

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((args.image_size, args.image_size)),
    ])
    
    dataset = GlomerulusDataset(root_dir=args.input_dir, transforms=transforms) 
    
    multiple_model_extraction(
        models=[model for model in MODELS if model.__name__.lower() not in models_to_skip],
        dataset=dataset,
        batch_size=args.batch_size,
        output_path=args.output_dir,
        pre_download_models=(not args.download_on_demand),
        device=args.device,
        mae_model_size=args.mae_size,
        mae_repr_method=args.mae_repr_method,
        checkpoints=args.mae_checkpoint, #TODO: Support checkpoints for multiple models
    )
