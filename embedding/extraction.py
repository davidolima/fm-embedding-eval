import os
import logging
from typing import *
from time import perf_counter

import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from models import MODELS, MAE, MAE_SIZES, MAE_REPR_METHODS
from embedding.utils import save_embedding_to_file
from data.glomerulus import GlomerulusDataset 
from utils.download_models import download_models, authenticate_hf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def single_model_extraction(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    mae_model_size: Literal[*MAE_SIZES],
    mae_repr_method: Literal[*MAE_REPR_METHODS],
    output_path: str,
    device: Literal['cpu', 'cuda'] = 'cuda',
) -> None:
    """
    Extract embeddings for a single model.

    Args:
        model (torch.nn.Module): The model to use for extraction.
        dataset (Dataset): The dataset to process.
        batch_size (int): Batch size for the DataLoader.
        mae_model_size (str): Size of the MAE model ("base", "large" or "huge").
        mae_repr_method (str): Representation method for MAE embeddings.
        output_path (str): Directory to save the embeddings.
        device (str): Device to use ("cpu" or "cuda").
    """
    os.makedirs(output_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_size, model_repr_method = None, None
    if isinstance(model, str):
        if model.lower().startswith('mae'):
            if mae_model_size != 'all' and mae_model_size not in model:
                logger.info(f"Skipping model `{model}`.")
                return
            if mae_repr_method != 'all' and mae_repr_method not in model:
                logger.info(f"Skipping model `{model}`.")
                return

            _, model_size, model_repr_method, = model.split('-')
            model = MAE(model_size=model_size, repr_method=model_repr_method, device=device)
            model.load_checkpoint()
        else:
            raise ValueError(f"Model `{model}` not recognized. Available options are: {', '.join([str(x) for x in MODELS])}.")
    else:
        model = model(device=device)

    embedding_fpath = os.path.join(output_path, model.name+'.npy')
    extracted_feats = np.memmap(embedding_fpath, dtype='float64', mode='w+', shape=(len(dataset), model.feat_dim))
    labels = []
    fnames = []

    batch_start = 0
    start_time = perf_counter()
    for idx, (x, y, path) in enumerate(tqdm(dataloader, desc=model.name)):
        x = x.to(device)
        batch_end = batch_start + batch_size

        batch_features = model(x)
        if model_repr_method == 'full':
            batch_features = batch_features.flatten(1)
        else:
            batch_features = batch_features.squeeze()

        batch_features = batch_features.detach().cpu().numpy()
        extracted_feats[batch_start:batch_end] = batch_features
        extracted_feats.flush()

        labels.extend(y)
        fnames.extend(path)

        batch_start = batch_end

    end_time = perf_counter()
    dt = end_time - start_time
    logger.info(f"Extraction using {model.name} finished in {dt:.2f} seconds.")

    extracted_feats.flush()
    save_embedding_to_file(
        fpath = os.path.join(output_path, model.name + "_info.npy"),
        model = model.name,
        fnames = fnames,
        labels = labels,
        embeddings = os.path.join(output_path, model.name+'.npy'),
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
) -> None:
    """
    Extract embeddings for multiple models.

    Args:
        models (list[torch.nn.Module]): List of models to use for extraction.
        dataset (Dataset): The dataset to process.
        batch_size (int): Batch size for the DataLoader.
        output_path (str): Directory to save the embeddings.
        mae_model_size (str): Size of the MAE model ("base", "large" or "huge").
        mae_repr_method (str): Representation method for MAE embeddings.
        checkpoints (str, optional): Path to the checkpoint file for MAE models. Defaults to None.
        pre_download_models (bool): Whether to download models before extraction. Defaults to True.
        device (str): Device to use ("cpu" or "cuda"). Defaults to "cuda".
    """

    logger.info("Evaluating the following models:", models)

    if pre_download_models:
        download_models()

    os.makedirs(output_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    start_time = perf_counter()
    logger.info(f"Starting evaluation. Outputting to `{output_path}`.")
    for model in models:
        model_name = model if isinstance(model, str) else model.__name__
        logger.info(f"Evaluating {model_name}.")
        model_output_path = os.path.join(output_path, model_name)
        os.makedirs(model_output_path, exist_ok=True)
        
        single_model_extraction(
            model = model,
            dataset = dataset,
            batch_size = batch_size,
            output_path = model_output_path,
            device = device,
            mae_model_size=mae_model_size,
            mae_repr_method=mae_repr_method,
        )

    end_time = perf_counter()
    dt = end_time - start_time
    logger.info(f"Extraction finished in {dt:.2f} seconds.")

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
        parser.add_argument("--classes",    action="extend", nargs='+', default=[])
        parser.add_argument("--image-size", type=int, default=224)
        parser.add_argument("--mae-checkpoint", type=str, default="./models/checkpoints/MAE/base.pth", help="checkpoint path for MAE.")
        parser.add_argument("--mae-size", type=str, default="all", help="MAE model size: all, base, large or huge")
        parser.add_argument("--mae-repr-method", type=str, default="all", help="Strategy for representing MAE patch-wise embeddings as a single embedding vector. Available options: all, full, mean, mean+cls or cls_tokens_only")
        parser.add_argument("--download-on-demand", action="store_true", help="Downloads models on demand. By default, downloads all models before the extraction begins.")
        return parser.parse_args()

    args = read_cli_args()
    models_to_skip = [x.lower() for x in args.skip_model]
    
    authenticate_hf()

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((args.image_size, args.image_size)),
        T.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),  # Handle RGBA by keeping only RGB channels
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
    ])
    
    if 'terumo' in args.classes:
        args.classes = ['Crescent', 'Sclerosis', 'Normal', 'Podocitopatia', 'Hypercelularidade', 'Membranous']

    dataset = GlomerulusDataset(root_dir=args.input_dir, classes=args.classes, transforms=transforms) 
    
    models_to_eval = []
    for model in MODELS:
        model_name = model.lower() if isinstance(model, str) else model.__name__.lower()

        if any([x in model_name for x in models_to_skip]):
            continue

        models_to_eval.append(model)

    multiple_model_extraction(
        models=models_to_eval,
        dataset=dataset,
        batch_size=args.batch_size,
        output_path=args.output_dir,
        pre_download_models=(not args.download_on_demand),
        device=args.device,
        mae_model_size=args.mae_size,
        mae_repr_method=args.mae_repr_method,
        checkpoints=args.mae_checkpoint, #TODO: Support checkpoints for multiple models
    )
