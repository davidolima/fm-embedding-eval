
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from mae_autoencoder.autoencoder import Autoencoder

EMBEDDING_PATH = "./extracted-embeddings/terumo-val/MAE/"
CHECKPOINT_PATH = "./mae_dim_reduction/checkpoints/"

MAE_SIZE_TO_INPUT_DIM = {
    'base': 768,
    'large': 1024,
    'huge': 1280,
}

verbose = False

def next_mae_model(repr_methods_to_skip = [], sizes_to_skip = []) -> (nn.Module, (str, str, int)):
    for file in os.listdir(CHECKPOINT_PATH):
        mae_size, mae_repr_method, _, latent_dim = file.rsplit('.')[0].split('-')

        if mae_size.lower() in sizes_to_skip or mae_repr_method.lower() in repr_methods_to_skip:
            continue

        input_dim = MAE_SIZE_TO_INPUT_DIM[mae_size]
        if mae_repr_method == 'mean+cls':
            input_dim *= 2
        elif mae_repr_method == 'full':
            input_dim *= 195 # FIXME?

        if verbose: print(f"[!] Loading model ({mae_size}, {mae_repr_method}) checkpoint...", end=' ')
        model = Autoencoder(input_dim=input_dim,latent_dim=int(latent_dim))
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, file), weights_only=False)
        model.load_state_dict(checkpoint['state_dict']())
        if verbose: print("done.")

        yield model, (mae_size, mae_repr_method, input_dim)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-smrm', action="extend", nargs='+', help="Skip MAE representation methods", default=[])
    parser.add_argument('-sms', action="extend", nargs='+', help="Skip MAE sizes", default=[])
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose mode")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    verbose = args.verbose

    output_path = os.path.join(EMBEDDING_PATH, "ae_reduced")
    os.makedirs(output_path, exist_ok=True)
    if verbose: print(f"[!] Output path: {output_path}")


    for ae, (mae_size, mae_repr_method, model_feat_dim) in next_mae_model(repr_methods_to_skip=args.smrm, sizes_to_skip=args.sms):
        model_name = f"MAE-{mae_size}-{mae_repr_method}"
        associated_embeddings_file = os.path.join(EMBEDDING_PATH, model_name.lower(), model_name + ".npy")
        output_fpath = os.path.join(output_path, f"{model_name}-reduced-{ae.latent_dim}.npy")
        embeddings = np.memmap(associated_embeddings_file, shape=(7566, model_feat_dim), dtype='float64')

        dataloader = DataLoader(embeddings, batch_size=64, shuffle=False)
        output_file = np.memmap(output_fpath, mode='w+', shape=(len(embeddings), ae.latent_dim), dtype='float64')
        batch_start = 0
        for x in dataloader:
            batch_end = batch_start + x.size(0)
            with torch.no_grad():
                output_file[batch_start:batch_end] = ae.encoder(torch.tensor(x, dtype=torch.float))
                output_file.flush()
            batch_start = batch_end
        
        np.save(os.path.join(output_path, f"{model_name}-reduced-{ae.latent_dim}_info.npy"), {
            'mae_size': mae_size,
            'mae_repr_method': mae_repr_method,
            'embeddings': output_fpath,
        })
