#!/bin/env python3

import os

import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms as T
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from models.mae_wrapper import MAE
from mae_autoencoder.autoencoder import Autoencoder

root_dir = "./extracted-embeddings/terumo-data-jpeg/MAE/"
checkpoint_path = "./mae_dim_reduction/checkpoints/"
batch_size = 64
epochs = 100
latent_dim = 1024
learning_rate = 1e-5

mae_size = 'base'
mae_repr_method = 'mean'

device = "cuda" if torch.cuda.is_available() else "cpu"
verbose = False 

def save_checkpoint(
    filename: str,
    model: torch.nn.modules.module.Module,
    optimizer: torch.optim.Optimizer,
    current_epoch: int # In order to know how many epochs the model has been trained for
    ) -> None:

    if not filename.endswith(".pt"):
        filename += ".pt"

    if verbose: print(f"Saving checkpoing to file '{filename}'...", end='')
    checkpoint = {
        "state_dict": model.state_dict,
        "optmizer": optimizer.state_dict,
        "epoch": current_epoch
    }
    torch.save(checkpoint, filename)
    if verbose: print("Saved.")

def train(epochs, checkpoint_path, device='cuda'):
    # Get MAE details
    embedding_model = MAE(checkpoint_path=f"./models/checkpoints/mae-{mae_size}/checkpoint-90.pth", model_size=mae_size, repr_method=mae_repr_method)
    input_dim = embedding_model.feat_dim
    del embedding_model

    # Load model, optimizer and criterion
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = AdamW(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    # Load MAE embeddings
    #data = np.load(os.path.join(root_dir, f"MAE_{mae_size}_{mae_repr_method}.npy"), mmap_mode='r', allow_pickle=True)
    data = np.memmap(os.path.join(root_dir, f"MAE-{mae_size}-{mae_repr_method}.npy"), shape=(7566, input_dim), mode='c', dtype='float64')
    dataloader = DataLoader(data, batch_size, shuffle=True)

    # Define metrics
    metrics = [
        "total",1
        "loss",
    ]

    running_metrics = dict.fromkeys(metrics, 0)
    best_metrics = dict.fromkeys(metrics, np.inf)

    if verbose: print("[!] Running on", device)
    for epoch in tqdm(range(epochs)):
        for x in dataloader:
            x = torch.tensor(x, device=device).float()

            x_hat = model(x)
            loss = criterion(x_hat, x)

            running_metrics["loss"] += loss.item()*x.size(0)
            running_metrics["total"] += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose: print(f"[Epoch {epoch}/{epochs}] Loss {running_metrics['loss']/running_metrics['total']:.2f}")

        if loss < best_metrics["loss"]:
            if verbose: print(f"[!] New Best Loss: {best_metrics['loss']} -> {loss}. ", end='')
            best_metrics["loss"] = loss
            save_checkpoint(
                filename=os.path.join(checkpoint_path, f"{mae_size}-{mae_repr_method}-best_loss-{latent_dim}"),
                model=model,
                optimizer=optimizer,
                current_epoch=epoch
            )



def run_for_all_mae(epochs, checkpoint_path):
    global mae_size, mae_repr_method
    for size in tqdm(('base', 'large')):
        for repr_method in tqdm(['full']):#,'cls_token_only', 'mean', 'mean+cls')):
            mae_size = size
            mae_repr_method = repr_method
            train(epochs, checkpoint_path)

if __name__ == "__main__":
    run_for_all_mae(epochs, checkpoint_path)
