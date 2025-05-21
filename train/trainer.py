from typing import *

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

import torchvision.transforms as T

from tqdm import tqdm
import logging

import train.metrics as metrics
from data.glomerulus import GlomerulusDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("./logs/training.log", mode='w')  # File output
    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        lr: float = 1e-5,
        optimizer: Optional[Optimizer] = AdamW,
        device: Optional[str] = 'cuda'
    ):  
        self.model = model
        self.criterion = criterion
        self.device = device

        self.optimizer = optimizer(model.parameters(), lr=lr)



    def train(
        self,
        num_epochs: int, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        early_stopping: Optional[int] = None,
        save_path: Optional[str] = './checkpoints/'
    ):
        self.model.to(self.device)

        logger.info("Training starting. Current configuration:")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Learning rate: {self.optimizer.defaults['lr']}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Saving checkpoints to {save_path}.")
        if early_stopping:
            logger.info(f"Early stopping set to {early_stopping} epochs.")

        train_metrics = {
            'loss': metrics.Loss(),
            'accuracy': metrics.Accuracy(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall(),
            'f1_score': metrics.F1Score(),
        }
        
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.model.train()
            
            for m in train_metrics.values():
                m.reset()

            for inputs, labels, _ in tqdm(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels.float())

                loss.backward()
                self.optimizer.step()
                train_metrics['loss'].update(loss.item())

                preds = outputs > 0.5
                train_metrics['accuracy'].update(preds, labels)
                train_metrics['precision'].update(preds, labels)
                train_metrics['recall'].update(preds, labels)
                train_metrics['f1_score'].update(preds, labels)

            running_loss /= len(train_loader)
            logger.info(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss}" + "".join([f"| {metric}" for metric in metrics.values()]))

            val_metrics = self.validate(val_loader)
            if val_metrics['loss'] < best_loss:
                save_fpath = os.path.join(save_path, f"best_val_loss.pth")
                self.save_model(save_fpath)
                logger.info(f"Loss decreased {best_loss} -> {val_metrics['loss']}. Model saved to `{save_fpath}`.")
                best_loss = val_metrics['loss']
            else:
                early_stopping_counter += 1
                if early_stopping and early_stopping_counter >= early_stopping:
                    logger.info(f"Early stopping triggered after {early_stopping} epochs without improvement.")
                    break

            logger.info(f"[Validation {epoch+1}/{num_epochs}] Loss: {val_metrics['loss']:.4f}" + "".join([f"| {metric}" for metric in val_metrics.values()]))
        
        logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
        self._print_metrics(val_metrics)
        if save_path:
            save_fpath = os.path.join(save_path, f"last_epoch_{epoch+1}.pth")
            self.save_model(save_fpath)
            logger.info(f"Last model saved to `{save_fpath}`.")

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_metrics = {
            'loss': metrics.Loss(),
            'accuracy': metrics.Accuracy(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall(),
            'f1_score': metrics.F1Score(),
        }

        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels.float())
                val_metrics['loss'].update(loss.item())

                preds = torch.argmax(outputs, 1) if outputs.ndim > 1 else (outputs > 0.5).long()
                val_metrics['accuracy'].update(preds, labels)
                val_metrics['precision'].update(preds, labels)
                val_metrics['recall'].update(preds, labels)
                val_metrics['f1_score'].update(preds, labels)

        return {k: v.compute() for k, v in val_metrics.items()}
    
    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        for _, metric in metrics.items():
            logger.info(metric)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


def cross_validation(
    model: nn.Module,
    criterion: Callable,
    data_path: str,
    splits_fpath: str,
    num_folds: int,
    classes: list[str] | Literal['terumo'],
    one_vs_all: Optional[str] = None,
    num_epochs: int = 10,
    early_stopping: Optional[int] = None,
    save_path: Optional[str] = './checkpoints/'
) -> None:
    if classes == 'terumo':
        classes = ["Crescent", "Hypercelularidade", "Membranous", "Normal", "Podocitopatia", "Sclerosis"]
    
    train = GlomerulusDataset(
        data_path,
        classes=classes,
        one_vs_all=one_vs_all,
        consider_augmented=True if one_vs_all is None else 'positive_only'
    )
    val = GlomerulusDataset(
        data_path,
        classes=classes,
        one_vs_all=one_vs_all,
        consider_augmented=True if one_vs_all is None else 'positive_only'
    )

    for fold in range(num_folds):
        val_split = fold+1

        train_splits = list(range(1, num_folds + 1))
        train_splits.remove(val_split)

        val.load_splits_from_json(val_split, splits_fpath, clear_data=True)
        train.load_splits_from_json(train_splits, splits_fpath, clear_data=True)

        logger.info(f"Fold {val_split}/{num_folds}")
        trainer = Trainer(model, criterion)
        fold_save_path = os.path.join(save_path, f"fold_{fold}")
        
        trainer.train(num_epochs, train_loader, val_loader, early_stopping, fold_save_path)