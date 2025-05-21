import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import Config
from train import Trainer
from utils.cross_validation import load_splits_from_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("./logs/efficientnet_b0_training.log", mode='w')  # File output
    ]
)
logger = logging.getLogger(__name__)

def build_model(out_features: int = 1) -> efficientnet_b0:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, out_features),
        nn.Sigmoid()
    )
    return model
    
if __name__ == '__main__':
    N_EPOCHS = 100
    CHECKPOINT_PATH = "./effnetb0-checkpoints/"
    SPLITS_JSON_FPATH = 'assets/folds_indices_Data_{}.json'

    transforms = T.Compose([
        T.ToTensor(),
        EfficientNet_B0_Weights.IMAGENET1K_V1.transforms(),
    ])

    for class_name in Config.CLASSES:
        logger.info('-'*20 + f"{class_name}" + '-'*20)
        for fold_qty in (2,3,4,5):
            logger.info(f"=> Executando para {fold_qty} folds.")
            
            trainer = Trainer(build_model(), nn.BCELoss(), lr=1e-3)

            train, val = load_splits_from_json(SPLITS_JSON_FPATH.format(class_name), fold_qty, one_vs_all=class_name)

            train.transforms = transforms
            val.transforms = transforms

            train_loader = DataLoader(train, batch_size=32, shuffle=True)
            val_loader = DataLoader(val, batch_size=32, shuffle=True)

            trainer.train(
                num_epochs = N_EPOCHS, 
                train_loader = train_loader, 
                val_loader = val_loader,
                early_stopping = 10,
                save_path = os.path.join(CHECKPOINT_PATH, class_name, f"{fold_qty}_folds")
            )
            break
        break