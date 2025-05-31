import os
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import Config
from train import Trainer
from utils.cross_validation import load_splits_from_json

SAVE_NAME = "efficientnet_b0_training_no-aug-val"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(f"./results/{SAVE_NAME}/training.log", mode='w')  # File output
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
    CHECKPOINT_PATH = f"./results/{SAVE_NAME}/"
    SPLITS_JSON_FPATH = 'data/cross-validation-folds/folds_indices_Data_{}.json'
    LR = 1e-3

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    transforms = T.Compose([
        T.ToTensor(),
        EfficientNet_B0_Weights.IMAGENET1K_V1.transforms(),
    ])

    metrics = ('loss', 'accuracy', 'precision', 'recall', 'f1_score')


    for class_name in Config.CLASSES:
        logger.info('-'*20 + f"{class_name}" + '-'*20)
        cls_history = {}
        os.makedirs(os.path.join(CHECKPOINT_PATH, class_name), exist_ok=True)

        for fold_qty in (2,3,4,5):
            logger.info(f"=> Executando para {fold_qty} folds.")

            fold_metrics = {x: list() for x in metrics}
            for val_fold_idx in range(fold_qty):
                logger.info(f"==> Validando no fold {val_fold_idx+1}")
                
                trainer = Trainer(build_model(), nn.BCELoss(), lr=LR)

                train, val = load_splits_from_json(
                    json_fpath=SPLITS_JSON_FPATH.format(class_name),
                    fold_no=fold_qty,
                    val_split_idx=val_fold_idx,
                    one_vs_all=class_name
                )

                train.transforms = transforms
                val.transforms = transforms

                train_ds_info = train.info(dont_print=True)
                val_ds_info = val.info(dont_print=True)

                [logger.info(line) for line in (" --- Training ---\n" + train_ds_info + "--- Validation ---\n" + val_ds_info).split('\n')]

                train_loader = DataLoader(train, batch_size=64, shuffle=True)
                val_loader = DataLoader(val, batch_size=64, shuffle=True)

                fold_results = trainer.train(
                    num_epochs = N_EPOCHS, 
                    train_loader = train_loader, 
                    val_loader = val_loader,
                    early_stopping = 10,
                    save_path = os.path.join(CHECKPOINT_PATH, class_name, f"{fold_qty}_folds", str(val_fold_idx + 1))
                )

                for metric in metrics:
                    fold_metrics[metric].append(fold_results[metric].compute())

            cls_history[f'{fold_qty}_folds'] = fold_metrics

        with open(os.path.join(CHECKPOINT_PATH, class_name, f"training_log.txt"), 'w+') as f:
            json.dump(cls_history, f, indent=4)

