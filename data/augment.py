import os
import random
from multiprocessing import Pool
from typing import *

import torch 
from torchvision.utils import save_image

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

from config import Config
import cv2

def get_train_transforms(seed: int = 42) -> A.Compose:
    albumentations_t = A.Compose([
        A.HorizontalFlip(Config.P_HORIZONTAL_FLIP),
        A.VerticalFlip(Config.P_VERTICAL_FLIP),
        A.Rotate(limit=Config.MAX_ROTATION_ANGLE,
                 p=Config.P_ROTATION),
        A.ColorJitter(
            brightness=Config.BRIGHTNESS_FACTOR,
            contrast=Config.CONTRAST_FACTOR,
            saturation=Config.SATURATION_FACTOR,
            hue=Config.HUE_FACTOR,
            p=Config.P_COLOR_JITTER),
        A.GaussNoise(
            var_limit=Config.GAUSS_NOISE_VAR_RANGE,
            mean=Config.GAUSS_NOISE_MEAN,
            p=Config.P_GAUSS_NOISE
        ),
        A.GaussianBlur(
            blur_limit=Config.GAUSS_BLUR_LIMIT,
            p=Config.P_GAUSS_BLUR
        ),
        A.CoarseDropout(
            max_holes=Config.MAX_HOLES,
            max_height=Config.MAX_H,
            max_width=Config.MAX_W,
            min_holes=Config.MIN_HOLES,
            min_height=Config.MIN_H,
            min_width=Config.MIN_W,
            fill_value=0,
            mask_fill_value=0,
            p=Config.P_COARSE_DROPOUT),
        A.OneOf(
            [
                A.OpticalDistortion(p=Config.P_OPTICAL_DISTORTION),
                A.GridDistortion(p=Config.P_GRID_DISTORTION),
                A.PiecewiseAffine(p=Config.P_PIECEWISE_AFFINE),
            ],
            p=Config.P_DISTORTION
        ),
        A.ShiftScaleRotate(
            shift_limit=Config.SHIFT_LIMIT,
            scale_limit=Config.SCALE_LIMIT,
            rotate_limit=Config.ROTATE_LIMIT,
            interpolation=cv2.INTER_LINEAR,
            border_mode=0,
            value=(0, 0, 0),
            p=Config.P_SHIFT
        ),
        A.pytorch.ToTensorV2(),
    ], seed=seed)
    return albumentations_t


def get_test_transforms():
    albumentations_t = A.Compose([
        A.pytorch.ToTensorV2(),
    ])
    return albumentations_t

def apply_to_one_image(image_path: str, transforms: A.Compose, save_path: Optional[str] = None) -> Any:
    """
    Apply transforms to an image and save the result.
    
    Args:
        image_path (str): Path to the input image.
        transforms (A.Compose): Albumentations transformations to apply.
        save_path (str, optional): Path to save the transformed image. If None, the image is not saved.
    Returns:
        str: Path to the saved image if save_path is provided.
        Any: Transformed image if save_path is None.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    augmented_image = transforms(image=image)    
    augmented_image = augmented_image['image']

    if save_path:
        # Prevent duplicates
        if os.path.basename(save_path) in os.listdir(os.path.dirname(save_path)):
            i = 1
            new_name = os.path.basename(save_path).split(".")[0] + f"_augmented_{i}." + os.path.basename(save_path).split(".")[-1]
            while new_name in os.listdir(os.path.dirname(save_path)):
                i += 1
                new_name = os.path.basename(save_path).split(".")[0] + f"_augmented_{i}." + os.path.basename(save_path).split(".")[-1]
            save_path = os.path.join(os.path.dirname(save_path), new_name)

        save_image(augmented_image.float() / 255.0, save_path)  # Normalize to [0, 1]
        
        return save_path

    return augmented_image

def apply_to_images(image_paths: list[str], transforms: A.Compose, save_dir: Optional[str], shuffle: bool = False, limit: int = -1, n_workers: int = 4) -> list[str]:
    """
    Apply transforms to a list of images and save the results.
    """
    os.makedirs(save_dir, exist_ok=True)

    images = [f for f in image_paths if f.split('.')[-1].lower() in Config.SUPPORTED_IMAGE_TYPES]
    if len(images) == 0:
        print(f"No images found: No supported image types.")
        return

    if shuffle:
        np.random.shuffle(images)
    
    if limit > 0:
        images = images[:limit]

    generated_images = []
    for i in range(0, len(images), n_workers):
        batch = images[i:i + n_workers]
        with Pool(processes=n_workers) as pool:
            generated_batch = pool.starmap(
                apply_to_one_image,
                [
                    ( # Arguments for function call
                        os.path.join(image_path), 
                        get_train_transforms(seed=seed),
                        os.path.join(save_dir, os.path.basename(image_path)) if save_dir else None
                    )
                    for seed, image_path in enumerate(batch)
                ]
            )
        generated_images.extend(generated_batch)

    return generated_images

def apply_to_directory(image_dir: str, transforms: A.Compose, save_dir: str, shuffle: bool = False, limit: int = -1) -> list[str]:
    """
    Apply transforms to all images in a directory and save the results.
    """
    os.makedirs(save_dir, exist_ok=True)

    images = os.listdir(image_dir)
    if len(images) == 0:
        print(f"No images found in {image_dir}: Directory is empty.")
        return

    images = [f for f in images if f.split('.')[-1].lower() in Config.SUPPORTED_IMAGE_TYPES]
    if len(images) == 0:
        print(f"No images found in {image_dir}: No supported image types.")
        return

    if shuffle:
        np.random.shuffle(images)

    if limit > 0:
        images = images[:limit]

    generated_images = []
    for filename in images:
        image_path = os.path.join(image_dir, filename)
        save_path = os.path.join(save_dir, filename)

        saved_image_path = apply_to_one_image(image_path, transforms, save_path)
        generated_images.append(saved_image_path)

    return generated_images

if __name__ == "__main__":
    test_image_path = "/datasets/terumo-data-jpeg/Terumo_Normal_HE/"

    transforms = get_train_transforms()
    
    save_path = "./data/augmented/"
    os.makedirs(save_path, exist_ok=True)
    
    apply_to_directory(test_image_path, transforms, save_path, shuffle=True, limit=5)