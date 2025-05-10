import os
import random

import torch 
from torchvision.utils import save_image

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

from config import Config
import cv2

def get_train_transforms():
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
    ])
    return albumentations_t


def get_test_transforms():
    albumentations_t = A.Compose([
        A.pytorch.ToTensorV2(),
    ])
    return albumentations_t

def apply_to_one_image(image_path: str, transforms: A.Compose, save_path: str) -> str:
    """
    Apply transforms to an image and save the result.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    augmented_image = transforms(image=image)    
    augmented_image = augmented_image['image']

    save_image(augmented_image.float() / 255.0, save_path)  # Normalize to [0, 1]

    return save_path

def apply_to_images(image_paths: list[str], transforms: A.Compose, save_dir: str, shuffle: bool = False, limit: int = -1) -> list[str]:
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
    for image_path in tqdm(images):
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, filename)

        saved_image_path = apply_to_one_image(image_path, transforms, save_path)
        generated_images.append(saved_image_path)

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