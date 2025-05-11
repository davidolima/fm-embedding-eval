import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from tqdm import tqdm

import os
import glob
from typing import *

from config import Config
from data.augment import get_train_transforms, apply_to_images

class GlomerulusDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        transforms: transforms.Compose | None = None,
        classes: list[str] | None = None,
        balance_dataset: bool = False,
    ):
        """
        Class for Terumo's Glomerulus dataset.
        Args:
            root_dir (str): Path to the root directory containing the dataset.
            transforms (transforms.Compose | None): Transformations to apply to the images.
            classes (list[str] | None): List of classes to use. If None, classes will be auto-detected based on folders found in `root_dir`.
        """
        super().__init__()

        self.root = root_dir
        self.transforms = transforms

        if classes and classes != []:
            self.classes = classes
        else:
            # Folders found in root_dir will be used as classes
            self.classes = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
            print(f"[!] Classes auto-detected: {self.classes}")

        self.data = GlomerulusDataset.load_data(self.root, self.classes)

        if balance_dataset:
            self.balance_classes()

    @staticmethod
    def get_image_from_folders(folder_path:str) -> list[str]:
        """
        Get all images from inside a folder.
        """
        if os.path.isfile(folder_path) or os.listdir(folder_path) == []:
            return []

        images = []
        for path, subfolders, files in os.walk(folder_path):
            # Get all files in the folder
            images.extend([os.path.join(path, x) for x in files if x.endswith(tuple(Config.SUPPORTED_IMAGE_TYPES))])
            for subfolder in subfolders:
                images.extend(GlomerulusDataset.get_image_from_folders(os.path.join(path, subfolder)))

        return images

    @staticmethod
    def iterate_over_class_images(root: str, classes: list[str]) -> dict[str, list[str]]:
        """
        Iterate over all classes and get all images from each class.
        """

        # Get all folders containing <class> in their name
        folders_per_class = dict.fromkeys(classes)
        for c in classes:
            folders_per_class[c] = [x for x in os.listdir(root) if c in x]

        # Get all images from each folder
        for c, folders in folders_per_class.items():
            class_images = []
            for folder in folders:
                images = GlomerulusDataset.get_image_from_folders(os.path.join(root, folder))
                class_images.extend(images)

            yield c, class_images
        
    @staticmethod
    def load_data(root_dir: str, classes: list[str]) -> list[Tuple[str, int]]:
        data = []
        for c, images in GlomerulusDataset.iterate_over_class_images(root_dir, classes):
            class_images = [(x, classes.index(c)) for x in images]
            data.extend(class_images)

        print(f"[!] Dataset loaded. ({len(data)} images and {len(classes)} classes)")
        return data

    def info(self) -> None:
        """
        Print dataset info.
        """
        print(f"Dataset: {self.__class__.__name__}")
        print(f"Number of images: {len(self.data)}")
        [print(f"  - {c}: {n}") for c, n in self.count_images_per_class().items()]
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        print(f"Root directory: {self.root}")

    def count_images_per_class(self) -> dict[str, int]:
        """
        Count the number of images per class.
        """
        count = {c: 0 for c in self.classes}
        for _, label in self.data:
            count[self.classes[label]] += 1
        return count

    def balance_classes(self) -> None:
        """
        Balance the classes in the dataset.
        """
        count = self.count_images_per_class()
        max_count = max(count.values())

        print(f"[!] Balancing each class to {max_count} images.")

        transforms = get_train_transforms()

        for c, n in tqdm(count.items()):
            if n < max_count:
                # Get the difference
                diff = max_count - n
                
                # Get random images from the class
                class_images = [x for x, label in self.data if label == self.classes.index(c)]
                random_images = np.random.choice(class_images, diff, replace=True)
                
                # Apply augmentations to the images
                random_images = apply_to_images(
                    image_paths=random_images,
                    transforms=transforms,
                    save_dir=os.path.join(self.root, f"{c}_augmented"),
                    shuffle=True,
                    limit=diff,
                    n_workers=16,
                )        

                # Add them to the dataset
                self.data.extend([(x, self.classes.index(c)) for x in random_images])
                print(f"[!] Class {c} balanced. ({n} + {len(random_images)} -> {max_count})")
            else:
                print(f"[!] Class {c} already balanced. ({n} images)")

        print(f"[!] Classes balanced. ({len(self.data)} images and {len(self.classes)} classes)")
        self.info()

    def __repr__(self) -> str:
        return f"<GlomerulusDataset with {self.__len__()} images and {len(self.classes)} classes>"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.data[idx]

        img = np.array(Image.open(img_path))
        if self.transforms:
            img = self.transforms(img)

        return (img, label, img_path)

if __name__ == '__main__':
    classes = ["Crescent", "Hypercelularidade", "Membranous", "Normal", "Podocitopatia", "Sclerosis"]
    ds = GlomerulusDataset("/datasets/terumo-data-jpeg", classes=classes)
    print(ds)

    image, label, path = ds[0]

    ds.info()
    ds.balance_classes()
    ds.info()