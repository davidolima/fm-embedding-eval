import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
import numpy as np

import os
import glob
from typing import *

from config import Config

class GlomerulusDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        transforms: transforms.Compose | None = None,
        classes: list[str] | None = None,
    ):
        """
        Class for Terumo's Glomerulus dataset.
        params:
         - root_dir: Dataset system path.
         - classes (Default: None): List of classes to search for in `root_dir`.
                                    If None, will infer based on folders found in `root_dir`.
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

    @staticmethod
    def get_image_from_folders(folder_path:str) -> list[str]:
        """
        Get all images from a folder.
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

            if c == 'Hypercelularidade':
                print(images)

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
    ds = GlomerulusDataset("/datasets/terumo-val")
    print(ds)

    image, label, path = ds[0]

    ds.info()
    #print(image)
    #print(label)
