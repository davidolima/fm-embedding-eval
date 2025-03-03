import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

import os
from typing import *

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
        self.classes = classes if classes is not None else \
                ["Membranous", "Sclerosis", "Crescent", "Normal", "Hypercelularidade", "Podocitopatia"]
        

        self.data = GlomerulusDataset.load_data(self.root, self.classes)

    @staticmethod
    def identify_class_folders(root: str, classes: list[str]) -> dict[str, list[str]]:
        folders_per_class = dict.fromkeys(classes)
        for c in classes:
            folders_per_class[c] = [x for x in os.listdir(root) if c in x]
        return folders_per_class

    @staticmethod
    def load_data(root_dir: str, classes: list[str]) -> list[Tuple[str, int]]:
        class_folders = GlomerulusDataset.identify_class_folders(root_dir, classes)

        data = []
        for c, folders in class_folders.items():
            class_images = []
            for folder in folders:
                folder_images = os.listdir(os.path.join(root_dir, folder))
                folder_images = [os.path.join(root_dir, folder, x) for x in folder_images]
                class_images.extend(folder_images)
                
            class_images = [(x, classes.index(c)) for x in class_images]
            data.extend(class_images)

        print(f"[!] Dataset loaded. ({len(data)} images and {len(classes)} classes)")
        return data

    def __repr__(self) -> str:
        return f"<GlomerulusDataset with {self.__len__()} images and {len(self.classes)} classes>"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img, label = self.data[idx]

        img = Image.open(img)
        if self.transforms:
            img = self.transforms(img)

        return (img, label)

if __name__ == '__main__':
    ds = GlomerulusDataset("/datasets/terumo-data-jpeg")

    image, label = ds[0]

    print(image)
    print(label)
