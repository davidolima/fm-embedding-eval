import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

import os
import json
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

    def count_images_per_class(self, count_augmented: bool = True) -> dict[str, int]:
        """
        Count the number of images per class.
        """
        count = {c: 0 for c in self.classes}
        if count_augmented:
            for _, label in self.data:
                count[self.classes[label]] += 1
        else:
            for img_path, label in self.data:
                if 'augmented' not in img_path:
                    count[self.classes[label]] += 1
        return count

    def _augment_class(self, class_name: str, n_images: int, save_dir: Optional[str] = None, transforms: None | A.Compose | transforms.Compose = None, n_workers: int = 4) -> None:
        """
        Apply transforms to a specified class' images and 
        """
        
        if transforms is None:
            transforms = get_train_transforms()

        generated_images = []
        class_images = [x for x, label in self.data if label == self.classes.index(class_name)]
        while len(generated_images) < n_images:
            qty_to_add = n_images - len(generated_images)
            if qty_to_add <= 0:
                break

            generated_batch = apply_to_images(
                image_paths=class_images,
                transforms=transforms,
                save_dir=save_dir if save_dir is not None else None,
                shuffle=True,
                limit=qty_to_add,
                n_workers=4,
            )
            print("[+] Generated", len(generated_batch), f"images for class {class_name}. {qty_to_add} images left...")
            generated_images.extend((x, self.classes.index(class_name)) for x in generated_batch)

        # Add them to the dataset
        self.data.extend(generated_images)

    def _balance_all_classes_equally(self, transforms: A.Compose | transforms.Compose, save_results: bool = False, n_workers: int = 4) -> None:
        """
        Make all classes have the same number of images.
        """
        count = self.count_images_per_class()
        max_count = max(count.values())

        print(f"[!] Balancing each class to {max_count} images.")
        for c, n in tqdm(count.items()):
            if n < max_count:
                self._augment_class(
                    class_name = c, 
                    n_images   = max_count - n,
                    transforms = transforms,
                    save_dir   = os.path.join(self.root, f"{c}_augmented") if save_results else None,
                    n_workers  = n_workers,
                )        
                print(f"[!] Class {c} balanced. ({n} + {len(random_images)} -> {max_count})")

            else:
                print(f"[!] Class {c} already balanced. ({n} images)")

        print(f"[!] Classes balanced. ({len(self.data)} images and {len(self.classes)} classes)")

    def _balance_one_class_vs_others(self, class_name: str, transforms: A.Compose | transforms.Compose, save_results: bool = False, count_aug_for_others: bool=True, n_workers: int = 4) -> None:
        """
        Balance one class against all others.

        Args:
            class_name (str): Class to balance against all others.
            transforms (A.Compose | transforms.Compose): Transformations to apply to the images.
            save_results (bool): Whether to save the augmented images or not.
            count_aug_for_others (bool): Whether to count augmented images for other classes or not. 
                                         Useful for generating one vs all balancing for all classes.
            n_workers (int): Number of workers to use for augmentations.

        Ex.: Balance Class1 against others.
           - Class1: x images    ->  Class1: x + (x - (y+z)) images
           - Class2: y images    ->  Class2: y images 
           - Class3: z images    ->  Class3: z images
        """
        if class_name not in self.classes:
            raise ValueError(f"[!] Class {class_name} not found in the dataset. Available classes: {self.classes}")
        
        qty_images_in_class = self.count_images_per_class()[class_name]
        images_per_class = self.count_images_per_class(count_augmented=count_aug_for_others)

        qty_images_in_others = sum([images_per_class[c] for c in self.classes if c != class_name])

        diff = qty_images_in_others - qty_images_in_class
        if diff <= 0:
            print(f"[!] Class {class_name} already balanced. ({qty_images_in_class} images)")
            return

        print(f"[!] Balancing class {class_name} ({qty_images_in_class} images) against all others ({qty_images_in_others} images). Adding {diff} images.")
        self._augment_class(
            class_name = class_name, 
            n_images   = diff,
            transforms = transforms,
            save_dir   = os.path.join(self.root, f"{class_name}_one_vs_all_augmented") if save_results else None,
            n_workers  = n_workers,
        )

    def balance_classes(self, one_vs_all: str = None, save_results: bool = False, n_workers: int = 4) -> None:
        """
        Balance the classes in the dataset.
        Args:
            one_vs_all (str): Class to balance against all others. If None, all classes will be balanced equally.
            save_results (bool): Whether to save the augmented images or not.
            n_workers (int): Number of workers to use for augmentations.
        """

        transforms = get_train_transforms()

        if one_vs_all:
            self._balance_one_class_vs_others(
                class_name=one_vs_all,
                transforms=transforms,
                save_results=save_results,
                n_workers=n_workers,
                count_aug_for_others=True,
            )            
        else:
            self._balance_all_classes_equally(
                transforms=transforms,
                save_results=save_results,
                n_workers=n_workers
            )

        self.info()

    def load_splits_from_json(self, split_no: int | list[int], json_path: str, clear_data: bool = True) -> None: 
        """
        Load cross-validation splits from a JSON file.
        Args:
            json_path (str): Path to the JSON file produced by `generate_cross_validation_splits` containing the splits.
        """

        with open(json_path, 'r') as f:
            splits = json.load(f)
                 
        if isinstance(split_no, int):
            split_no = [split_no]
        
        if clear_data:
            self.data = []
        
        for i in split_no:
            if f"split_{i}" not in splits:
                raise ValueError(f"[!] Split {i} not found in `{json_path}`. Available splits: {splits.keys()}")

            self.classes.extend([x for x in splits[f"split_{i}"].keys() if x != 'splits' and x not in self.classes])
            for label, imgs in splits[f"split_{i}"].items():
                for img in imgs:
                    self.data.append((img, self.classes.index(label)))

        #print(f"[!] Loaded {len(splits)} cross-validation splits from `{json_path}`.")

    def generate_cross_validation_splits(self, n_splits: int = 5, out_dir: str = None) -> list[list[str]]:
        """
        Generate cross-validation splits for the dataset.
        Args:
            n_splits (int): Number of splits to generate.
        """

        n_images = len(self.data)
        n_images_per_split = n_images // n_splits
        
        # Shuffle and separate image indices into splits
        indices = np.arange(n_images)
        np.random.shuffle(indices)

        splits = []
        for i in range(n_splits):
            start = i*n_images_per_split
            end = (i+1)*n_images_per_split if i != n_splits-1 else n_images
            splits.append(indices[start:end])
        assert(sum(len(split) for split in splits) == n_images), "Error: Not all images are included in the splits."

        # Transform indices in file paths
        for idx, split in enumerate(splits):
            splits[idx] = [self.data[i] for i in split]

        print(f"[!] Generated {n_splits} cross-validation splits of size {n_images_per_split}.")
        for i, split in enumerate(splits):
            print(f"  - Split {i+1}: {len(split)} images")

        if out_dir: # Create splits in `out_dir` using symlinks
            print(f"[!] Saving splits to `{out_dir}`.")
            os.makedirs(out_dir, exist_ok=True)

            splits_info = {} 

            for split_idx, split_images in enumerate(splits):
                split_dir = os.path.join(out_dir, f"split_{split_idx+1}")
                os.makedirs(split_dir, exist_ok=True)
                
                splits_info[ f"split_{split_idx+1}" ] = {self.classes[c]: [] for c in range(len(self.classes))}

                # Create symlink to images in splits
                for image_path, label in split_images:
                    label_dir = os.path.join(split_dir, str(self.classes[label]))
                    os.makedirs(label_dir, exist_ok=True)

                    splits_info[f"split_{split_idx+1}"][ self.classes[label] ].append(image_path)

                    image_name = os.path.basename(image_path)                    
                    # HACK: Augmented images should have a different name than the original file
                    if 'augmented' in image_path:
                        fname, ext = image_name.split('.') # Consider images that include '.' ?
                        image_name = f"{fname}_augmented.{ext}"

                    os.symlink(image_path, os.path.join(label_dir, image_name))

                #print(f"[!] Created {len(split_images)} symbolic links for split {split_idx+1} in `{split_dir}`.")

            with open(os.path.join(out_dir, 'splits_info.json'), 'w+') as f:
                f.write(json.dumps(splits_info))

        return splits

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
    train = GlomerulusDataset("/datasets/terumo-data-jpeg/", classes=classes)
    #val = GlomerulusDataset("/datasets/terumo-data-jpeg/", classes=classes)

    train.info()
    #val.info()
    
    for cls in classes:
        train._balance_one_class_vs_others(
            class_name=cls,
            transforms=get_train_transforms(),
            save_results=True,
            count_aug_for_others=False,
            n_workers=4
        )
        

    # for i in range(5):
    #     val_split = i+1
    #     train_splits = list(range(1, 6))
    #     train_splits.remove(val_split)

    #     val.load_splits_from_json(val_split, "/datasets/terumo-splits-augmented/10_splits/splits_info.json", clear_data=True)
    #     train.load_splits_from_json(train_splits, "/datasets/terumo-splits-augmented/10_splits/splits_info.json", clear_data=True)

    #     print("--" * 20, "Train", "--" * 20)
    #     train.info()
    #     print("--" * 20, "Validation", "--" * 20)
    #     val.info()

    # splits_folder = "/datasets/terumo-splits-augmented/"
    # for qty_splits in list(range(5,11)):
    #     ds.generate_cross_validation_splits(qty_splits, out_dir=os.path.join(splits_folder,f'{qty_splits}_splits'))