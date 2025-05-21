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
        balance_dataset: bool = False,
        transforms: Optional[transforms.Compose] = None,
        classes: Optional[list[str]] = None,
        one_vs_all: Optional[str] = None,
        consider_augmented: bool | Literal['others_only', 'positive_only'] = True,
    ):
        """
        Class for Terumo's Glomerulus dataset.
        Args:
            root_dir (str): Path to the root directory containing the dataset.
            balance_dataset (bool): Whether to balance the dataset or not.
            transforms (Optional[transforms.Compose]): Transformations to apply to the images.
            classes (Optional[list[str]]): List of classes to use. If None, classes will be auto-detected based on folders found in `root_dir`.
            one_vs_all (Optional[str]): Class to use against all others for binary classification. If None, all classes will be used.
            consider_augmented (bool | Literal['others_only', 'positive_only']): Whether to consider augmented images or not. 
                Use 'False' to ignore all augmented images, 'True' to consider all augmented images,
                'positive_only' to only consider augmented images for the positive class,
                or 'others_only' to only consider augmented images for all classes except the `one_vs_all` class.
        """
        super().__init__()

        self.root = root_dir
        self.transforms = transforms
        self.one_vs_all = one_vs_all
        self.consider_augmented = consider_augmented

        if classes and classes != []:
            self.classes = classes
        elif self.root != '':
            # Folders found in root_dir will be used as classes
            self.classes = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
            print(f"[!] Classes auto-detected: {self.classes}")


        if self.root != '':
            self.data = GlomerulusDataset.load_data(self.root, self.classes, one_vs_all=self.one_vs_all, consider_augmented=consider_augmented)
        else:
            self.data = []

        if self.one_vs_all:
            self.classes = ['others', self.one_vs_all]

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
    def load_data(root_dir: str, classes: list[str], one_vs_all: Optional[str] = None, consider_augmented: bool | Literal['others_only', 'positive_only'] = True) -> list[Tuple[str, int]]:
        data = []
        for c, images in GlomerulusDataset.iterate_over_class_images(root_dir, classes):
            consider_augmented_for_this_class = True

            # Figure out if we should consider augmented images for this class
            if isinstance(consider_augmented, str):
                assert(one_vs_all is not None), "Parameter `one_vs_all` must be set if consider_augmented `others_only` or `positive_only` is used."
                # If I'm not in the `one_vs_all` class and `consider_augmented` is `others_only`, I will consider augmented images
                consider_augmented_for_this_class = (consider_augmented == 'others_only' and c != one_vs_all)
                # If I'm in the `one_vs_all` class and `consider_augmented` is `positive_only`, I will consider augmented images
                consider_augmented_for_this_class = (consider_augmented == 'positive_only' and c == one_vs_all) or consider_augmented_for_this_class 
            elif isinstance(consider_augmented, bool):
                consider_augmented_for_this_class = consider_augmented
                    
            if not consider_augmented_for_this_class: # Do not consider augmented images for this class
                images = [x for x in images if 'augmented' not in x]

            if one_vs_all:
                class_images = [(x, int(c == one_vs_all)) for x in images] # Positive class is 1, negative class is 0
            else:
                class_images = [(x, classes.index(c)) for x in images]

            data.extend(class_images)

        if one_vs_all:
            classes = ['others', one_vs_all]

        print(f"[!] Dataset loaded. ({len(data)} images and {len(classes)} classes)")
        return data

    def info(self) -> None:
        """
        Print dataset info.
        """
        print(f"Dataset: {self.__class__.__name__}", f"[{self.one_vs_all} vs All]" if self.one_vs_all else "")
        print(f"Number of images: {len(self.data)}")
        count = self.count_images_per_class()
        if self.one_vs_all:
            print(f"  + Positive class ({self.one_vs_all}): {count[self.one_vs_all]}")
            print(f"  - Negative class (Others): {count['others']}")
        else:
            [print(f"  - {self.classes[c]}: {n}") for c, n in count.items()]
        print(f"Number of classes: {len(self.classes)}")
        print(f"Root directory: {self.root}")
        print("Considering augmented images:", self.consider_augmented)

    def count_images_per_class(self, count_augmented: bool = True) -> dict[str, int]:
        """
        Count the number of images per class.
        """
        if self.one_vs_all:
            count = {self.one_vs_all: 0, 'others': 0}
        else:
            count = {i: 0 for i in range(len(self.classes))}

        for img_path, label in self.data:
            if not count_augmented and 'augmented' in img_path:
                continue
            if self.one_vs_all:
                label = self.one_vs_all if label == 1 else 'others' 

            count[label] += 1 
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

            for label, imgs in splits[f"split_{i}"].items():
                for img in imgs:
                    if self.one_vs_all:
                        label = 1 if label == self.one_vs_all else 0
                    else:
                        label = self.classes.index(label)
                        
                    self.data.append((img, label))
                        

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

                    if image_name in os.listdir(label_dir):
                        k = 1
                        image_name = image_name.split('.')[0] + f"_{k}." + image_name.split('.')[-1]
                        while image_name in os.listdir(label_dir):
                            k += 1 
                            image_name = image_name.split('.')[0] + f"_{k}." + image_name.split('.')[-1]

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
    #splits_folder = "/datasets/terumo-splits-augmented/"
    #for cls in classes:
    #    for qty_splits in list(range(5,11)):
    #        train = GlomerulusDataset("/datasets/terumo-data-jpeg/", classes=classes, one_vs_all=cls, consider_augmented='positive_only')
    #        train.generate_cross_validation_splits(qty_splits, out_dir=os.path.join(splits_folder, f'{cls}_vs_all', f'{qty_splits}_splits'))
        
    # val = GlomerulusDataset("/datasets/terumo-data-jpeg/", classes=classes, one_vs_all="Crescent", consider_augmented='positive_only')

    train.info()
    # val.info()

    # for cls in classes:
    #     train._balance_one_class_vs_others(
    #         class_name=cls,
    #         transforms=get_train_transforms(),
    #         save_results=True,
    #         count_aug_for_others=False,
    #         n_workers=4
    #     )
        

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
