import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pytorch_lightning as pl

import subprocess

from src.data.dataset_from_subset import DatasetFromSubset



class TrashNetDataModule(pl.LightningDataModule):

    def __init__(self,  batch_size=128, model_version="B0", data_dir = "./data/", mean_norm = (0.6732, 0.6399, 0.6049), std_norm=(0.2062, 0.2072, 0.2293), generator_seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_sizes = {
            "B0": (224, 256),
            "B1": (240, 276),
            "B2": (260, 296),
            "B3": (224, 256),
            "B4": (224, 256),
            "B5": (224, 256),
            "B6": (224, 256),
            "B7": (224, 256)
        }
        self.crop_image_size = self.image_sizes[model_version][0]
        self.scale_image_size = self.image_sizes[model_version][1]
        
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.generator_seed = generator_seed

    def prepare_data(self):
        # download only
        url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
        subprocess.run(["wget",  "-nc", "-P", self.data_dir, url])
        subprocess.run(["unzip",  "-qn",  self.data_dir + "dataset-resized.zip", "-d", self.data_dir])
        subprocess.run(["rm",  "-rf",  self.data_dir + "__MACOSX/"])

    def setup(self, stage=None):
        # called on every GPU
        # use our dataset and defined transformations
        trashnet_dataset = ImageFolder(root=self.data_dir + "dataset-resized", is_valid_file=self.is_valid_image)
        train_subset, val_subset, test_subset = random_split(trashnet_dataset, [2023, 252, 252], generator=torch.Generator().manual_seed(self.generator_seed))
        
        if stage == 'fit' or stage is None:
            self.dataset_train = DatasetFromSubset(train_subset, transform=self.get_train_transform())
            self.dataset_val = DatasetFromSubset(val_subset, transform=self.get_val_test_transform())
            
        if stage == 'test' or stage is None:
            self.dataset_test = DatasetFromSubset(test_subset, transform=self.get_val_test_transform())
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    def get_train_transform(self):
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.scale_image_size),
                A.RandomCrop(height=self.crop_image_size, width=self.crop_image_size),
                A.Rotate(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.HueSaturationValue(),
                A.FancyPCA(),
                A.Perspective(),
                #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                #A.RandomBrightnessContrast(p=0.5),
                #A.CoarseDropout(max_holes=1, max_height=96, max_width=96),
                A.Normalize(mean=self.mean_norm, std=self.std_norm),
                ToTensorV2(),
            ]
        )
    
    def get_val_test_transform(self):
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.scale_image_size),
                A.CenterCrop(height=self.crop_image_size, width=self.crop_image_size),
                A.Normalize(mean=self.mean_norm, std=self.std_norm),
                ToTensorV2()
            ]
        )

    def is_valid_image(self, path):
        return cv2.imread(path) is not None