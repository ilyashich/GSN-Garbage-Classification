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

    def __init__(self,  batch_size=64, image_size=224, data_dir = "./data/dataset-resized", mean_norm = (0.6732, 0.6399, 0.6049), std_norm=(0.2062, 0.2072, 0.2293)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.mean_norm = mean_norm
        self.std_norm = std_norm

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
        train_subset, val_subset, test_subset = random_split(trashnet_dataset, [2023, 252, 252], generator=torch.Generator().manual_seed(42))
        
        if stage == 'fit' or stage is None:
            self.dataset_train = DatasetFromSubset(train_subset, transform=self.get_train_transform())
            self.dataset_val = DatasetFromSubset(test_subset, transform=self.get_val_test_transform())
            
        if stage == 'test' or stage is None:
            self.dataset_test = DatasetFromSubset(val_subset, transform=self.get_val_test_transform())
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    def get_train_transform(self):
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=int(self.image_size*1.1)),
                A.RandomCrop(height=self.image_size, width=self.image_size),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=0.5),
                A.Flip(p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=1, max_height=96, max_width=96),
                A.Normalize(mean=self.mean_norm, std=self.std_norm),
                ToTensorV2(),
            ]
        )
    
    def get_val_test_transform(self):
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=int(self.image_size*1.1)),
                A.CenterCrop(height=self.image_size, width=self.image_size),
                A.Normalize(mean=self.mean_norm, std=self.std_norm),
                ToTensorV2()
            ]
        )

    def is_valid_image(self, path):
        return cv2.imread(path) is not None