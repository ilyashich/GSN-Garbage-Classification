import copy
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.trashnet_datamodule import TrashNetDataModule

def visualize_train_augmentations(dataset, num_imgs=10, samples=10, cols=5):
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    
    for idx in range(num_imgs):
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data_module = TrashNetDataModule()
    data_module.prepare_data()
    data_module.setup()

    visualize_train_augmentations(dataset=data_module.dataset_train)
    