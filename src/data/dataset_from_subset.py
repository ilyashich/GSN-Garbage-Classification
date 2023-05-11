from torch.utils.data import Dataset

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.subset[index]
        image = np.asarray(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
        
    def __len__(self):
        return len(self.subset)