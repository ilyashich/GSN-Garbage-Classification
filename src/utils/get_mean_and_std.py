import torch
from torchvision import transforms, datasets


def get_mean_and_std(data_dir="./dataset-resized"):
    raw_dataset = datasets.ImageFolder(root=data_dir, transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])) 
    raw_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=32)
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in raw_loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print("Mean: ", mean)
    print("Std: ", std)

    return mean, std