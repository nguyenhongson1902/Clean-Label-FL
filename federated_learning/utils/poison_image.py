from .apply_noise_patch import apply_noise_patch
import torch
from torch.utils.data import Dataset


class poison_image(Dataset):
    def __init__(self, dataset,indices,noise,transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        if idx in self.indices:
            image = torch.clamp(apply_noise_patch(self.noise,image,mode='add'),-1,1)
        label = self.dataset[idx][1]
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)