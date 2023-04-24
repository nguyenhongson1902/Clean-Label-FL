from torch.utils.data import Dataset
import torch
from .apply_noise_patch import apply_noise_patch


class poison_image_label(Dataset):
    def __init__(self, dataset,indices,noise,target,transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        image = torch.clamp(apply_noise_patch(self.noise, image, mode='add'), -1, 1)
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.target) # applied trigger to image and return (image, target class)

    def __len__(self):
        return len(self.indices)