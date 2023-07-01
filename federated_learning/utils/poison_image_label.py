from torch.utils.data import Dataset
import torch
from .apply_noise_patch import apply_noise_patch


class poison_image_label(Dataset):
    def __init__(self, dataset, indices, noise, target, transform, patch_mode):
        """
        The function initializes an object with the given dataset, indices, noise, target, transform,
        and patch mode.
        
        :param dataset: The dataset parameter is the input dataset that you want to apply the patch to.
        It could be an image dataset, text dataset, or any other type of dataset that you want to modify
        :param indices: The `indices` parameter is a list of indices that specifies which samples from
        the dataset should be used. It allows you to select a subset of the dataset to work with
        :param noise: It is the optimized noise generated from a clean-label backdoor attack
        :param target: The target parameter is used to specify the desired output label for the dataset.
        It is used to flip all the target examples into the target label for computing ASR
        :param transform: The `transform` parameter is used to apply transformations to the data. It is
        typically used to preprocess the data or apply data augmentation techniques. Transformations can
        include resizing, cropping, rotating, flipping, normalizing, etc.
        :param patch_mode: The `patch_mode` parameter is used to specify how the patches should be
        extracted from the dataset. It determines the size and shape of the patches that will be used
        for training or testing. The available options for `patch_mode` could be "add", or "change"
        """
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target
        self.transform = transform
        self.patch_mode = patch_mode

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        # image = torch.clamp(apply_noise_patch(self.noise, image, mode='add'), -1, 1)
        image = torch.clamp(apply_noise_patch(self.noise, image, mode=self.patch_mode), -1, 1)
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.target) # applied trigger to image and return (image, target class)

    def __len__(self):
        return len(self.indices)